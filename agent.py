import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
import gym
from model import *
from memory import ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

LR = 1e-4
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY
GAMMA = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.steps_done = 0

        self.policy_net = DQN(action_size).to(device)
        self.target_net = DQN(action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.icm = ICMModel(state_size, action_size)
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.ce = nn.CrossEntropyLoss()
        self.forward_mse = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.icm.parameters()),
                                    lr=LR)

        # Replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.batch_size = 32 
        self.eta = 0.01


    def select_action(self, state):
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END)* \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state.to(device)).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        """
        zip(*transitions) unzips the transitions into
        Transition(*) creates new named tuple
        batch.state - tuple of all the states (each state is a tensor)
        batch.next_state - tuple of all the next states (each state is a tensor)
        batch.reward - tuple of all the rewards (each reward is a float)
        batch.action - tuple of all the actions (each action is an int)    
        """
        batch = Transition(*zip(*transitions))
        
        actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)
        
        non_final_next_states = torch.cat([s for s in batch.next_stateif s is not None]).to(device)
    
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # --------------------------------------------------------------------------------
        # for Curiosity-driven
        action_onehot = torch.FloatTensor(self.batch_size, self.action_size)
        action_onehot.zero_()
        action_onehot.scatter_(1, action_batch.view(-1, 1), 1)
        real_next_state_feature, pred_next_state_feature, pred_action = self.icm([state_batch, non_final_next_states, action_onehot])
        inverse_loss = self.ce(pred_action, torch.squeeze(action_batch))
        forward_loss = self.forward_mse(pred_next_state_feature, real_next_state_feature.detach())
        # ---------------------------------------------------------------------------------

        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        dqn_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = dqn_loss + forward_loss + inverse_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    ## intrinsic reward in progress...
    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)

        action_onehot = torch.FloatTensor(len(action), self.action_size)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm([state, next_state, action_onehot])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()
