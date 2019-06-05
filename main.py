import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
from collections import deque
import time
import gym
import argparse
import agent
from animalai.envs.arena_config import ArenaConfig
from animalai.envs import UnityEnvironment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

parser = argparse.ArgumentParser(description="Process parameters for experiment")
parser.add_argument("--env_field", type=str, default='configs/exampleTraining.yaml') # 'configs/movingFood.yaml'が動かない.....。
args = parser.parse_args()

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def convert_action(action):
    actions_array = np.array([[0,0],[0,1],[0,2],[1,0], [1,1],[1,2], [2,0],[2,1],[2,2]])
    return actions_array[action]

def train(env, n_episodes, render=False):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    score = 0

    for episode in range(n_episodes):
        action_info = env.reset(arenas_configurations_input=arena_config_in)
        obs = action_info[brain_name].visual_observations[0][0]
        state = get_state(obs)
        #total_reward = 0.0
        for t in range(100):
            action = agent.select_action(state)
            conv_action = convert_action(action)

            action_info = env.step(conv_action)
            obs = action_info[brain_name].visual_observations[0][0]
            reward = action_info[brain_name].rewards[0]
            score += reward
            done   = action_info[brain_name].local_done[0]

            #total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            agent.memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if agent.steps_done > INITIAL_MEMORY:
                agent.optimize_model()

                if agent.steps_done % TARGET_UPDATE == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

            if done:
                break
        scores_window.append(score)
        scores.append(score)
        if episode % 20 == 0:
                #print('Total steps: {} \t Episode: {}\t Total reward: {}'.format(agent.steps_done, episode, total_reward))
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))

    env.close()
    return

if __name__ == '__main__':
    env_path = 'env/AnimalAI'
    brain_name='Learner'
    train_mode=True
    arena_config_in = ArenaConfig(args.env_field)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    agent = agent.Agent(action_size=9)
    env=UnityEnvironment(file_name=env_path) 
    # train model
    train(env, 400)