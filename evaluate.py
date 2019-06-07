from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import torch
from model import *

parser = argparse.ArgumentParser(description="Process parameters for experiment")
parser.add_argument("--env_field", type=str, default='configs/exampleTraining.yaml')
args = parser.parse_args()

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def convert_action(action):
    actions_array = np.array([[0,0],[0,1],[0,2],[1,0], [1,1],[1,2], [2,0],[2,1],[2,2]])
    return actions_array[action]

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_path = 'env/AnimalAI'
    brain_name='Learner'
    arena_config_in = ArenaConfig(args.env_field)

    model = DQN(action_size=9).to(device)
    model.eval()
    model.load_state_dict(torch.load("./models/dqn/dqn.pt"))

    env=UnityEnvironment(file_name=env_path)
    #環境リセット
    action_info = env.reset(arenas_configurations_input=arena_config_in, train_mode=False)
    obs = action_info[brain_name].visual_observations[0][0]
    state = get_state(obs)

    for step in range(1000):
        time.sleep(0.05)
        #ランダム行動
        action_values = model(state)
        action = np.argmax(action_values.cpu().data.numpy())
        conv_action = convert_action(action)
    
        action_info = env.step(conv_action)
        obs = action_info[brain_name].visual_observations[0][0]
        reward = action_info[brain_name].rewards[0]
        done   = action_info[brain_name].local_done[0]
        max_reach=action_info[brain_name].max_reached
        next_state = get_state(obs)
        state = next_state
        #表示
        #print('\n ===== {} step ======'.format(step))
        #print('\naction=', action)
        #print('\nstate=', state.shape)
        #print('\nreward=', reward)
        #print('\ndone=', done)
        #print('\nmax_reach=', max_reach)

    #plt.imshow(state[0][0])
    #plt.show()
