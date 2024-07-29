import math
import random
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import GoBangEnv
from utils import DQN, ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.system('cls')

print(f'    Using {device} device.')

env = GoBangEnv()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 0.00025
print(f'''    BATCH_SIZE: {BATCH_SIZE}
    GAMMA: {GAMMA}
    EPS_START: {EPS_START}
    EPS_END: {EPS_END}
    EPS_DECAY: {EPS_DECAY}
    TAU: {TAU}
    LR: {LR}\n''')

print('    Creating policy network...')
black_policy_net = DQN().to(device)
print('    Creating target network...')
black_target_net = DQN().to(device)

print('    Creating policy network...')
white_policy_net = DQN().to(device)
print('    Creating target network...')
white_target_net = DQN().to(device)

black_target_net.load_state_dict(black_policy_net.state_dict())
black_target_net.eval()

white_target_net.load_state_dict(white_policy_net.state_dict())
white_target_net.eval()

black_optimizer = optim.Adam(black_policy_net.parameters(), lr=LR)
print('    Creating memory...')
black_memory = ReplayMemory(10000)
print(f'    memory size: 10000')
white_optimizer = optim.Adam(white_policy_net.parameters(), lr=LR)
white_memory = ReplayMemory(10000)

black_steps_done = 0
def black_select_action(state):
    global black_steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * black_steps_done / EPS_DECAY)
    black_steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return black_policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 224)]], device=device, dtype=torch.long)

white_steps_done = 0
def white_select_action(state):
    global white_steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * white_steps_done / EPS_DECAY)
    white_steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return white_policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 224)]], device=device, dtype=torch.long)

black_episode_durations = []

def black_optimize_model():
    if len(black_memory) < BATCH_SIZE:
        return 
    transitions = black_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = black_policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = black_target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    black_optimizer.zero_grad()
    loss.backward()
    for param in black_policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    black_optimizer.step()

white_episode_durations = []

def white_optimize_model():
    if len(white_memory) < BATCH_SIZE:
        return 
    transitions = white_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = white_policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = white_target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    white_optimizer.zero_grad()
    loss.backward()
    for param in white_policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    white_optimizer.step()

num_episodes = 600
print(f'    num_episodes: {num_episodes}')
print('-'*110)
episode_tar = tqdm(total=num_episodes, desc='Training', unit='episodes', leave=False)
score_list = []
avg_score_list = []
player = 1

for i_episode in range(num_episodes):
    state = env.state()
    state = torch.tensor(state, device=device).float().view(-1)
    score = 0
    for t in count():
        # black
        if player == 1:
            action = black_select_action(state)
            pos = np.unravel_index(action.cpu(), (15, 15))
            x, y = pos[0].item(), pos[1].item()
            pos = (x, y)
            env.render(player, pos)
            state, reward, done, _, illegal = env.step(player, pos)
            state = torch.tensor(state, device=device).float().view(-1)
            score += reward
            if (not done):
                next_state = env.state()
                next_state = torch.tensor(next_state, device=device).float().view(-1)
            else:
                next_state = None
            reward = torch.tensor([reward], device=device)
            black_memory.push(state, action, next_state, reward)
            if illegal:
                continue
            player = 2
            state = next_state
            black_optimize_model()
            black_target_net_state_dict = black_target_net.state_dict()
            black_policy_net_state_dict = black_policy_net.state_dict()
            for key in black_policy_net_state_dict:
                black_target_net_state_dict[key] = black_policy_net_state_dict[key]*TAU + black_target_net_state_dict[key]*(1-TAU)
            black_target_net.load_state_dict(black_target_net_state_dict)
            if done:
                black_episode_durations.append(t + 1)
                env.reset()
                break
        # white
        action = white_select_action(state)
        pos = np.unravel_index(action.cpu(), (15, 15))
        x, y = pos[0].item(), pos[1].item()
        pos = (x, y)
        env.render(player, pos)
        state, reward, done, _, illegal = env.step(player, pos)
        state = torch.tensor(state, device=device).float().view(-1)
        score += reward
        if (not done):
            next_state = env.state()
            next_state = torch.tensor(next_state, device=device).float().view(-1)
        else:
            next_state = None
        reward = torch.tensor([reward], device=device)
        white_memory.push(state, action, next_state, reward)
        if illegal:
            continue
        player = 1
        state = next_state
        white_optimize_model()
        white_target_net_state_dict = white_target_net.state_dict()
        white_policy_net_state_dict = white_policy_net.state_dict()
        for key in white_policy_net_state_dict:
            white_target_net_state_dict[key] = white_policy_net_state_dict[key]*TAU + white_target_net_state_dict[key]*(1-TAU)
        white_target_net.load_state_dict(white_target_net_state_dict)
        if done:
            white_episode_durations.append(t + 1)
            env.reset()
            break
    score_list.append(score)
    avg_score = sum(score_list[-10:])/len(score_list[-10:])
    avg_score_list.append(avg_score)
    episode_tar.set_postfix(Duration=t+1, score=score)
    episode_tar.update()
print('Complete')
torch.save(black_policy_net, './model/black_dqn-policy-whole.pth')
torch.save(white_target_net, './model/white_dqn-target-whole.pth')
episode_tar.close()
plt.figure(1)
plt.title('Result')
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(score_list, label='score', color='blue')
plt.plot(avg_score_list, label='average score', color='red')
plt.legend()
plt.show()