import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

def find_pos(x, y):
    for i in range(27, 670, 44):
        for j in range(27, 670, 44):
            L1 = i - 22
            L2 = i + 22
            R1 = j - 22
            R2 = j + 22
            if x >= L1 and x <= L2 and y >= R1 and y <= R2:
                return i, j
    return -1, -1

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(15 * 15, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 15 * 15)
    
    def forward(self, x):
        x = x.view(-1, 15 * 15)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

