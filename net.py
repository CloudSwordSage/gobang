# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = x + y
        return F.relu(y)

class Net(nn.Module):
    def __init__(self, num_channels=256, num_res_blocks=7):
        super(Net, self).__init__()
        # 预处理
        self.conv1 = nn.Conv2d(4, num_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        # 特征提取
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        # 策略网络
        self.policy_conv = nn.Conv2d(num_channels, 16, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 15 * 15, 225)
        # 价值网络
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 15 * 15, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_x = policy_x.view(policy_x.size(0), -1)
        policy_x = self.policy_fc(policy_x)
        value_x = F.relu(self.value_bn(self.value_conv(x)))
        value_x = value_x.view(value_x.size(0), -1)
        value_x = self.value_fc1(value_x)
        value_x = self.value_fc2(value_x)
        return F.softmax(policy_x, dim=1), F.tanh(value_x)

class PolicyValueNet:
    def __init__(self, model_file=None, device=torch.device("cuda")):
        self.device = device
        self.l2_const = 2e-3
        self.policy_value_net = Net().to(device)
        self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=1e-3, weight_decay=self.l2_const)
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))
        
    def policy_value(self, states):
        self.policy_value_net.eval()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        log_act_probs, value =  self.policy_value_net(states)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        return act_probs, value.detach().cpu().numpy()

    def policy_value_fn(self, env):
        self.policy_value_net.eval()
        legal_position = env.availables
        current_state = np.ascontiguousarray(env.current_state().reshape(-1, 4, 15, 15)).astype(np.float32)
        current_state = torch.tensor(current_state, dtype=torch.float32).to(self.device)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
        act_probs = zip(legal_position, act_probs[legal_position])
        return act_probs, value.detach().cpu().numpy()

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)
    
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=2e-3):
        self.policy_value_net.train()
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32).to(self.device)
        winner_batch = torch.tensor(winner_batch, dtype=torch.long).to(self.device)
        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, (-1,))
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(mcts_probs * log_act_probs, axis=1)
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(log_act_probs * torch.log(log_act_probs + 1e-5), axis=1)
        return loss.item(), entropy.numpy()[0]


if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    test_input = torch.rand(8, 4, 15, 15).to(device)
    x, y = model(test_input)
    print(x.shape)
    print(y.shape)
    summary(model, (4, 15, 15), device='cuda')

