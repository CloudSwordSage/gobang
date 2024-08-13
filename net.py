# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast

from typing import *

from game import Board

class ResBlock(nn.Module):
    def __init__(self, num_filters: int=256) -> None:
        """
        初始化ResBlock类
        
        Args:
            num_filters (int, optional): 卷积层的滤波器数量，默认为256。
        
        Returns:
            None
        
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters, )
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量x进行前向传播，并返回输出张量。
        
        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, channels, height, width)。
        
        Returns:
            torch.Tensor: 输出张量，形状与输入张量x相同。
        
        """
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return F.relu(y)


class Net(nn.Module):
    def __init__(self, num_channels: int=256, num_res_blocks: int=7) -> None:
        """
        初始化Net类
        
        Args:
            num_channels (int, optional): 通道数，默认为256.
            num_res_blocks (int, optional): 残差块数量，默认为7.
        
        Returns:
            None
        """
        super(Net, self).__init__()
        # 初始化特征
        self.conv_block = nn.Conv2d(in_channels=4, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_block_bn = nn.BatchNorm2d(256)
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 15 * 15, 15 * 15)
        # 价值头
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 15 * 15, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数，计算策略(policy)和价值(value)
        
        Args:
            x (Tensor): 输入的Tensor，形状为[batch_size, channels, height, width]
        
        Returns:
            Tuple[Tensor, Tensor]: 包含两个Tensor的元组，分别为策略(policy)和价值(value)
        
                - policy (Tensor): 策略输出，形状为[batch_size, num_actions]，表示每个动作的概率分布
                - value (Tensor): 价值输出，形状为[batch_size]，表示每个状态对应的价值估计
        
        """
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = F.relu(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = torch.reshape(policy, [-1, 16 * 15 * 15])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = torch.reshape(value, [-1, 8 * 15 * 15])
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = F.tanh(value)
        return policy, value


class PolicyValueNet:
    def __init__(self, model_file: str=None, use_gpu: bool=True, device: str= 'cuda') -> None:
        """
        初始化类实例。
        
        Args:
            model_file (str, optional): 模型文件路径。默认为None，表示不使用预训练模型。
            use_gpu (bool, optional): 是否使用GPU。默认为True。
            device (str, optional): 设备类型，支持'cuda'和'cpu'。默认为'cuda'。
        
        Returns:
            None
        """
        self.use_gpu = use_gpu
        self.l2_const = 2e-3    # l2 正则化
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))  # 加载模型参数

    def policy_value(self, state_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据给定的状态批量计算策略和价值。
        
        Args:
            state_batch (List[np.ndarray]): 形状为(batch_size, state_dim)的状态批量。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回元组，包含以下两个元素：
            - act_probs (np.ndarray): 形状为(batch_size, num_actions)的策略概率数组。
            - value (np.ndarray): 形状为(batch_size,)的价值数组。
        
        """
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    def policy_value_fn(self, board: Board) -> Tuple[List[Tuple[int, float]], float]:
        """
        获取当前棋盘状态下，所有合法动作的概率分布及状态价值。
        
        Args:
            board (Board): 棋盘对象，包含当前棋盘状态信息。
        
        Returns:
            Tuple[List[Tuple[int, float]], float]: 包含两个元素的元组，
            第一个元素为包含所有合法动作及其概率的列表，列表的每个元素为一个元组，
            元组的第一个元素为动作（即落子位置），第二个元素为对应的概率；
            第二个元素为状态价值。
        
        """
        self.policy_value_net.eval()
        # 获取合法动作列表
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, 15, 15)).astype('float16')
        current_state = torch.as_tensor(current_state).to(self.device)
        # 使用神经网络进行预测
        with autocast():
            log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = log_act_probs.cpu() , value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy().astype('float16').flatten())
        # 只取出合法动作
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # 返回动作概率，以及状态价值
        return act_probs, value.detach().numpy()

    # 保存模型
    def save_model(self, model_file: str) -> None:
        """
        保存模型参数到指定文件。
        
        Args:
            model_file (str): 模型参数保存的文件路径。
        
        Returns:
            None
        
        """
        torch.save(self.policy_value_net.state_dict(), model_file)

    def train_step(self, state_batch: List[np.ndarray], mcts_probs: List[np.ndarray], winner_batch: List[float], lr: float=0.002) -> Tuple[float, float]:
        self.policy_value_net.train()
        """
        训练一步模型。
        
        Args:
            state_batch (List[np.ndarray]): 游戏状态的批量数据，每个数据为二维的 numpy 数组。
            mcts_probs (List[np.ndarray]): 通过 MCTS 得到的策略概率的批量数据，每个数据为一维的 numpy 数组。
            winner_batch (List[float]): 游戏结果的批量数据，每个数据为 0 或 1 的浮点数。
            lr (float, optional): 学习率，默认为 0.002。
        
        Returns:
            Tuple[float, float]: 包含两个元素的元组，第一个元素为训练损失，第二个元素为策略熵。
        
        """
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)
        self.optimizer.zero_grad()
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        value_loss = F.mse_loss(input=value, target=winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

if __name__ == '__main__':
    net = Net().to('cuda')
    from torchsummary import summary
    summary(net, (4, 15, 15), device='cuda')