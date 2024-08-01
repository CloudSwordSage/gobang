# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

import copy

import numpy as np

from config import CONFIG

def softmax(x):
    """计算 x 中每组分数的 softmax 值."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class TreeNode:
    """树节点类."""
    def __init__(self, parent, prior_p):
        # 父节点
        self.parent = parent
        # 子节点
        self.children = {}
        self.n_visit = 0
        self.Q = 0
        self.u = 0           # 置信上线，PUTC算法
        self.p = prior_p

    def expand(self, actions_probs):
        """根据动作 action 扩展节点."""
        for action, prob in actions_probs:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)
    
    def select(self, c_puct):
        """选择节点."""
        action_max = max(self.children.items(), key=lambda x: x[1].get_value(c_puct))
        return action_max

    def get_value(self, c_puct) -> None:
        self.u = (c_puct * self.p * np.sqrt(self.parent.n_visit) / (1 + self.n_visit))
        return self.u + self.Q
    
    def update(self, leaf_value) -> None:
        self.n_visit += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visit
    
    def update_recursive(self, leaf_value) -> None:
        """递归更新节点."""
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def is_leaf(self) -> bool:
        """判断是否为叶子节点."""
        return self.children == {}
    
    def is_root(self) -> bool:
        """判断是否为根节点."""
        return self.parent is None

class MCTS:
    """蒙特卡洛树搜索."""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        # 策略函数
        self.policy_value_fn = policy_value_fn
        # 置信上限，PUTC算法
        self.c_puct = c_puct
        # 模拟次数
        self.n_playout = n_playout
        # 根节点
        self.root = TreeNode(None, 1.0)
    
    def playout(self, state):
        """模拟游戏."""
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            state.do_move(action)
        action_probs, leaf_value = self.policy_value_fn(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player_color() else -1.0
                )
        node.update_recursive(-leaf_value)
    
    def get_move_probs(self, state, temp=1e-3):
        """获取当前状态下的动作概率分布."""
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)
        
        act_visits = [(act, node.n_visit) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """更新树."""
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
    
    def __str__(self):
        return 'MCTS'

class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_function, c_puct=c_puct, n_playout=n_playout)
        self.is_selfplay = is_selfplay
        self.agent = 'ai'
    
    def set_player_ind(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_with_move(-1)
    
    def __str__(self) -> str:
        return f'MCTS-{self.player}'
    
    def get_action(self, board, temp=1e-3, return_prob=0):
        move_probs = np.zeros(225)
        acts, probs = self.mcts.get_move_probs(board, temp=temp)
        move_probs[list(acts)] = probs
        if self.is_selfplay:
            move = np.random.choice(acts, p=0.75*probs + 0.25 * np.random.dirichlet(np.ones(len(probs)) * CONFIG['dirichlet']))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        
        if return_prob:
            return move, move_probs
        else:
            return move
    

