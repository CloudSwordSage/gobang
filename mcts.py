# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

import copy
import numpy as np
from game import Board
from typing import *

def softmax(x: np.ndarray)-> np.ndarray:
    """
    计算输入数组x的softmax值。
    
    Args:
        x (np.ndarray): 输入的数组，形状为(N,)，其中N为数组长度。
    
    Returns:
        np.ndarray: 经过softmax函数计算后的数组，形状与输入数组x相同。
    
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class TreeNode:
    def __init__(self, parent:int|None=None, prior_p:float=0.0) -> None:
        """
        初始化一个节点对象。
        
        Args:
            parent (int|None): 当前节点的父节点编号，若当前节点为根节点，则为None。默认为None。
            prior_p (float): 当前节点的先验概率。默认为0.0。
        
        Returns:
            None
        
        """
        self.parent = parent
        self.children = {}
        self.n_visits = 0.0
        self.Q = 0.0
        self.U = 0.0
        self.P = prior_p
    
    def select(self, c_puct:float) -> tuple:
        """
        从当前节点的子节点中选择一个最优节点。
        
        Args:
            c_puct (float): UCB公式中的参数，用于平衡探索和利用。
        
        Returns:
            tuple: 包含最优节点动作和对应节点对象的元组。
        
        """
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_score(c_puct))
    
    def expand(self, action_priors:List[Tuple[int, float]]) -> None:
        """
        根据给定的先验动作概率扩展树节点。
        
        Args:
            action_priors (List[Tuple[int, float]]): 包含动作和对应先验概率的元组列表。
        
        Returns:
            None: 该方法无返回值，但会更新树节点的子节点。
        
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(parent=self, prior_p=prob)
    
    def update(self, leaf_value:float) -> None:
        """
        更新叶节点的访问次数和平均价值。
        
        Args:
            leaf_value (float): 叶节点的价值。
        
        Returns:
            None
        
        """
        self.n_visits += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits
    
    def update_recursive(self, leaf_value:float) -> None:
        """
        递归更新树节点的值
        
        Args:
            leaf_value (float): 需要更新的叶子节点的值
        
        Returns:
            None
        
        """
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def get_score(self, c_puct:float) -> float:
        """
        计算并返回当前节点的分数。
        
        Args:
            c_puct (float): 用于计算 U 的常数，用于平衡探索和利用。
        
        Returns:
            float: 当前节点的分数，由 Q 值和 U 值相加得到。
        
        """
        self.U = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.U
    
    def is_leaf(self) -> bool:
        """
        判断当前节点是否为叶子节点。
        
        Args:
            无参数。
        
        Returns:
            bool: 若当前节点没有子节点，返回True；否则返回False。
        """
        return self.children == {}
    
    def is_root(self) -> bool:
        """
        判断当前节点是否为根节点。
        
        Args:
            无参数。
        
        Returns:
            bool: 若当前节点为根节点，则返回True；否则返回False。
        
        """
        return self.parent is None
    
class MCTS:
    def __init__(self, policy_value_fn:Callable[[Board], tuple[list[tuple[int, float]], float]], c_puct:float=1.0, n_playout:int=2000) -> None:
        """
        初始化函数，用于创建MCTS树。
        
        Args:
            policy_value_fn (Callable[[Board], tuple[list[tuple[int, float]], float]]): 策略价值函数，输入当前状态，输出每个动作的概率分布和当前状态的价值。
            c_puct (float, optional):UCT公式中的探索系数，用于平衡探索和利用。默认为1.0。
            n_playout (int, optional):每次模拟游戏进行的次数。默认为2000。
        
        Returns:
            None
        """
        self.root = TreeNode(None, 1.0)
        self.c_puct = c_puct
        self.policy = policy_value_fn
        self.n_playout = n_playout
    
    def playout(self, state:Board) -> None:
        """
        进行游戏的一步蒙特卡洛树搜索的模拟
        
        Args:
            state (Board): 当前游戏状态
        
        Returns:
            None
        
        """
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            state.do_move(action)
        action_probs, leaf_value = self.policy(state)
        end, winner = state.game_end
        if not end:
            node.expand(action_probs)
        else:
            if winner is None:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player else -1.0
        node.update_recursive(-leaf_value)
    
    def get_move(self, state:Board) -> int:
        """
        基于蒙特卡洛树搜索，获取最佳下棋动作。
        
        Args:
            state (Board): 游戏当前状态。
        
        Returns:
            int: 最佳下棋动作编号。
        
        """
        for _ in range(self.n_playout):
            _state = copy.deepcopy(state)
            self.playout(_state)
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]
    
    def get_move_probs(self, state: Board, temp=1e-3) -> Tuple[int, np.ndarray[np.float64]]:
        """
        根据给定的状态和温度参数，返回下一步的移动和对应的概率分布。
        
        Args:
            state (Board): 游戏状态，包含棋盘、玩家信息等。
            temp (float, optional): 温度参数，用于控制概率分布的平滑度。默认为1e-3。
        
        Returns:
            tuple: 包含两个元素的元组，第一个元素为下一步的移动id，第二个元素为225维的numpy数组，表示每个移动的概率分布。
        
        """
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)
        
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        move_probs = np.zeros(225)
        move_probs[list(acts)] = act_probs
        move = np.random.choice(acts, p=0.75*act_probs + 0.25 * np.random.dirichlet(np.ones(len(act_probs)) * 0.03))
        return move, move_probs
    
    def update_with_move(self, last_move: int) -> None:
        """
        根据上一步的移动更新树结构。
        
        Args:
            last_move (int): 上一步的移动id
        
        Returns:
            None
        
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

