# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

import numpy as np
import copy
from pprint import pprint

from typing import *

init_board = [[0] * 15 for i in range(15)]

board2array = {0: np.array([0, 0]), 1: np.array([1, 0]), 2: np.array([0, 1])}

def list2array(state: List[List[int]]) -> np.ndarray:
    """
    将二维列表转换为三维numpy数组
    
    Args:
        state (List[List[int]]): 二维列表，每个元素表示一个位置的状态，值为0-2的整数
    
    Returns:
        np.ndarray: 三维numpy数组，形状为(15, 15, 2)，每个位置的值与输入列表对应位置的元素相同，其余位置为0
    
    """
    _state = np.zeros((15, 15, 2))
    for i in range(15):
        for j in range(15):
            _state[i][j] = board2array[state[i][j]]
    return _state

def array2list(state: np.ndarray) -> List[List[int]]:
    """
    将NumPy数组转换为Python列表。
    
    Args:
        state (numpy.ndarray): 待转换的NumPy数组。
    
    Returns:
        List[List[int]]: 转换后的Python列表。
    
    """
    board = [[0] * 15 for i in range(15)]
    for i in range(15):
        for j in range(15):
            board[i][j] = list(filter(lambda array: (board2array[array] == state[i][j]).all(), board2array))[0]
    return board

def get_all_legal_moves() -> tuple[dict, dict]:
    """
    获取所有合法移动映射关系。
    
    Args:
        无参数。
    
    Returns:
        返回一个包含两个字典的元组，分别为：
        - move_id2move_action: 将移动id映射到移动动作（即棋盘上的位置坐标）的字典。
        - move_action2move_id: 将移动动作映射到移动id的字典。
    
    """
    move_id2move_action = {}
    move_action2move_id = {}
    for i in range(15 * 15):
        x, y = np.unravel_index(i, (15, 15))
        move_id2move_action[i] = (x, y)
        move_action2move_id[(x, y)] = i
    return move_id2move_action, move_action2move_id

move_id2move_action, move_action2move_id = get_all_legal_moves()

def check_bound(pos: Tuple[int, int]) -> bool:
    """
    判断一个坐标点是否在棋盘的范围内
    
    Args:
        pos (Tuple[int, int]): 包含两个整数的元组，表示一个坐标点 (x, y)
    
    Returns:
        bool: 如果坐标点在范围内返回 True，否则返回 False
    
    """
    x, y = pos
    if 0 <= x < 15 and 0 <= y < 15:
        return True
    return False

def check_obstruct(piece: int) -> bool:
    """
    判断传入的棋子是否为空，若为空则返回True，否则返回False。
    
    Args:
        piece (int): 棋子的值，0表示空，1表示黑子，2表示白子。
    
    Returns:
        bool: 若传入的棋子为空则返回True，否则返回False。
    
    """
    if piece == 0:
        return True
    return False

def get_legal_moves(state: List[List[int]]) -> List[tuple]:
    """
    获取当前状态下的所有合法移动。
    
    Args:
        state List[List[int]]: 形状为 (15, 15, 2) 的NumPy数组，表示棋盘的状态。
    
    Returns:
        List[int]: 所有合法移动的id.
    
    """
    legal_moves = []
    for i in range(15 * 15):
        x, y = np.unravel_index(i, (15, 15))
        if check_bound((x, y)):
            piece = state[x][y]
            if check_obstruct(piece):
                legal_moves.append(i)
    return legal_moves

def is_win(state: List[List[int]], player:int, pos: Tuple[int, int]) -> bool:
    """
    判断给定玩家在给定位置是否胜利
    
    Args:
        state (List[List[int]]): 游戏棋盘状态，15x15的二维列表，每个元素为0或1或2，分别表示空位和玩家棋子
        player (int): 玩家编号，1或2
        pos (Tuple[int, int]): 玩家当前位置，为(x, y)的元组形式
    
    Returns:
        bool: 若玩家胜利则返回True，否则返回False
    """
    x, y = pos
    directions = [
        ([-1, 0], [1, 0]),   # 上下
        ([0, -1], [0, 1]),   # 左右
        ([-1, -1], [1, 1]),  # 左上到右下
        ([-1, 1], [1, -1])   # 右上到左下
    ]
    for left, right in directions:
        i_left, j_left = x, y
        i_right, j_right = x + right[0], y + right[1]
        count_left, count_right = 0, 0
        while 0 <= i_left < 15 and 0 <= j_left < 15 and state[i_left][j_left] == player:
            i_left, j_left = i_left + left[0], j_left + left[1]
            count_left += 1
            if count_left == 5:
                return True
        while 0 <= i_right < 15 and 0 <= j_right < 15 and state[i_right][j_right] == player:
            i_right, j_right = i_right + right[0], j_right + right[1]
            count_right += 1
            if count_right == 5:
                return True
        ans = count_left + count_right
        if ans >= 5:
            return True
    return False

class Board:
    def __init__(self) -> None:
        """
        初始化游戏对象。
        """
        self.state = copy.deepcopy(init_board)
        self.running = False
        self.winner = None

    def init_board(self, start_player: int=1) -> None:
        """
        初始化棋盘，设置初始状态。
        
        Args:
            start_player (int, optional): 起始玩家编号，默认为1。
        
        Returns:
            None
        
        """
        self.current_player_color = start_player
        self.state = copy.deepcopy(init_board)
        self.last_move = -1
        self.running = False
        self.action_count = 0
        self.winner = None
    
    @property
    def availables(self) -> List[int]:
        """
        返回当前游戏状态下所有合法移动的列表。
        
        Args:
            无参数。
        
        Returns:
            List[int]: 包含所有合法移动的列表，具体类型取决于游戏的状态表示。
        
        """
        return get_legal_moves(self.state)
    
    def current_state(self) -> np.ndarray:
        """
        CHW, 4x15x15

        返回当前游戏状态，以numpy数组形式表示。
        
        Args:
            无
        
        Returns:
            np.ndarray: 当前游戏状态，形状为(4, 15, 15)的numpy数组。
            第一个维度表示黑子状态，第二个维度表示白子状态，第三个维度表示上一次落子点的状态，第四个维度表示当前选手是否为先手。
        
        """
        _curr_state = np.zeros((4, 15, 15))
        if self.running:
            _curr_state[:2] = list2array(self.state).transpose(2, 0, 1)
            move = move_id2move_action[self.last_move]
            x, y = move
            _curr_state[3][x][y] = 1
            if self.action_count % 2 == 0:
                _curr_state[3][:, :] = 1.0
        return _curr_state
    
    def do_move(self, move: int) -> None:
        '''
        执行一个移动，并返回下一个状态。
        Args:
            move (int): 移动的id.
        Returns:
            None
        '''
        self.running = True
        self.action_count += 1
        move_action = move_id2move_action[move]
        x, y = move_action
        state = copy.deepcopy(self.state)
        state[x][y] = self.current_player_color
        self.current_player_color = 1 if self.current_player_color == 2 else 2
        self.last_move = move
        self.state = state
    
    def has_a_winner(self) -> Tuple[bool, int]:
        """
        判断当前游戏是否有胜者。
        
        Args:
            无。
        
        Returns:
            Tuple[bool, int]: 第一个元素为bool类型，表示是否有胜者；
            第二个元素为int类型，表示胜者的颜色，若无胜者则为-1。
        
        """
        color = 1 if self.current_player_color == 2 else 2
        if self.last_move != -1:
            if is_win(self.state, color, move_id2move_action[self.last_move],):
                self.winner = color
        if self.winner is not None:
            return True, self.winner
        elif all(0 not in row for row in self.state):
            return False, -1
        return False, -1
    
    @property
    def game_end(self) -> Tuple[bool, int]:
        """
        判断游戏是否结束。
        
        Args:
            无参数。
        
        Returns:
            Tuple[bool, int]: 第一个元素为布尔值，表示游戏是否结束；
            第二个元素为整数，表示胜利者的编号（如果有的话），-1 表示游戏未结束或平局。
        
        """
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif all(0 not in row for row in self.state):
            return True, -1
        return False, -1
    
    @property
    def get_current_player(self) -> int:
        """
        获取当前玩家的颜色。
        
        Args:
            无。
        
        Returns:
            int: 当前玩家的颜色，1代表先手玩家，2代表后手玩家。
        
        """
        return self.current_player_color

if __name__ == "__main__":
    board = copy.deepcopy(init_board)
    board[7][8] = 1
    board[8][7] = 2
    board = list2array(board)
    pprint(array2list(board))