# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

import copy
import time
from collections import deque
from typing import Any

import numpy as np

from config import CONFIG

board_init = [[0] * 15 for _ in range(15)]

state_q = deque(maxlen=4)

for _ in range(4):
    state_q.append(copy.deepcopy(board_init))

board2array = {0: np.array([0, 0]), 1: np.array([1, 0]), 2: np.array([0, 1])}

def list2array(state: list[list[int]]) -> np.ndarray:
    """
    将二维列表转换为三维numpy数组
    
    Args:
        state (list[list[int]]): 二维列表，每个元素表示一个位置的状态，值为0-2的整数
    
    Returns:
        np.ndarray: 三维numpy数组，形状为(15, 15, 2)，每个位置的值与输入列表对应位置的元素相同，其余位置为0
    
    """
    _state = np.zeros((15, 15, 2))
    for i in range(15):
        for j in range(15):
            _state[i][j] = board2array[state[i][j]]
    return _state

def array2list(state: np.ndarray) -> list[list[int]]:
    """
    将NumPy数组转换为Python列表。
    
    Args:
        state (numpy.ndarray): 待转换的NumPy数组。
    
    Returns:
        list: 转换后的Python列表。
    
    """
    return list(filter(lambda array: (board2array[array] == state).all(), board2array))[0]

def place(state_list, pos: tuple, player: int):
    """
    在棋盘上的指定位置放置棋子。
    
    Args:
        state_list (list): 棋盘状态列表，二维列表表示棋盘。
        pos (tuple): 放置棋子的位置，格式为 (x, y)，其中 x 和 y 均为整数。
        player (int): 放置棋子的玩家编号，取值为 1 或 2。
    
    Returns:
        list: 放置棋子后的棋盘状态列表，与输入参数 state_list 格式相同。
    
    """
    copy_state = copy.deepcopy(state_list)
    x, y = pos
    copy_state[x][y] = player
    return copy_state

def print_board(state_array: np.ndarray):
    """
    打印15*15棋盘的状态。
    
    Args:
        state_array (np.ndarray): 形状为 (15, 15, 2) 的NumPy数组，表示棋盘的状态。
    
    Returns:
        None
    
    """
    '''HWC: 15*15*2'''
    board_line = []
    for i in range(15):
        for j in range(15):
            board_line.append(array2list(state_array[i][j]))
        print(board_line)
        board_line.clear()

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

def file_map(array):
    pass

def check_bound(pos):
    """
    判断一个坐标点是否在棋盘的范围内
    
    Args:
        pos (tuple): 包含两个整数的元组，表示一个坐标点 (x, y)
    
    Returns:
        bool: 如果坐标点在范围内返回 True，否则返回 False
    
    """
    x, y = pos
    if 0 <= x < 15 and 0 <= y < 15:
        return True
    return False

def check_obstruct(piece):
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

def get_legal_moves(state: list[list[int]]) -> list[tuple]:
    """
    获取当前状态下的所有合法移动。
    
    Args:
        state list[list[int]]: 形状为 (15, 15, 2) 的NumPy数组，表示棋盘的状态。
    
    Returns:
        list[int]: 所有合法移动的id.
    
    """
    legal_moves = []
    for i in range(15 * 15):
        x, y = np.unravel_index(i, (15, 15))
        if check_bound((x, y)):
            piece = state[x][y]
            if check_obstruct(piece):
                legal_moves.append(i)
    return legal_moves

def is_win(state, player:int, pos: tuple, direction: str) -> bool:
    x, y = pos
    if direction == 'transverse':
        left = [0, -1]
        right = [0, 1]
    elif direction == 'vertical':
        left = [-1, 0]
        right = [1, 0]
    elif direction == 'diagonal':
        left = [-1, -1]
        right = [1, 1]
    elif direction == 'anti-diagonal':
        left = [-1, 1]
        right = [1, -1]
    else:
        raise ValueError(f'direction {direction} is not supported')
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
    return ans >= 5

class Board:
    def __init__(self):
        self.state = copy.deepcopy(board_init)
        self.running = False
        self.winner = None

    def init_board(self, start_player: str='people'):
        if start_player == 'people':
            self.id2color = {'people': 1, 'ai': 2}
            self.color2id = {1: 'people', 2: 'ai'}
        elif start_player == 'ai':
            self.id2color = {'people': 2, 'ai': 1}
            self.color2id = {1: 'ai', 2: 'people'}
        self.current_player_color = self.id2color['people']
        self.current_player_id = self.color2id[1]
        self.state = copy.deepcopy(board_init)
        self.last_move = -1
        self.running = False
        self.action_count = 0
        self.line_count = 0
        self.winner = None
    
    @property
    def availables(self):
        return get_legal_moves(self.state)
    
    def current_state(self):
        '''CHW, 4*15*15
        第一个维度黑子，第二维度白子，第三个维度上一次的落子点，第四个维度当前选手是否为先手
        '''
        _curr_state = np.zeros((4, 15, 15))
        if self.running:
            _curr_state[:2] = list2array(self.state).transpose(2, 0, 1)
            move = move_id2move_action[self.last_move]
            x, y = move
            _curr_state[3][x][y] = 1
            if self.action_count % 2 == 0:
                _curr_state[3][:, :] = 1.0
        return _curr_state
    
    def do_move(self, move: int):
        '''
        执行一个移动，并返回下一个状态。
        Args:
            move (int): 移动的id.
        Returns:
            list[list[int]]: 下一个棋盘的状态.
        '''
        self.running = True
        self.action_count += 1
        move_action = move_id2move_action[move]
        x, y = move_action
        state = copy.deepcopy(self.state)
        state[x][y] = self.current_player_color
        self.current_player_color = 1 if self.current_player_color == 2 else 2
        self.current_player_id = 'ai' if self.current_player_id == 'people' else 'people'
        self.last_move = move
        self.state = state
    
    def has_a_winner(self):
        color = 1 if self.current_player_color == 2 else 2
        if self.last_move != -1:
            if is_win(self.state, color, move_id2move_action[self.last_move], 'transverse') or\
              is_win(self.state, color, move_id2move_action[self.last_move], 'vertical') or\
              is_win(self.state, color, move_id2move_action[self.last_move], 'diagonal') or\
              is_win(self.state, color, move_id2move_action[self.last_move], 'anti-diagonal'):
                self.winner = color
        if self.winner is not None:
            return True, self.winner
        elif all(0 not in row for row in self.state):
            return False, -1
        return False, -1
    
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif all(0 not in row for row in self.state):
            return True, -1
        return False, -1
    
    def get_current_player_color(self):
        return self.current_player_color
    
    def get_current_player_id(self):
        return self.current_player_id

class Game:
    def __init__(self, board):
        self.board = board
    
    def graphic(self, board, player1_color, player2_color):
        print('player1 VS player2')
        print(f'{board.color2id[player1_color]} VS {board.color2id[player2_color]}')
        print(f'{player1_color} VS {player2_color}')
        print_board(list2array(board.state))
    
    def start_play(self, player1, player2, start_player='people', is_display=True):
        if start_player not in ('people', 'ai'):
            raise Exception('start player must be people or ai')
        self.board.init_board(start_player)
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        if is_display:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player_color()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_display:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                print('Game Over')
                if winner == -1:
                    print('Nobody Wins!')
                else:
                    print(f'Winner is {players[winner].player}')
                return winner
    
    def start_self_play(self, player, is_display=False, temp=1e-3):
        self.board.init_board()
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        _count = 0
        while True:
            _count += 1
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
                print(f'one step cost {time.time() - start_time}')
            else:
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.get_current_player_color())
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_display:
                    if winner != -1:
                        print(f'Game Over, winner is {winner}')
                    else:
                        print('Game Over, nobody wins')
                return winner, zip(states, mcts_probs, winner_z)
                


        

if __name__ == '__main__':
    import random
    class people:
        def __init__(self, x) -> None:
            self.x = x
            self.count = 0
            self.ys = [7, 8, 9, 10, 11]
        def get_action(self, board):
            y = self.ys[self.count]
            self.count += 1
            return move_action2move_id[(self.x, y)]
        
        def set_player_ind(self, p):
            self.player = p
    
    people1 = people(6)
    people2 = people(7)
    board = Board()
    game = Game(board)
    # for i in range(20):
    game.start_play(people2, people1)