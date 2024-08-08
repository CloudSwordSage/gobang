# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

import tornado
import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.websocket
import tornado.options
import json
import random
import numpy as np

from tornado.options import define, options
from game import *

define('debug', type=bool, default=False, help='run in debug mode')

from mcts import MCTSPlayer

def limits(legal_moves, last_move_id, lim_distance=4):
        """
        根据上一次移动的位置，返回一定范围内的合法移动
        
        Args:
            legal_moves (list): 当前合法的移动列表
            last_move_id (int): 上一次移动的id
            lim_distance (int, optional): 范围大小，默认为4
        
        Returns:
            list: 在一定范围内的合法移动列表
        
        """
        ans = []
        x, y = move_id2move_action[last_move_id]
        for i in range(x - lim_distance, x + lim_distance + 1):
            for j in range(y - lim_distance, y + lim_distance + 1):
                if 0 <= i < 15 and 0 <= j < 15:
                    move_id = move_action2move_id[(i, j)]
                    if move_id in legal_moves:
                        ans.append(move_id)
        return ans

def random_policy_value(state:Game) -> tuple[list[tuple[int, float]], float]:
    """
    根据给定游戏状态生成随机的策略值和叶子节点值。
    
    Args:
        state (Game): 游戏状态对象，包含游戏当前状态信息。
    
    Returns:
        tuple[list[tuple[int, float]], float]: 一个元组，包含两个元素：
            - list[tuple[int, float]]: 一个列表，包含可选动作的随机概率分布，每个元素是一个元组，包含可选动作和对应的概率。
            - float: 叶子节点的值，初始化为0.0。
    
    """
    legal_moves = state.availables
    action_probs = [(move, random.random()) for move in legal_moves]
    leaf_value = 0.0
    return action_probs, leaf_value


def distance_based_move_probabilities(state: Game) -> tuple[list[tuple[int, float]], float]:  
    """  
    基于距离计算游戏状态中的合法走步及其对应的概率。  
  
    Args:  
        state (Game): 游戏状态对象，包含最后一步走步、合法走步等信息。  
  
    Returns:  
        tuple[list[tuple[int, float]], float]: 包含合法走步和对应概率的列表，以及叶子节点的值。  
    """  
    def distance(a, b):
        x1, y1 = move_id2move_action[a]
        x2, y2 = move_id2move_action[b]
        return abs(x1 - x2) + abs(y1 - y2)
    
    last_move_id = state.last_move
    legal_moves = state.availables
    legal_moves = limits(legal_moves, last_move_id)
    dis = []
    min_d = np.inf
    max_d = -np.inf
    for move in legal_moves:
        d = distance(last_move_id, move)
        min_d = min(min_d, d)
        max_d = max(max_d, d)
        dis.append((move, d))
    action_probs = []
    for move, d in dis:
        p = (max_d - d) / (max_d - min_d + 1e-8)
        action_probs.append((move, p))
    leaf_value = 0.0
    return action_probs, leaf_value

def policy_value(state: Game) -> tuple[list[tuple[int, float]], float]:
    last_move_id = state.last_move
    last_move = move_id2move_action[last_move_id]
    legal_moves = limits(state.availables, last_move_id)
    player = state.get_current_player_color
    board = state.state

    def dfs(board, i, j, dx, dy):
        if not (0 <= i < 15 and 0 <= j < 15 and board[i][j] == player):
            return 0
        board[i][j] = 0
        ans = 1
        for dx_i, dy_i in (dx, dy), (dy, -dx):
            x, y = i + dx_i, j + dy_i
            if 0 <= x < 15 and 0 <= y < 15 and board[x][y] == player:
                ans += dfs(board, x, y, dx_i, dy_i)
        return ans

    directions = [
        ([0, 0], [-1, 1]),
        ([-1, 1], [0, 0]),
        ([-1, 1], [-1, 1]),
        ([-1, 1], [1, -1])
    ]
    
    max_length = 0
    preferred_directions = []
    for dx, dy in directions:
        length = dfs(copy.deepcopy(board), last_move[0], last_move[1], dx, dy)
        if length >= 3:
            max_length = max(max_length, length)
            preferred_directions.append((dx, dy))

    if preferred_directions:
        left, right = preferred_directions[0]
        pr = 1.0 if max_length == 4 else 0.9
        sign = True
    else:
        left, right = [0, 0], [0, 0]
        pr = 0.0
        sign = False


    x, y = last_move
    i_left, j_left = x, y
    i_right, j_right = x + right[0], y + right[1]
    while 0 <= i_left < 15 and 0 <= j_left < 15 and state.state[i_left][j_left] == player:
        i_left, j_left = i_left + left[0], j_left + left[1]
    while 0 <= i_right < 15 and 0 <= j_right < 15 and state.state[i_right][j_right] == player:
        i_right, j_right = i_right + right[0], j_right + right[1]
    if sign:
        a = move_action2move_id[(i_left + left[0], j_left + left[1])]
        b = move_action2move_id[(i_right + right[0], j_right + right[1])]
    else:
        a, b = -1, -1

    def distance(a, b):
        x1, y1 = move_id2move_action[a]
        x2, y2 = move_id2move_action[b]
        return abs(x1 - x2) + abs(y1 - y2)

    dis = []
    min_d = np.inf
    max_d = -np.inf
    for move in legal_moves:
        d = distance(last_move_id, move)
        min_d = min(min_d, d)
        max_d = max(max_d, d)
        dis.append((move, d))

    action_probs = []
    for move, d in dis:
        if move == a or move == b:
            p = pr
        else:
            p = (max_d - d) / (max_d - min_d + 1e-8)
        action_probs.append((move, p))

    leaf_value = 0.0
    return action_probs, leaf_value

from net import PolicyValueNet

Net = PolicyValueNet('./model/current_model_batch_3000.pth')

game = Board()
mcts_player = MCTSPlayer(policy_value_function=Net.policy_value_fn, c_puct=5, n_playout=2000)
game.init_board('people')

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('./index.html')

class GameWebSocket(tornado.websocket.WebSocketHandler):
    def initialize(self):
        self.boardArray = game.state
        self.current_player = game.get_current_player_color()
        self.board_size = 15

    def check_board_full(self):
        return all(cell != 0 for row in self.boardArray for cell in row)

    def open(self):
        self.write_message(json.dumps({'boardArray': self.boardArray, 'currentPlayer': self.current_player, 'input': 1}))

    def on_message(self, message):
        data = json.loads(message)
        reset = data.get('reset', False)
        player = data.get('player_currentPlayer', None)
        if reset:
            game.init_board(1)
            self.boardArray = game.state
            self.write_message(json.dumps({'boardArray': self.boardArray, 'currentPlayer': self.current_player, 'input': 1}))
        move = data.get('move', None)
        if move is not None:
            x, y = np.unravel_index(move, (15, 15))
            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.boardArray[x][y] == 0:
                game.do_move(move)
                done, winner = game.game_end()
                self.boardArray = game.state
                self.write_message(json.dumps({'boardArray': self.boardArray, 'currentPlayer': self.current_player, 'input': 0}))
                if done:
                    self.write_message(json.dumps({'winner': winner, 'gameOver': True}))
                    return
            try:
                move_mcts = mcts_player.get_action(game)
                x, y = np.unravel_index(move_mcts, (15, 15))
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.boardArray[x][y] == 0:
                    game.do_move(move_mcts)
                    done, winner = game.game_end()
                    self.boardArray = game.state
                    self.write_message(json.dumps({'boardArray': self.boardArray, 'currentPlayer': self.current_player, 'input': 1}))
                    if done:
                        self.write_message(json.dumps({'winner': winner, 'gameOver': True}))
                        return
            except Exception as e:
                print(e, flush=True)
                data = {}


    def on_close(self):
        pass

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/websocket", GameWebSocket),
    ], debug=options.debug)

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()