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
import torch

from tornado.options import define, options
from game import *

define("port", type=int, default=8000, help='run on the given port')
define('debug', type=bool, default=True, help='run in debug mode')

from net import PolicyValueNet
from mcts import MCTSPlayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game = Board()
policy_value_net = PolicyValueNet(device=device)
mcts_player = MCTSPlayer(policy_value_function=policy_value_net.policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=False)
game.init_board(1)

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
