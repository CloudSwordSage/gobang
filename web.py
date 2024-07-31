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

from tornado.options import define, options
from env import *

define("port", type=int, default=8000, help='run on the given port')
define('debug', type=bool, default=False, help='run in debug mode')

env = GoBangEnv()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('./index.html')

class GameWebSocket(tornado.websocket.WebSocketHandler):
    def initialize(self):
        self.boardArray = env.state
        self.board_size = len(self.boardArray)
        self.current_player = 1

    def check_board_full(self):
        return all(cell != 0 for row in self.boardArray for cell in row)

    def open(self):
        self.write_message(json.dumps({'boardArray': self.boardArray, 'currentPlayer': self.current_player}))

    def on_message(self, message):
        data = json.loads(message)
        move = data.get('move')
        if move is not None:
            x, y = np.unravel_index(move, (15, 15))
            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.boardArray[x][y] == 0:
                self.boardArray, rec, done, winner, _, = env.step(self.current_player, (x, y))
                self.current_player = 3 - self.current_player
                self.write_message(json.dumps({'boardArray': self.boardArray, 'currentPlayer': self.current_player, 'reward': rec}))
                if done:
                    self.write_message(json.dumps({'gameOver': True}))

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
