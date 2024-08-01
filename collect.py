from collections import deque
import copy
import os
import pickle
import time
from game import Game, Board, move_action2move_id, move_id2move_action
from net import PolicyValueNet
from mcts import MCTSPlayer
from config import CONFIG
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CollectPipeline:
    def __init__(self, init_model=None):
        self.board = Board()
        self.game = Game(self.board)
        self.temp = 1
        self.n_playout = CONFIG['n_playout']
        self.c_puct = CONFIG['c_puct']
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

    def load_model(self, model_path=CONFIG['model_path']):
        try:
            self.policy_value = PolicyValueNet(model_file=model_path, device=device)
            print('load model from {}'.format(model_path))
        except:
            self.policy_value = PolicyValueNet(device=device)
            print('load model failed, use initial policy')
        self.mcts = MCTSPlayer(self.policy_value.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=True)
        
    def get_equl_data(Self, play_data):
        pass

    def collect_selfplay_data(self, n_game=1):
        for i in range(n_game):
            self.load_model()
            winner, play_data = self.game.start_self_play(
                self.mcts, is_display=False, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            if os.path.exists(CONFIG['train_data_buffer_path']):
                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as f:
                            data_file = pickle.load(f)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                            del data_file
                            self.iters += 1
                            self.data_buffer.extend(play_data)
                        print('load data from {}'.format(CONFIG['train_data_buffer_path']))
                        break
                    except:
                        time.sleep(30)
            else:
                self.data_buffer.extend(play_data)
                self.iters += 1
            
            data_file = {
                'data_buffer': self.data_buffer,
                'iters': self.iters
            }
            with open(CONFIG['train_data_buffer_path'], 'wb') as f:
                pickle.dump(data_file, f)
        return self.iters

    def run(self):
        try:
            while True:
                iters = self.collect_selfplay_data()
                print(f'batch i {iters}, episodic len {self.episode_len}')
        except KeyboardInterrupt:
            print('\n\rexit')

if __name__ == '__main__':
    pipeline = CollectPipeline()
    pipeline.run()
