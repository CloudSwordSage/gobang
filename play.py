import random

from game import move_action2move_id, move_id2move_action, Board, Game
from net import PolicyValueNet
from mcts import MCTSPlayer
import torch

class People:
        def get_action(self, board):
            return move_action2move_id[tuple(map(int, input().split()))]
        
        def set_player_ind(self, p):
            self.player = p
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_value_net = PolicyValueNet('./model/current_model.pth', device=device)
mcts_player = MCTSPlayer(policy_value_function=policy_value_net.policy_value_fn, c_puct=5, n_playout=5, is_selfplay=False)

people = People()

game = Game(Board())
game.start_play(people, mcts_player, 1, True)