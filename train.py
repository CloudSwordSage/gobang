import random
import numpy as np
import pickle
import time
from net import PolicyValueNet
from config import CONFIG
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainPipeline:
    def __init__(self, init_model=None):
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ']
        self.check_freq = 100
        self.game_batch_num = CONFIG['game_batch_num']
        if init_model:
            try:
                self.policy_value = PolicyValueNet(init_model, device=device)
                print("Loaded model from {}".format(init_model))
            except:
                self.policy_value = PolicyValueNet(device=device)
                print(f'No model found at {init_model}, use initial model')
        else:
            self.policy_value = PolicyValueNet(device=device)
            print('No model found, use initial model')
    
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype(np.float32)

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype(np.float32)

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype(np.float32)

        old_probs, old_v = self.policy_value.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                         self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))

        print('kl: {:.5f}, lr_multiplier: {:.3f}'.format(kl, self.lr_multiplier))
        print('loss: {:.5f}, entropy: {:.5f}'.format(loss, entropy))
        print('explained_var: {:.3f}, {:.3f}'.format(np.mean(explained_var_old), np.mean(explained_var_new)))
        return loss, entropy

    def run(self):
        try:
            for i in range(self.game_batch_num):
                time.sleep(30)
                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as f:
                            date_file = pickle.load(f)
                            self.data_buffer = date_file['data_buffer']
                            self.iters = date_file['iters']
                            del date_file
                        print('load data from {}'.format(CONFIG['train_data_buffer_path']))
                        break
                    except:
                        time.sleep(30)
                print(f'step: {self.iters}')
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                self.policy_value.save_model(CONFIG['model_path'])
                if (i + 1) % self.check_freq == 0:
                    print(f'current selfplay batch: {i + 1}')
                    self.policy_value.save_model('./model/current_model.pth')
        except KeyboardInterrupt:
            print('\n\rexit')


if __name__ == '__main__':
    train_pipeline = TrainPipeline(CONFIG['model_path'])
    train_pipeline.run()