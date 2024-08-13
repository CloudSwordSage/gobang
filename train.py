# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

from collections import deque
import copy
import os
import pickle
import time
from game import Board, move_action2move_id, move_id2move_action
from net import PolicyValueNet
from mcts import MCTS
from config import *
import torch
import numpy as np
import random
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CollectPipeline:
    def __init__(self, init_model=None) -> None:
        """
        初始化收集类。
        
        Args:
            init_model (Any, optional): 初始化模型，默认为None。
        
        Returns:
            None
        """
        self.board = Board()
        self.temp = 1
        self.n_playout = N_PLAYOUT
        self.c_puct = C_PUCT
        self.buffer_size = BUFFER_SIZE
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

    def load_model(self, model_path=MODEL_PATH, process_id=0) -> None:
        """
        加载训练好的模型用于蒙特卡洛树搜索算法。
        
        Args:
            model_path (str): 模型文件路径，默认为全局变量MODEL_PATH。
        
        Returns:
            None
        
        """
        try:
            self.policy_value = PolicyValueNet(model_file=model_path, device=device)
            print(f'收集进程{process_id}: load model from {model_path}')
        except:
            self.policy_value = PolicyValueNet(device=device)
            print(f'收集进程{process_id}: load model failed, use initial policy')
        self.mcts = MCTS(self.policy_value.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout)
        
    def get_equi_data(self, play_data) -> list:
        """
        对输入的棋盘状态、MCTS概率和胜者进行旋转和翻转，生成等价数据。
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(15, 15)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def self_paly(self, process_id=0):
        """
        进行一场自我对弈，并返回对弈结果
        
        Args:
            无
        
        Returns:
            Tuple[int, Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
                一个元组，包含两个元素：
                - 第一个元素为胜利者的玩家编号（整数类型），如果游戏平局则返回-1
                - 第二个元素为一个迭代器，每次迭代返回一个包含三个元素的元组：
                    1. 当前棋盘状态（numpy数组类型）
                    2. 当前棋盘状态下，使用MCTS算法得到的各个下棋点的概率分布（numpy数组类型）
                    3. 当前棋盘状态对应的胜负标签（numpy数组类型），胜者位置为1.0，败者位置为-1.0，平局时全为0.0
        
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        _count = 0
        while True:
            _count += 1
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = self.mcts.get_move_probs(self.board, temp=self.temp)
                self.mcts.update_with_move(move)
                print(f'收集进程{process_id}: one step cost {time.time() - start_time}')
            else:
                move, move_probs = self.mcts.get_move_probs(self.board, temp=self.temp)
                self.mcts.update_with_move(move)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.get_current_player)
            self.board.do_move(move)
            end, winner = self.board.game_end
            if end:
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                return winner, zip(states, mcts_probs, winner_z)

    def collect_selfplay_data(self, n_game=1, process_id=0):
        """
        收集自对弈数据，并将数据存储到缓存文件中。
        
        Args:
            n_game (int): 自对弈游戏次数，默认为1。
        
        Returns:
            int: 当前数据缓存中的迭代次数。
        
        """
        for i in range(n_game):
            self.load_model(MODEL_PATH, process_id=process_id)
            winner, play_data = self.self_paly(process_id=process_id)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            equi_data = self.get_equi_data(play_data)
            if os.path.exists(BUFFER_PATH):
                while True:
                    try:
                        with open(BUFFER_PATH, 'rb') as f:
                            data_file = pickle.load(f)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                            del data_file
                            self.iters += 1
                            self.data_buffer.extend(play_data)
                            self.data_buffer.extend(equi_data)
                        print(f'收集进程{process_id}: load data from {BUFFER_PATH}')
                        break
                    except:
                        time.sleep(30)
            else:
                self.data_buffer.extend(play_data)
                self.data_buffer.extend(equi_data)
                self.iters += 1
            
            data_file = {
                'data_buffer': self.data_buffer,
                'iters': self.iters
            }
            with open(BUFFER_PATH, 'wb') as f:
                pickle.dump(data_file, f)
        return self.iters

    def run(self, process_id):
        """
        运行自对弈数据收集流程。
        
        Args:
            process_id (int): 进程编号。
        
        Returns:
            None
        
        """
        try:
            while True:
                iters = self.collect_selfplay_data(5, process_id)
                print(f'收集进程{process_id}: batch i {iters}, episodic len {self.episode_len}')
        except KeyboardInterrupt:
            print('\n\rexit')

class TrainPipeline:
    def __init__(self, init_model=None):
        """
        初始化模型训练所需参数并加载预训练模型
        
        Args:
            init_model (str, optional): 预训练模型路径，默认为None。
        
        Returns:
            None
        
        """
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.kl_targ = KL_TARG
        self.check_freq = 100
        self.game_batch_num = GAME_BATCH_EPOCHS
        try:
            with open('start.txt', 'r') as f:
                self.start = int(f.read())
        except:
            self.start = 0
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
        """
        更新策略网络参数
        
        Args:
            无
        
        Returns:
            tuple: 包含两个元素的元组，分别为训练过程中的损失值和熵值
        
        """
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
            if kl > self.kl_targ * 4:  # 在kl大于4倍目标值时，停止训练
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))

        print('kl散度: {:.5f}, \t lr_multiplier 学习率乘数: {:.3f}'.format(kl, self.lr_multiplier))
        print('loss 损失值: {:.5f}, \t entropy 熵: {:.5f}'.format(loss, entropy))
        print('explained_var 解释方差: {:.3f}, {:.3f}'.format(np.mean(explained_var_old), np.mean(explained_var_new)))
        return loss, entropy

    def run(self):
        """
        运行游戏，进行策略迭代和价值迭代。
        
        Args:
            无。
        
        Returns:
            无返回值。
        """
        try:
            for i in range(self.start, self.game_batch_num):
                time.sleep(30)
                while True:
                    try:
                        with open(BUFFER_PATH, 'rb') as f:
                            try:
                                date_file = pickle.load(f)
                            except:
                                # 删除收集好的数据，重新收集
                                os.remove(BUFFER_PATH)
                            self.data_buffer = date_file['data_buffer']
                            self.iters = date_file['iters']
                            del date_file
                        print('load data from {}'.format(BUFFER_PATH))
                        break
                    except Exception as e:
                        time.sleep(30)
                        print(f'open buffer Error: {e}')
                print()
                print(f"{'训练进程':=^30}")
                print(f'epoch: {i + 1}', end='\t')
                print(f'step: {self.iters}')
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                self.policy_value.save_model(MODEL_PATH)
                with open('start.txt', 'w') as f:
                    f.write(str(i + 1))
                if (i + 1) % self.check_freq == 0:
                    print(f'current selfplay batch: {i + 1}')
                    self.policy_value.save_model(f'./model/current_model_batch_{i + 1}.pth')
                print(f"{'':=^30}")
        except KeyboardInterrupt:
            print('\n\rexit')
        except Exception as e:
            print(f'run Error: {e}')
    

if __name__ == '__main__':
    import argparse
    import multiprocessing
    import time
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--collect_num', type=int, default=4, help='收集数据的进程数')
    args = parser.parse_args()
    if args.collect_num <= 0:
        raise ValueError('collect_num must be greater than 0')
    else:
        COLLECT_NUM = args.collect_num
    for i in range(COLLECT_NUM):
        time.sleep(1)
        collect = CollectPipeline()
        multiprocessing.Process(target=collect.run, args=(i + 1,)).start()
    time.sleep(1)
    train = TrainPipeline(MODEL_PATH)
    multiprocessing.Process(target=train.run).start()