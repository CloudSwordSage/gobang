# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

CONFIG = {
    'dirichlet': 0.05,
    'n_playout' : 2000,
    'c_puct': 5,
    'buffer_size': 100000,
    'model_path': './model/temp_model.pth',
    'train_data_buffer_path': './data/train_buffer.pkl',
    'batch_size': 256,
    'epochs': 5,
    'kl_targ' : 0.2,
    'game_batch_num': 3000,
    }