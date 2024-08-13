# -*- coding: utf-8 -*-
# @Author: chenfeng
# @LICENSE: MIT

N_PLAYOUT = 2000                            # 训练时，每局游戏进行多少步
C_PUCT = 5.0                                # 探索因子
BUFFER_SIZE = 100000                        # 缓存大小
MODEL_PATH = './model/temp_model.pth'       # 模型存储路径
BUFFER_PATH = './buffer/temp_buffer.pkl'    # 缓存存储路径
BATCH_SIZE = 256                            # 批量训练大小
EPOCHS = 20                                 # 训练轮数
KL_TARG = 0.2                               # KL散度目标值
GAME_BATCH_EPOCHS = 3000                    # 训练游戏模型轮数