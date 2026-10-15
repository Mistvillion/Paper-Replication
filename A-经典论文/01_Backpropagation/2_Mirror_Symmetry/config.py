"""实验 2（6 位镜像对称）的实验配置"""

from __future__ import annotations

# EXPERIMENT 管理随机种子、输入位数、训练轮次和日志间隔
EXPERIMENT = {
    "seed": 1,
    "n_bits": 6,
    "epochs": 1_425,
    "log_interval": 285,
}

# MODEL 管理网络结构与初始化范围
MODEL = {
    "n_hidden": 2,
    "init_scale": 0.3,
}

# OPTIMIZER 管理全批量动量梯度下降的超参数
OPTIMIZER = {
    "learning_rate": 0.1,
    "momentum": 0.9,
}

# TEST_INPUTS 为训练后额外查看预测的输入，每行须为 n_bits 个 0/1
TEST_INPUTS = [
    [0, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 0],
]
