"""实验 1 的实验配置：用 MLP 学习 XOR，验证反向传播与动量更新"""

from __future__ import annotations

# EXPERIMENT 管理随机种子、网络结构、初始化范围、训练轮次、学习率、动量系数和收敛阈值
EXPERIMENT = {
    "seed": 7,
    "layer_sizes": (2, 4, 1),
    "init_scale": 0.5,
    "epochs": 5_000,
    "learning_rate": 0.7,
    "momentum": 0.9,
    "loss_tolerance": 1e-3,
    "log_every": 1_000,
}

# TEST_INPUTS 为训练后查看预测的输入，每个元素只能是 0 或 1，可自行增删行
TEST_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
