"""实验 2-1 的实验配置"""

from __future__ import annotations

# EXPERIMENT 管理随机种子、数据路径、批量大小、验证比例、样本量、epoch 数和 L2 系数
EXPERIMENT = {
    "seed": 7,
    "data_dir": "data/adam",
    "batch_size": 128,
    "validation_fraction": 0.10,
    "mnist": {
        "max_train_samples": 12_000,
        "max_test_samples": 2_000,
        "epochs": 10,
        "l2_strength": 1e-4,
    },
}

# MNIST_OPTIMIZERS 管理各优化器的基础学习率及专属超参数
MNIST_OPTIMIZERS = {
    "Adam": {
        "learning_rate": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    },
    "AdaGrad": {"learning_rate": 1e-1, "epsilon": 1e-8},
    "Nesterov": {"learning_rate": 1e-1, "momentum": 0.9},
}
