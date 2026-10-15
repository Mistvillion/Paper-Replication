"""实验 3：集中管理 MNIST 多层网络、正则化和优化器配置。"""

# EXPERIMENT 管理随机种子、数据与输出路径、样本量、网络结构和训练预算
EXPERIMENT = {
    "seed": 7,
    # 各随机流在基础种子上的偏移量：保证测试集抽样、验证划分、批次顺序、Dropout 掩码互相独立；
    # 批次顺序与掩码序列对所有优化器使用同一偏移，因此各方法看到相同序列。
    "seed_offsets": {
        "test_subset": 1,
        "validation_split": 2,
        "batch_order": 100,
        "dropout": 200,
    },
    "data_dir": "data/mnist",
    # 优先使用自己的缓存；没有时复用实验 2-1 的数据，不依赖其代码。
    "reuse_data_file": "../2-1_Logistic_Regression_MNIST/data/adam/mnist.npz",
    "output_dir": "results",
    "max_train_samples": 6000,
    "max_test_samples": 2000,
    "validation_fraction": 0.1,
    "input_size": 784,
    "num_classes": 10,
    "hidden_sizes": (128, 128),
    "epochs": 10,
    "batch_size": 128,
    "evaluation_batch_size": 512,
    "show_plots": True,
}

# REGIMES 管理两组正则化方案；两组使用相同数据、初值和批次顺序。
# Dropout 概率表示丢弃概率，依次对应输入层、隐藏层 1、隐藏层 2。
REGIMES = {
    "L2": {"l2_strength": 1e-4, "dropout_rates": (0.0, 0.0, 0.0)},
    "Dropout": {"l2_strength": 0.0, "dropout_rates": (0.2, 0.5, 0.5)},
}

# OPTIMIZERS 管理各优化器的专属超参数。以下为本实现的起始值，不是论文调参后的最优值；
# 本实验使用恒定学习率。
OPTIMIZERS = {
    "Adam": {"learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    "AdaGrad": {"learning_rate": 1e-2, "epsilon": 1e-8},
    "RMSProp": {"learning_rate": 1e-3, "rho": 0.9, "epsilon": 1e-8},
    "Nesterov": {"learning_rate": 0.05, "momentum": 0.9},
    "AdaDelta": {"learning_rate": 1.0, "rho": 0.95, "epsilon": 1e-6},
}
