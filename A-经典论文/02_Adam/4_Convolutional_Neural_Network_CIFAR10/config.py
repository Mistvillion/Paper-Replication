"""实验 4：CIFAR-10 卷积网络的教学规模、预处理和优化器配置。"""

EXPERIMENT = {
    "seed": 7,
    "data_dir": "data/cifar10",
    "output_dir": "results",
    "max_train_samples": 3000,
    "max_test_samples": 1000,
    "validation_fraction": 0.1,
    # 在划分后的训练集上拟合整幅图像的正则化 ZCA；不是逐通道标准化。
    "whitening": "zca",  # 可改为 "none"，但这将不再使用论文要求的白化。
    "zca_epsilon": 0.1,
    "preprocessing_batch_size": 256,
    "channels": (16, 16, 32),
    "hidden_units": 128,
    "epochs": 10,
    "batch_size": 128,
    "evaluation_batch_size": 256,
    # 前三轮每轮约评估四次，用于观察图 3 左侧的早期进展。
    "detail_epochs": 3,
    "evaluations_per_early_epoch": 4,
    "l2_strength": 0.0,
    "device": "cpu",  # 显式可选 "cuda" 或 "mps"；不可用时报错，不自动切换。
    "torch_threads": 4,
    "deterministic": True,
    "show_plots": True,
}

REGIMES = {
    "NoDropout": {"input_dropout": 0.0, "hidden_dropout": 0.0},
    "Dropout": {"input_dropout": 0.2, "hidden_dropout": 0.5},
}

# 未搜索的教学起点。各层共用该方法的基础学习率，不套用 6.1 节的衰减。
OPTIMIZERS = {
    "Adam": {"learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    "AdaGrad": {"learning_rate": 1e-2, "epsilon": 1e-8},
    "Nesterov": {"learning_rate": 1e-2, "momentum": 0.9},
}
