"""实验 5 的实验配置：数据规模、网络结构、训练预算、随机种子与超参数网格。"""

# EXPERIMENT 集中管理随机种子、数据路径、样本量、网络规模、训练与评估预算及运行设备
EXPERIMENT = {
    "seed": 7,
    # 各随机流相对 seed 的偏移量，使数据抽样、划分、二值化、批次顺序、训练噪声与评估噪声互相独立
    "seed_offsets": {
        "test_subset": 1,
        "validation_split": 2,
        "binarization": 10,  # 训练/验证/测试依次使用 10、11、12
        "batch_order": 100,
        "train_noise": 200,
        "eval_train": 300,
        "eval_validation": 400,
        "eval_test": 500,
    },
    "data_dir": "data/mnist",
    # 本实验没有缓存时，优先复用实验 2-1 已下载的 MNIST；设为 None 则禁用复用
    "reuse_data_file": "../2-1_Logistic_Regression_MNIST/data/adam/mnist.npz",
    "output_dir": "results",
    "max_train_samples": 2000,  # None 表示使用官方全部 60000 个训练样本
    "max_test_samples": 1000,  # None 表示使用官方全部 10000 个测试样本
    "validation_fraction": 0.1,
    "hidden_units": 128,  # 论文为 500
    "latent_dim": 20,  # 论文为 50
    "initial_weight_std": 0.01,
    "epochs": 10,  # 论文图 4 的训练预算为 100 轮
    # 论文图 4 展示第 10、100 轮；默认第 1、10 轮用于小规模观察
    "checkpoint_epochs": (1, 10),
    "batch_size": 128,
    "evaluation_batch_size": 256,
    "train_mc_samples": 1,
    "evaluation_mc_samples": 3,
    "plot_split": "train",  # 可改为 "validation"；两者都使用固定评估噪声
    "device": "cpu",  # 可显式改为 "cuda" 或 "mps"
    "torch_threads": 4,
    "deterministic": True,
    "show_plots": True,
}

# SWEEP 管理 Adam 的超参数网格；每个格点都分别运行有/无偏差修正，默认共 2×3×5×2=60 次训练
SWEEP = {
    "beta1_values": (0.0, 0.9),
    "beta2_values": (0.99, 0.999, 0.9999),
    "log10_learning_rates": (-5, -4, -3, -2, -1),
    "epsilon": 1e-8,
}
