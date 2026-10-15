"""实验 2-2：集中管理数据、训练与优化器配置。"""

EXPERIMENT = {
    "seed": 7,
    "data_dir": "data/imdb",
    "dataset_url": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "archive_name": "aclImdb_v1.tar.gz",
    "max_train_samples": 5_000,  # 划分前的有标签训练池；None 使用全部 25,000 条。
    "max_test_samples": 2_000,  # None 使用全部 25,000 条。
    "validation_fraction": 0.10,
    "vocabulary_size": 10_000,
    "binary_bow": True,  # True 记录是否出现；False 记录词频，不做长度归一化。
    "dropout_rate": 0.50,
    "l2_strength": 0.0,  # IMDB 的 L2 系数未在论文明确；默认只用输入 Dropout。
    "batch_size": 128,
    "evaluation_batch_size": 1_024,
    "epochs": 10,
    "learning_rate_decay": True,  # alpha_t = alpha / sqrt(t)，t 为全局更新步。
    "show_plots": True,
    "output_dir": "results",
}

# 下列学习率是教学起点，不是论文网格搜索所得的最佳参数。
IMDB_OPTIMIZERS = {
    "Adam": {"learning_rate": 1e-2, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    "AdaGrad": {"learning_rate": 1e-1, "epsilon": 1e-8},
    "RMSProp": {"learning_rate": 1e-2, "beta2": 0.9, "epsilon": 1e-8},
    "Nesterov": {"learning_rate": 1e-1, "momentum": 0.9},
}


def validate_config(config: dict, optimizers: dict) -> None:
    """在下载前检查样本量、训练预算与数值超参数的范围。"""
    import math

    # 检查词表大小、训练批量、评估批量、训练轮数：必须是正整数
    for key in ("vocabulary_size", "batch_size", "evaluation_batch_size", "epochs"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} 必须为正整数")

    # 检查训练样本量上限、测试样本量上限：必须是4的倍数
    for key in ("max_train_samples", "max_test_samples"):
        value = config[key]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 4):
            raise ValueError(f"{key} 必须为至少 4 的整数或 None")

    # 检查验证集所占比例、dropout比例、L2系数、优化器配置
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction 必须位于 (0, 1)")
    if not 0 <= config["dropout_rate"] < 1:
        raise ValueError("dropout_rate 必须位于 [0, 1)")
    if not math.isfinite(config["l2_strength"]) or config["l2_strength"] < 0:
        raise ValueError("l2_strength 必须是有限非负数")
    if not optimizers:
        raise ValueError("至少配置一个优化器")

    for name, settings in optimizers.items():
        if name not in {"Adam", "AdaGrad", "RMSProp", "Nesterov"}:
            raise ValueError(f"未知优化器：{name}")
        for key in ("learning_rate", "epsilon"):
            value = settings.get(key, 1e-8)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}.{key} 必须是有限正数")
        for key in ("beta1", "beta2", "momentum"):
            if key in settings and not 0 <= settings[key] < 1:
                raise ValueError(f"{name}.{key} 必须位于 [0, 1)")
