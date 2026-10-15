"""实验 3（家族树关系补全）的实验配置"""

from __future__ import annotations

# EXPERIMENT 管理随机种子、初始化范围、训练轮数、损失超参数和演示查询
EXPERIMENT = {
    "seed": 8,
    "init_scale": 0.3,  # 参数在 [-init_scale, init_scale] 内均匀初始化
    "epochs": 5_000,  # 全批量训练，一个 epoch 即一次参数更新
    "positive_weight": 3.0,  # 加权二元交叉熵中正例（答案人物）的权重
    "l2_strength": 1e-5,  # 只作用于权重，不作用于偏置
    # 演示查询，每项写成 (人物, 关系)；名称见 data.py 的 PEOPLE 与 RELATIONS
    "test_queries": [("Colin", "mother"), ("Colin", "aunt"), ("Sophia", "father")],
}

# ADAM 管理 Adam 优化器的超参数
ADAM = {
    "learning_rate": 1e-2,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
}
