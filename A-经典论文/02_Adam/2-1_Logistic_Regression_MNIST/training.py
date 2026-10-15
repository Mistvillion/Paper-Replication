"""统一训练循环及公平优化器对比"""

# 包含2个函数：
# train_one_optimizer()：完成一个优化器的训练（使用optimizer_name变量来区分不同的优化器）
# run_optimizer_suite()：依次调用train_one_optimizer()函数，完成全部优化器的对比

from __future__ import annotations
from time import perf_counter

import numpy as np

from data import make_feature_batch
from evaluation import evaluate_logistic_regression
from model import initialize_logistic_parameters, softmax_loss_and_gradients
from optimizers import create_optimizer


def train_one_optimizer(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    initial_parameters: dict[str, np.ndarray],
    *,
    optimizer_name: str,
    optimizer_settings: dict[str, float],
    epochs: int,
    batch_size: int,
    l2_strength: float,
    seed: int,
) -> dict[str, object]:
    """在固定数据顺序下训练一个优化器，并记录每个训练轮次epoch的指标。"""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs 和 batch_size 必须为正数")

    # 每次开始时都复制公共初值
    parameters = {}
    for name, value in initial_parameters.items():
        parameters[name] = value.copy()

    # 创建优化器：optimizer_name变量决定创建什么优化器
    optimizer = create_optimizer(optimizer_name, parameters, optimizer_settings)

    # 准备学习率、随机数生成器和计时器
    base_learning_rate = float(optimizer_settings["learning_rate"])
    rng = np.random.default_rng(seed)
    update_count = 0
    started_at = perf_counter()

    # 创建训练历史记录：每个列表在每轮结束时追加一个值
    history = {
        "epoch": [],
        "train_nll": [],
        "train_objective": [],
        "train_accuracy": [],
        "validation_nll": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    # 训练的主循环（外层：训练轮次epoch，内层：批量大小batch）
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(train_labels))  # 每轮开始时打乱样本索引
        for start in range(0, len(train_labels), batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_features = make_feature_batch(train_features, batch_indices)
            batch_labels = train_labels[batch_indices]

            # Nesterov 在前瞻位置求梯度；Adam/AdaGrad 在当前位置求梯度
            gradient_parameters = optimizer.parameters_for_gradient()
            _, _, gradients = softmax_loss_and_gradients(batch_features, batch_labels, gradient_parameters, l2_strength=l2_strength)
            update_count += 1  # 更新累计步数
            learning_rate = base_learning_rate / np.sqrt(update_count)  # 更新学习率
            optimizer.step(gradients, learning_rate)  # 让优化器更新模型参数

        # 一轮所有参数更新完成后，用当前模型重新评估整个训练集和整个验证集
        train_metrics = evaluate_logistic_regression(
            train_features,
            train_labels,
            parameters,
            batch_size=batch_size,
            l2_strength=l2_strength,
        )
        validation_metrics = evaluate_logistic_regression(
            validation_features,
            validation_labels,
            parameters,
            batch_size=batch_size,
            l2_strength=l2_strength,
        )

        # 保存这一轮的数据
        history["epoch"].append(epoch)
        history["train_nll"].append(train_metrics["nll"])
        history["train_objective"].append(train_metrics["objective"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["validation_nll"].append(validation_metrics["nll"])
        history["validation_accuracy"].append(validation_metrics["accuracy"])
        history["learning_rate"].append(learning_rate)
        print(
            f"{optimizer_name:>9s} | epoch {epoch:03d}/{epochs:03d} | "
            f"train NLL {train_metrics['nll']:.4f} | "
            f"val acc {validation_metrics['accuracy']:.4f}"
        )

    return {
        "parameters": parameters,
        "history": history,
        "elapsed_seconds": perf_counter() - started_at,
    }


def run_optimizer_suite(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    n_features: int,
    n_classes: int,
    optimizer_settings: dict[str, dict[str, float]],
    epochs: int,
    batch_size: int,
    l2_strength: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    """只生成一次初值，依次训练全部优化器并收集结果。"""
    initial_parameters = initialize_logistic_parameters(n_features, n_classes, seed=seed)
    results: dict[str, dict[str, object]] = {}
    for optimizer_name, settings in optimizer_settings.items():
        print(f"\n=== {optimizer_name} ===")
        results[optimizer_name] = train_one_optimizer(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            initial_parameters,
            optimizer_name=optimizer_name,
            optimizer_settings=settings,
            epochs=epochs,
            batch_size=batch_size,
            l2_strength=l2_strength,
            seed=seed + 100,
        )
    return results
