"""纯 NumPy 的多类 Softmax 回归及手工解析梯度。"""

# 包含4个函数：
# initialize_logistic_parameters()：初始化参数
# stable_softmax()：执行稳定的Softmax，将分数转换为概率
# softmax_loss_and_gradients()：计算损失和参数梯度
# predict_probabilities()：使用训练后的参数预测类别概率

from __future__ import annotations

import numpy as np


def initialize_logistic_parameters(
    n_features: int,
    n_classes: int,
    *,
    seed: int,
    scale: float = 0.01,
) -> dict[str, np.ndarray]:
    """创建一份可供所有优化器复制的相同逻辑回归初始参数"""
    rng = np.random.default_rng(seed)
    return {
        "weights": rng.normal(0.0, scale, size=(n_features, n_classes)),
        "bias": np.zeros(n_classes, dtype=np.float64),
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """逐行减最大值，避免指数溢出且不改变 Softmax 概率"""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def softmax_loss_and_gradients(
    features: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    l2_strength: float,
) -> tuple[float, float, dict[str, np.ndarray]]:
    """返回一个批次的含 L2 总损失、未正则化 NLL 和参数梯度。"""
    if len(labels) == 0:
        raise ValueError("批次不能为空")

    # 前向计算
    logits = features @ parameters["weights"] + parameters["bias"]
    probabilities = stable_softmax(logits)
    row_indices = np.arange(len(labels))
    nll = -float(np.mean(np.log(np.clip(probabilities[row_indices, labels], 1e-12, 1.0))))  # 计算负对数似然
    regularization = 0.5 * l2_strength * float(np.sum(parameters["weights"] ** 2))  # L2正则化：对大权重加大惩罚

    # 计算总目标对参数的梯度
    logits_gradient = probabilities.copy()
    logits_gradient[row_indices, labels] -= 1.0
    logits_gradient /= len(labels)
    gradients = {
        "weights": features.T @ logits_gradient + l2_strength * parameters["weights"],
        "bias": np.sum(logits_gradient, axis=0),
    }
    return nll + regularization, nll, gradients


def predict_probabilities(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """只执行前向计算：线性映射和Softmax"""
    logits = features @ parameters["weights"] + parameters["bias"]
    return stable_softmax(logits)
