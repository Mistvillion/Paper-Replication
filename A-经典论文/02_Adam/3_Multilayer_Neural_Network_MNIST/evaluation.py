"""关闭 Dropout，分批评估固定参数下的完整数据集。"""

import numpy as np

from model import cross_entropy, forward, l2_penalty


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    batch_size: int,
    l2_strength: float,
) -> dict[str, float]:
    """按样本数加权计算 NLL/准确率，L2 惩罚只在最终目标值中加一次。"""
    if batch_size <= 0 or len(labels) == 0:
        raise ValueError("评估需要正批量大小和非空数据集")
    nll_sum, correct = 0.0, 0
    for start in range(0, len(labels), batch_size):
        batch_labels = labels[start:start + batch_size]
        logits, _ = forward(features[start:start + batch_size], parameters)
        nll, _ = cross_entropy(logits, batch_labels)
        nll_sum += len(batch_labels) * nll
        correct += int(np.sum(logits.argmax(axis=1) == batch_labels))
    nll = nll_sum / len(labels)
    metrics = {
        "nll": nll,
        "objective": nll + l2_penalty(parameters, l2_strength),
        "accuracy": correct / len(labels),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("评估出现非有限值，请检查学习率和输入数据")
    return metrics
