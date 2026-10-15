"""评估当前Softmax回归模型在一整个数据集上的表现（NLL、含L2目标值和准确率）"""

from __future__ import annotations

import numpy as np

from data import make_feature_batch
from model import predict_probabilities


def evaluate_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    batch_size: int,
    l2_strength: float,
) -> dict[str, float]:
    """按样本累计指标；评估不修改参数，也不消耗随机数。"""
    total_nll = 0.0  # 保存所有已处理样本的负对数似然之和
    total_correct = 0  # 保存预测正确的样本数量
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))  # start为当前batch的起点；stop为当前batch的终点
        indices = np.arange(start, stop)

        # 取出当前batch并计算预测概率
        batch_features = make_feature_batch(features, indices)
        probabilities = predict_probabilities(batch_features, parameters)
        targets = labels[indices]

        # 累计NLL负对数似然之和与预测正确的样本数量
        rows = np.arange(len(targets))
        total_nll -= float(np.sum(np.log(np.clip(probabilities[rows, targets], 1e-12, 1.0))))
        total_correct += int(np.sum(np.argmax(probabilities, axis=1) == targets))

    # 先累计样本损失再除以样本总数，最后一个小批次也获得正确权重。
    mean_nll = total_nll / len(labels)

    # L2正则放在循环外，只需计算一次
    objective = mean_nll + 0.5 * l2_strength * float(np.sum(parameters["weights"] ** 2))
    return {
        "nll": mean_nll,
        "objective": objective,
        "accuracy": total_correct / len(labels),
    }
