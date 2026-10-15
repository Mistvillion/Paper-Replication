"""在关闭 Dropout 的完整数据上，分批评估确定性的 NLL、目标与准确率。"""

import numpy as np

from model import logits, negative_log_likelihood


def evaluate(features, labels: np.ndarray, parameters: dict,
             l2_strength: float, batch_size: int) -> dict[str, float]:
    """按样本累加损失，正确加权末尾小批量；L2 项仅在平均后加入一次。"""
    total_nll, correct = 0.0, 0
    for start in range(0, len(labels), batch_size):
        targets = labels[start:start + batch_size]
        scores = logits(features[start:start + batch_size], parameters)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("评估 logit 非有限，请降低学习率并检查输入")
        total_nll += float(negative_log_likelihood(scores, targets).sum())
        correct += int(np.count_nonzero((scores >= 0) == targets))
    nll = total_nll / len(labels)
    objective = nll + 0.5 * l2_strength * float(parameters["w"] @ parameters["w"])
    if not np.isfinite(objective):
        raise FloatingPointError("评估目标非有限")
    return {"nll": nll, "objective": objective, "accuracy": correct / len(labels)}
