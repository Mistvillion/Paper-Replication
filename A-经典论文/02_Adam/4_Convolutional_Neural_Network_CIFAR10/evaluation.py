"""在固定参数、关闭 Dropout 的情况下评估完整数据集。"""

from math import isfinite

import torch
from torch.nn import functional as F

from model import l2_penalty


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, dataset: tuple[torch.Tensor, torch.Tensor],
    *, batch_size: int, device: torch.device, l2_strength: float,
) -> dict[str, float]:
    """按样本累计 NLL/正确数，最后统一求均值；临时切换 eval，结束后恢复原模式。"""
    images, labels = dataset
    if len(labels) == 0 or batch_size <= 0:
        raise ValueError("评估数据不能为空，批量大小必须为正")
    was_training = model.training
    model.eval()
    nll_sum, correct = 0.0, 0
    try:
        for start in range(0, len(labels), batch_size):
            batch_images = images[start:start + batch_size].to(device)
            batch_labels = labels[start:start + batch_size].to(device)
            logits = model(batch_images)
            nll_sum += F.cross_entropy(logits, batch_labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        nll = nll_sum / len(labels)
        metrics = {
            "nll": nll,
            "objective": nll + l2_penalty(model, l2_strength).item(),
            "accuracy": correct / len(labels),
        }
        if not all(isfinite(value) for value in metrics.values()):
            raise FloatingPointError("评估指标含非有限值")
        return metrics
    finally:
        model.train(was_training)
