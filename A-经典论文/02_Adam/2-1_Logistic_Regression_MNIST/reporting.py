"""最终测试指标汇总与训练 NLL 曲线绘制"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from evaluation import evaluate_logistic_regression


def print_final_metrics(
    results: dict[str, dict[str, object]],
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    batch_size: int,
    l2_strength: float,
) -> None:
    """打印每个优化器的最终训练、验证、测试指标和耗时。"""
    # 测试集只在训练完成后评估，不参与配置选择。
    print("optimizer | train_nll | val_acc | test_nll | test_acc | seconds")
    for name, result in results.items():
        history = result["history"]
        test_metrics = evaluate_logistic_regression(
            test_features,
            test_labels,
            result["parameters"],
            batch_size=batch_size,
            l2_strength=l2_strength,
        )
        print(
            f"{name:>9s} | {history['train_nll'][-1]:.4f} | "
            f"{history['validation_accuracy'][-1]:.4f} | "
            f"{test_metrics['nll']:.4f} | {test_metrics['accuracy']:.4f} | "
            f"{result['elapsed_seconds']:.1f}"
        )


def plot_training_costs(
    results: dict[str, dict[str, object]],
    axis: Axes,
    *,
    title: str,
) -> None:
    """在给定坐标轴上绘制各优化器的未正则化训练 NLL。"""
    # 横轴一个单位代表一次完整数据遍历，对应论文 Figure 1 的含义。
    for name, result in results.items():
        history = result["history"]
        axis.plot(
            history["epoch"],
            history["train_nll"],
            marker="o",
            markersize=3,
            label=name,
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training NLL")
    axis.legend()
