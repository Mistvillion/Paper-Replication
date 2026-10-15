"""实验 2-1：MNIST 稠密输入的逻辑回归优化器对比，统一运行入口。"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import EXPERIMENT, MNIST_OPTIMIZERS
from data import load_mnist_dataset, split_train_validation
from reporting import plot_training_costs, print_final_metrics
from training import run_optimizer_suite


def main() -> None:
    """整个 MNIST 逻辑回归优化器对比实验的总入口"""

    # 设置输出格式与绘图风格
    np.set_printoptions(precision=6, suppress=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 确定数据目录
    seed = EXPERIMENT["seed"]
    data_dir = Path(EXPERIMENT["data_dir"])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent / data_dir
    print("Data directory:", data_dir)

    # 加载并预处理 MNIST
    mnist = load_mnist_dataset(
        data_dir,
        max_train_samples=EXPERIMENT["mnist"]["max_train_samples"],
        max_test_samples=EXPERIMENT["mnist"]["max_test_samples"],
        seed=seed,
    )

    # 把原训练集拆成：训练集train、验证集validation
    (
        mnist_train_features,
        mnist_train_labels,
        mnist_validation_features,
        mnist_validation_labels,
    ) = split_train_validation(
        mnist["train_features"],
        mnist["train_labels"],
        EXPERIMENT["validation_fraction"],
        seed=seed,
    )

    # 分别使用三种优化器训练模型
    # 给 run_optimizer_suite() 传入公共初始参数，此函数会返回三种优化器使用公共参数训练的结果
    mnist_results = run_optimizer_suite(
        mnist_train_features,
        mnist_train_labels,
        mnist_validation_features,
        mnist_validation_labels,
        n_features=784,
        n_classes=10,
        optimizer_settings=MNIST_OPTIMIZERS,
        epochs=EXPERIMENT["mnist"]["epochs"],
        batch_size=EXPERIMENT["batch_size"],
        l2_strength=EXPERIMENT["mnist"]["l2_strength"],
        seed=seed,
    )

    # 在独立测试集test上评估并打印最终指标
    print("\nMNIST final metrics")
    print_final_metrics(
        mnist_results,
        mnist["test_features"],
        mnist["test_labels"],
        batch_size=EXPERIMENT["batch_size"],
        l2_strength=EXPERIMENT["mnist"]["l2_strength"],
    )

    # 绘制训练损失对比曲线，显示图像
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    plot_training_costs(mnist_results, axis, title="MNIST Logistic Regression")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
