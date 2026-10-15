"""实验 2：训练 6 -> 2 -> 1 网络，判断 6 位二进制向量是否镜像对称，统一运行入口。"""

from __future__ import annotations

import numpy as np

from config import EXPERIMENT, MODEL, OPTIMIZER, TEST_INPUTS
from data import build_full_dataset, mirror_labels, prepare_test_inputs
from model import MomentumSGD, forward, init_parameters, train


def main() -> None:
    """构造全部输入、训练网络并打印训练集准确率与额外输入的预测。"""
    np.set_printoptions(precision=6, suppress=True)

    # 训练集是全部 2^n_bits 种输入
    n_bits = EXPERIMENT["n_bits"]
    features, targets = build_full_dataset(n_bits)
    test_inputs = prepare_test_inputs(TEST_INPUTS, n_bits)

    # 初始化参数与优化器，全批量训练
    params = init_parameters(n_bits, MODEL["n_hidden"], MODEL["init_scale"], EXPERIMENT["seed"])
    optimizer = MomentumSGD(params, OPTIMIZER["learning_rate"], OPTIMIZER["momentum"])
    train(
        params,
        features,
        targets,
        optimizer,
        epochs=EXPERIMENT["epochs"],
        log_interval=EXPERIMENT["log_interval"],
    )

    # 打印训练集准确率与额外输入的预测
    probabilities = forward(params, features)
    accuracy = float(((probabilities >= 0.5).astype(int) == targets.astype(int)).mean())
    print(f"训练集准确率: {accuracy:.2%} ({round(accuracy * len(features))}/{len(features)})")

    expected = mirror_labels(test_inputs).ravel().astype(int)
    test_probabilities = forward(params, test_inputs).ravel()
    test_predictions = (test_probabilities >= 0.5).astype(int)
    print("输入 -> 正确标签 -> 对称概率 -> 预测标签")
    for row, target, probability, prediction in zip(
        test_inputs.astype(int), expected, test_probabilities, test_predictions
    ):
        print(f"{''.join(map(str, row))} -> {target} -> {probability:.6f} -> {prediction}")


if __name__ == "__main__":
    main()
