"""实验 1：用 2 -> 4 -> 1 的 Sigmoid MLP 学习 XOR，验证反向传播与动量更新的实现。"""

from __future__ import annotations

import numpy as np

from config import EXPERIMENT, TEST_INPUTS
from model import forward, total_error, train


def main() -> None:
    """生成 XOR 数据、全批量训练、检查收敛并打印测试输入的预测。"""
    np.set_printoptions(precision=6, suppress=True)

    # XOR 的全部 4 个样本：两个输入不同时标签为 1
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    parameters, losses = train(X, y, config=EXPERIMENT)
    for epoch, loss in enumerate(losses, start=1):
        if epoch % EXPERIMENT["log_every"] == 0:
            print(f"epoch {epoch:>6d} | loss {loss:.8f}")

    # 收敛检查：训练总误差须低于阈值
    final_loss = total_error(forward(parameters, X)[0], y)
    if final_loss >= EXPERIMENT["loss_tolerance"]:
        raise AssertionError(f"训练损失 {final_loss} 未降到阈值 {EXPERIMENT['loss_tolerance']} 以下")
    print(f"训练损失: {final_loss:.8f}")

    # 对测试输入打印：输入 -> 正确标签 -> 输出概率 -> 预测类别（概率不小于 0.5 预测为 1）
    inputs = np.array(TEST_INPUTS, dtype=np.float64)
    if not np.all((inputs == 0) | (inputs == 1)):
        raise ValueError("测试输入只能包含 0 或 1")
    expected = (inputs[:, 0] != inputs[:, 1]).astype(int)
    probabilities = forward(parameters, inputs)[0].ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print("输入 -> 正确标签 -> 输出概率 -> 预测类别")
    for row, target, probability, prediction in zip(inputs, expected, probabilities, predictions):
        print(f"{row.tolist()} -> {target} -> {probability:.6f} -> {prediction}")


if __name__ == "__main__":
    main()
