"""6 -> 2 -> 1 镜像对称检测网络：参数初始化、前向传播、反向传播、动量更新与全批量训练"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid，对正负输入分别使用不会溢出的写法。"""
    x = np.asarray(x, dtype=np.float64)
    output = np.empty_like(x)
    positive = x >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    output[~positive] = exp_x / (1.0 + exp_x)
    return output


def init_parameters(
    n_inputs: int, n_hidden: int, init_scale: float, seed: int
) -> dict[str, np.ndarray]:
    """在 [-init_scale, init_scale] 内均匀初始化权重与偏置。

    形状：W1 (n_inputs, n_hidden)，b1 (n_hidden,)，W2 (n_hidden, 1)，b2 (1,)。
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.uniform(-init_scale, init_scale, size=(n_inputs, n_hidden)),
        "b1": rng.uniform(-init_scale, init_scale, size=(n_hidden,)),
        "W2": rng.uniform(-init_scale, init_scale, size=(n_hidden, 1)),
        "b2": rng.uniform(-init_scale, init_scale, size=(1,)),
    }


def forward(params: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """前向传播，输入 (样本数, n_inputs)，输出对称概率 (样本数, 1)。"""
    features = np.asarray(features, dtype=np.float64)
    n_inputs = params["W1"].shape[0]
    if features.ndim != 2 or features.shape[1] != n_inputs:
        raise ValueError(f"输入形状应为 (样本数, {n_inputs})，实际为 {features.shape}")
    hidden = sigmoid(features @ params["W1"] + params["b1"])
    return sigmoid(hidden @ params["W2"] + params["b2"])


def loss_and_gradients(
    params: dict[str, np.ndarray], features: np.ndarray, targets: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    """全批量半平方误差 E = 1/2 * sum (y_hat - y)^2 及其解析梯度。

    反向传播：输出层 delta_o = (y_hat - y) * y_hat * (1 - y_hat)，
    隐藏层 delta_h = (delta_o W2^T) * h * (1 - h)；梯度对全部样本求和（不取平均）。
    """
    hidden = sigmoid(features @ params["W1"] + params["b1"])
    output = sigmoid(hidden @ params["W2"] + params["b2"])

    delta_output = (output - targets) * output * (1.0 - output)
    delta_hidden = (delta_output @ params["W2"].T) * hidden * (1.0 - hidden)
    grads = {
        "W2": hidden.T @ delta_output,
        "b2": delta_output.sum(axis=0),
        "W1": features.T @ delta_hidden,
        "b1": delta_hidden.sum(axis=0),
    }
    loss = float(0.5 * np.sum((output - targets) ** 2))
    return loss, grads


class MomentumSGD:
    """动量更新：v <- momentum * v - learning_rate * grad，w <- w + v。"""

    def __init__(self, params: dict[str, np.ndarray], learning_rate: float, momentum: float):
        """记录学习率与动量系数，并为每个参数创建全零速度。"""
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """根据梯度原地更新参数字典。"""
        for name in params:
            self.velocity[name] = (
                self.momentum * self.velocity[name] - self.learning_rate * grads[name]
            )
            params[name] += self.velocity[name]


def train(
    params: dict[str, np.ndarray],
    features: np.ndarray,
    targets: np.ndarray,
    optimizer: MomentumSGD,
    *,
    epochs: int,
    log_interval: int,
) -> list[float]:
    """训练 epochs 个全批量 epoch，每个 epoch 先求梯度再更新一次。

    返回每个 epoch 更新前的损失列表；第 1 个及每隔 log_interval 个 epoch 打印一次。
    """
    losses = []
    for epoch in range(1, epochs + 1):
        loss, grads = loss_and_gradients(params, features, targets)
        optimizer.step(params, grads)
        losses.append(loss)
        if epoch % log_interval == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | loss {loss:.6f}")
    return losses
