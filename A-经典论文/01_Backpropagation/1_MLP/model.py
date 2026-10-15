"""通用 Sigmoid MLP：参数初始化、前向传播、总误差、反向传播、动量更新与全批量训练"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid：x 为正、负时分别用不同写法，避免 exp 溢出。"""
    x = np.asarray(x, dtype=np.float64)
    output = np.empty_like(x)
    positive = x >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    output[~positive] = exp_x / (1.0 + exp_x)
    return output


def initialize_parameters(
    layer_sizes: tuple[int, ...], *, seed: int, init_scale: float
) -> dict[str, np.ndarray]:
    """在 [-init_scale, init_scale] 内均匀初始化 W{l}（形状 (n_in, n_out)）与 b{l}（形状 (n_out,)）。"""
    rng = np.random.default_rng(seed)
    parameters = {}
    for layer, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), start=1):
        parameters[f"W{layer}"] = rng.uniform(-init_scale, init_scale, size=(n_in, n_out))
        parameters[f"b{layer}"] = rng.uniform(-init_scale, init_scale, size=(n_out,))
    return parameters


def forward(
    parameters: dict[str, np.ndarray], X: np.ndarray
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """逐层计算 A = sigmoid(A_prev @ W + b)。

    输入 X 形状 (B, n_0)；返回输出层激活 (B, n_L) 与每层 (输入, 输出) 的缓存，供反向传播复用。
    """
    n_layers = len(parameters) // 2
    activation = np.asarray(X, dtype=np.float64)
    expected_width = parameters["W1"].shape[0]
    if activation.ndim != 2 or activation.shape[1] != expected_width:
        raise ValueError(f"输入形状应为 (样本数, {expected_width})，实际为 {activation.shape}")

    cache = []
    for layer in range(1, n_layers + 1):
        previous = activation
        activation = sigmoid(previous @ parameters[f"W{layer}"] + parameters[f"b{layer}"])
        cache.append((previous, activation))
    return activation, cache


def total_error(prediction: np.ndarray, y: np.ndarray) -> float:
    """总误差 E = 1/2 * sum (预测 - 标签)^2，对 batch 求和而非取平均。"""
    return float(0.5 * np.sum((prediction - y) ** 2))


def backward(
    parameters: dict[str, np.ndarray],
    cache: list[tuple[np.ndarray, np.ndarray]],
    prediction: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    """从输出层向前传播误差信号 delta，返回与 parameters 同名的梯度字典。"""
    grads = {}
    # 输出层：dE/dA = A - Y；之后每层 delta = (dE/dA) * A * (1 - A)
    delta_from_next = prediction - y
    for layer in range(len(cache), 0, -1):
        previous, activation = cache[layer - 1]
        delta = delta_from_next * activation * (1.0 - activation)
        grads[f"W{layer}"] = previous.T @ delta
        grads[f"b{layer}"] = delta.sum(axis=0)
        # 把误差信号传回更靠近输入一侧的激活
        delta_from_next = delta @ parameters[f"W{layer}"].T
    return grads


class MomentumSGD:
    """动量更新：v = momentum * v - lr * grad，参数加上 v（v 初始为 0）。"""

    def __init__(self, parameters: dict[str, np.ndarray], *, learning_rate: float, momentum: float) -> None:
        """为每个参数建立同形状的速度，初值为 0。"""
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {name: np.zeros_like(value) for name, value in parameters.items()}

    def step(self, parameters: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """原地更新 parameters。"""
        for name in parameters:
            self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * grads[name]
            parameters[name] += self.velocity[name]


def train(X: np.ndarray, y: np.ndarray, *, config: dict) -> tuple[dict[str, np.ndarray], list[float]]:
    """用全部样本做全批量训练，每个 epoch 更新一次参数。

    返回训练后的参数和每个 epoch 更新前的总误差序列。
    """
    parameters = initialize_parameters(
        config["layer_sizes"], seed=config["seed"], init_scale=config["init_scale"]
    )
    optimizer = MomentumSGD(
        parameters, learning_rate=config["learning_rate"], momentum=config["momentum"]
    )

    losses = []
    for _ in range(config["epochs"]):
        prediction, cache = forward(parameters, X)
        losses.append(total_error(prediction, y))
        grads = backward(parameters, cache, prediction, y)
        optimizer.step(parameters, grads)
    return parameters, losses
