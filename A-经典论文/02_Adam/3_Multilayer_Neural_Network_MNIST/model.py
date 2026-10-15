"""NumPy 手写两隐藏层 ReLU 网络、倒置 Dropout 与反向传播。"""

import numpy as np


def initialize_parameters(
    input_size: int, hidden_sizes: tuple[int, int], num_classes: int, seed: int
) -> dict[str, np.ndarray]:
    """生成 输入-h1-h2-类别数 网络；权重用 He 正态初始化，偏置为零。"""
    if len(hidden_sizes) != 2 or any(size <= 0 for size in hidden_sizes):
        raise ValueError("hidden_sizes 必须包含两个正整数")
    rng = np.random.default_rng(seed)
    sizes = (input_size, *hidden_sizes, num_classes)
    parameters = {}
    for layer in range(1, 4):
        fan_in, fan_out = sizes[layer - 1], sizes[layer]
        parameters[f"W{layer}"] = rng.normal(
            0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)
        )
        parameters[f"b{layer}"] = np.zeros(fan_out, dtype=np.float64)
    return parameters


def apply_dropout(
    values: np.ndarray, rate: float, rng: np.random.Generator | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """训练时按保留率缩放掩码；rate=0 时不抽随机数且返回原数组。"""
    if not 0.0 <= rate < 1.0:
        raise ValueError("Dropout 丢弃概率必须位于 [0, 1)")
    if rate == 0.0:
        return values, None
    if rng is None:
        raise ValueError("启用 Dropout 时必须提供随机数生成器")
    mask = (rng.random(values.shape) >= rate) / (1.0 - rate)
    return values * mask, mask


def forward(
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
    dropout_rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple]:
    """返回 B×10 logits 与反传缓存；默认关闭 Dropout，用于确定性评估。"""
    if len(dropout_rates) != 3:
        raise ValueError("dropout_rates 应依次指定输入层、隐藏层 1、隐藏层 2")
    x, _ = apply_dropout(np.asarray(features, dtype=np.float64), dropout_rates[0], rng)
    z1 = x @ parameters["W1"] + parameters["b1"]
    h1, mask1 = apply_dropout(np.maximum(z1, 0), dropout_rates[1], rng)
    z2 = h1 @ parameters["W2"] + parameters["b2"]
    h2, mask2 = apply_dropout(np.maximum(z2, 0), dropout_rates[2], rng)
    logits = h2 @ parameters["W3"] + parameters["b3"]
    return logits, (x, z1, h1, z2, h2, mask1, mask2)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """用 log-sum-exp 计算平均 NLL，同时返回稳定的 Softmax 概率。"""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizer = exponentials.sum(axis=1, keepdims=True)
    nll = np.mean(np.log(normalizer[:, 0]) - shifted[np.arange(len(labels)), labels])
    return float(nll), exponentials / normalizer


def l2_penalty(parameters: dict[str, np.ndarray], strength: float) -> float:
    """计算 lambda/2 乘三层权重的平方和；不惩罚偏置。"""
    return float(0.5 * strength * sum(np.sum(parameters[f"W{i}"] ** 2) for i in range(1, 4)))


def loss_and_gradients(
    features: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    l2_strength: float,
    dropout_rates: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[float, float, dict[str, np.ndarray]]:
    """返回当前批次的目标值、NLL 和全部参数梯度；掩码与前向传播保持一致。"""
    logits, cache = forward(features, parameters, dropout_rates, rng)
    x, z1, h1, z2, h2, mask1, mask2 = cache
    nll, delta = cross_entropy(logits, labels)
    objective = nll + l2_penalty(parameters, l2_strength)

    # Softmax 与平均交叉熵的联合导数：(P - Y) / B。
    delta[np.arange(len(labels)), labels] -= 1.0
    delta /= len(labels)
    gradients = {
        "W3": h2.T @ delta + l2_strength * parameters["W3"],
        "b3": delta.sum(axis=0),
    }
    delta = delta @ parameters["W3"].T
    if mask2 is not None:
        delta *= mask2
    delta *= z2 > 0
    gradients["W2"] = h1.T @ delta + l2_strength * parameters["W2"]
    gradients["b2"] = delta.sum(axis=0)
    delta = delta @ parameters["W2"].T
    if mask1 is not None:
        delta *= mask1
    delta *= z1 > 0
    gradients["W1"] = x.T @ delta + l2_strength * parameters["W1"]
    gradients["b1"] = delta.sum(axis=0)
    # 输入不求导；W1 的梯度已经使用了加噪后的 x。
    return objective, nll, gradients
