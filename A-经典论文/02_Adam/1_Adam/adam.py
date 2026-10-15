"""Adam 优化器代码"""

import numpy as np


def adam(initial_parameter, gradients, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """输入参数向量和逐步梯度，返回参数轨迹（含初值）与每步更新量。"""
    theta = np.array(initial_parameter, dtype=np.float64, copy=True)
    gradients = np.asarray(gradients, dtype=np.float64)
    if (theta.ndim != 1 or theta.size == 0 or gradients.ndim != 2
            or gradients.shape[0] == 0 or gradients.shape[1:] != theta.shape):
        raise ValueError("初始参数须为非空向量，梯度须为形状 (步数, 参数个数) 的非空数组")

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = [theta.copy()]
    updates = []

    for t, g in enumerate(gradients, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        theta = theta - update

        history.append(theta.copy())
        updates.append(update)

    return np.asarray(history), np.asarray(updates)