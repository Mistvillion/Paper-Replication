"""镜像对称任务的数据构造"""

from __future__ import annotations

from itertools import product

import numpy as np


def mirror_labels(inputs: np.ndarray) -> np.ndarray:
    """计算镜像对称标签。

    输入形状为 (样本数, n_bits)；输出形状为 (样本数, 1)。
    当 x_i 与 x_{n+1-i} 对所有 i 都相等时标签为 1，否则为 0。
    """
    return np.all(inputs == inputs[:, ::-1], axis=1).astype(np.float64)[:, None]


def build_full_dataset(n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """构造全部 2^n_bits 种二进制输入及其标签，返回 (特征, 标签)。"""
    features = np.array(list(product([0.0, 1.0], repeat=n_bits)), dtype=np.float64)
    return features, mirror_labels(features)


def prepare_test_inputs(rows: list[list[int]], n_bits: int) -> np.ndarray:
    """校验并转换额外测试输入，要求形状为 (样本数, n_bits) 且元素为 0/1。"""
    inputs = np.array(rows, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != n_bits or not np.all((inputs == 0) | (inputs == 1)):
        raise ValueError(f"测试输入须为每行 {n_bits} 个 0/1")
    return inputs
