"""MNIST 下载、稠密像素预处理、固定抽样与训练/验证划分"""

from __future__ import annotations
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def download_file(url: str, destination: Path) -> Path:
    """下载数据文件"""
    destination = Path(destination)

    # 若已下载过数据文件，则直接返回
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urlretrieve(url, temporary)
        temporary.replace(destination)
    except Exception:
        # 下载失败时删除残缺文件，让下一次运行可以干净重试。
        temporary.unlink(missing_ok=True)
        raise
    return destination


def take_subset(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从数据集中随机抽取指定数量的样本，None 表示保留全部样本。"""
    if max_samples is None or max_samples >= len(labels):
        return features, labels
    if max_samples <= 0:
        raise ValueError("max_samples 必须为正数或 None")

    # 固定seed，每次运行抽到的子集相同，便于复现实验
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))[:max_samples]  # 选定样本的索引
    return features[indices], labels[indices]


def split_train_validation(
    features: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把原训练集随机拆成训练集和验证集，不接收测试集。"""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须位于 (0, 1)")
    if len(labels) < 2:
        raise ValueError("至少需要两个样本才能划分训练集和验证集")

    # 不把原训练集的数据传到函数内，仅根据索引来划分训练集与验证集
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(labels))
    validation_size = max(1, int(len(labels) * validation_fraction))  # 验证集大小，至少为1
    validation_indices = permutation[:validation_size]  # 原训练集中被划分到验证集的样本的索引
    training_indices = permutation[validation_size:]
    return (
        features[training_indices],
        labels[training_indices],
        features[validation_indices],
        labels[validation_indices],
    )


def load_mnist_dataset(
    data_dir: Path,
    *,
    max_train_samples: int | None,
    max_test_samples: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    """加载MNIST，将图像展平为784维向量，并缩放到[0, 1]。"""
    # 常驻数据保留 float32 节省内存，计算批次再转成 float64。
    path = download_file(MNIST_URL, Path(data_dir) / "mnist.npz")
    with np.load(path, allow_pickle=False) as dataset:
        train_images = dataset["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        train_labels = dataset["y_train"].astype(np.int64)
        test_images = dataset["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
        test_labels = dataset["y_test"].astype(np.int64)

    # 使用不同种子独立抽样，不改变原训练集/测试集边界。
    train_images, train_labels = take_subset(train_images, train_labels, max_train_samples, seed=seed)
    test_images, test_labels = take_subset(test_images, test_labels, max_test_samples, seed=seed + 1)
    return {
        "train_features": train_images,
        "train_labels": train_labels,
        "test_features": test_images,
        "test_labels": test_labels,
    }


def make_feature_batch(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """按索引取出稠密像素批次，并转为 float64 计算精度。"""
    return np.asarray(features[indices], dtype=np.float64)
