"""MNIST 缓存、按固定种子抽样及训练/验证划分。"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def resolve_path(path: str | Path) -> Path:
    """把相对路径解析到当前实验目录，不依赖启动目录。"""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def get_mnist_file(data_dir: str, reuse_data_file: str | None) -> Path:
    """优先复用缓存，均不存在时才下载；失败时删除临时文件。"""
    destination = resolve_path(data_dir) / "mnist.npz"
    if destination.is_file():
        return destination
    if reuse_data_file is not None:
        reusable = resolve_path(reuse_data_file)
        if reusable.is_file():
            return reusable
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.part")
    try:
        urlretrieve(MNIST_URL, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def take_subset(
    images: np.ndarray, labels: np.ndarray, limit: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """抽取固定子集后展平并缩放像素；None 表示全部数据。"""
    if limit is not None and limit <= 0:
        raise ValueError("样本上限必须为正整数或 None")
    if limit is not None and limit < len(labels):
        indices = np.random.default_rng(seed).permutation(len(labels))[:limit]
        images, labels = images[indices], labels[indices]
    features = images.reshape(len(labels), -1).astype(np.float32) / 255.0
    return features, labels.astype(np.int64)


def load_mnist(config: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """保留官方测试集边界，仅把抽样后的官方训练集划分为训练/验证集。"""
    fraction = config["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction 必须在 (0, 1) 内")
    path = get_mnist_file(config["data_dir"], config["reuse_data_file"])
    print("MNIST cache:", path.resolve())
    with np.load(path, allow_pickle=False) as archive:
        features, labels = take_subset(
            archive["x_train"], archive["y_train"],
            config["max_train_samples"], config["seed"],
        )
        test = take_subset(
            archive["x_test"], archive["y_test"],
            config["max_test_samples"], config["seed"] + config["seed_offsets"]["test_subset"],
        )
    if len(labels) < 2:
        raise ValueError("至少需要两个原训练样本")
    indices = np.random.default_rng(config["seed"] + config["seed_offsets"]["validation_split"]).permutation(len(labels))
    n_validation = min(len(labels) - 1, max(1, int(len(labels) * fraction)))
    validation_indices, train_indices = indices[:n_validation], indices[n_validation:]
    return {
        "train": (features[train_indices], labels[train_indices]),
        "validation": (features[validation_indices], labels[validation_indices]),
        "test": test,
    }
