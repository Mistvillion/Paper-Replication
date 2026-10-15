"""复用或下载 MNIST，在划分后进行一次固定 Bernoulli 二值化。"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def resolve_path(path: str | Path) -> Path:
    """相对路径以本实验目录为基准，与启动目录无关。"""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def get_mnist_file(data_dir: str, reuse_data_file: str | None) -> Path:
    """优先读取自己的缓存，再复用指定缓存；都没有才在主动运行时下载。"""
    destination = resolve_path(data_dir) / "mnist.npz"
    if destination.is_file():
        return destination
    if reuse_data_file is not None and resolve_path(reuse_data_file).is_file():
        return resolve_path(reuse_data_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.part")
    try:
        urlretrieve(MNIST_URL, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def take_subset(images: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    """在官方数据边界内固定抽样，None 表示全部；VAE 不使用类别标签。"""
    if limit is not None and limit <= 0:
        raise ValueError("样本上限必须为正整数或 None")
    if limit is None or limit >= len(images):
        return images
    indices = np.random.default_rng(seed).permutation(len(images))[:limit]
    return images[indices]


def binarize(images: np.ndarray, seed: int, batch_size: int) -> torch.Tensor:
    """把 [0,255] 灰度当作 Bernoulli 概率，每幅图像只采样一次并保存为 N×784。"""
    flat = images.reshape(len(images), 784)
    rng = np.random.default_rng(seed)
    binary = np.empty(flat.shape, dtype=np.float32)
    for start in range(0, len(flat), batch_size):
        probabilities = flat[start:start + batch_size].astype(np.float32) / 255.0
        binary[start:start + len(probabilities)] = rng.random(probabilities.shape) < probabilities
    return torch.from_numpy(binary)


def load_mnist(config: dict) -> tuple[dict[str, torch.Tensor], dict]:
    """仅拆分官方训练集；固定二值化数据在全部网格实验中复用，不逐轮重新抽样。"""
    path = get_mnist_file(config["data_dir"], config["reuse_data_file"])
    print("MNIST cache:", path.resolve(), flush=True)
    with np.load(path, allow_pickle=False) as archive:
        images = take_subset(archive["x_train"], config["max_train_samples"], config["seed"])
        test_images = take_subset(archive["x_test"], config["max_test_samples"], config["seed"] + config["seed_offsets"]["test_subset"])
    if len(images) < 3:
        raise ValueError("至少需要三个原训练样本")
    permutation = np.random.default_rng(config["seed"] + config["seed_offsets"]["validation_split"]).permutation(len(images))
    n_validation = max(1, int(len(images) * config["validation_fraction"]))
    if n_validation >= len(images):
        raise ValueError("验证划分不能占用全部原训练数据")
    raw = {
        "train": images[permutation[n_validation:]],
        "validation": images[permutation[:n_validation]],
        "test": test_images,
    }
    data = {
        split: binarize(values, config["seed"] + config["seed_offsets"]["binarization"] + index, config["evaluation_batch_size"])
        for index, (split, values) in enumerate(raw.items())
    }
    metadata = {
        "source_url": MNIST_URL,
        "sample_counts": {split: len(values) for split, values in data.items()},
        "binarization": "one fixed Bernoulli draw per pixel, before all trials",
        "class_labels_used": False,
    }
    return data, metadata
