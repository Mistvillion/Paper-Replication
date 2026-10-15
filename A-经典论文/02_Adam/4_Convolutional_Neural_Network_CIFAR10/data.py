"""读取官方 CIFAR-10 二进制归档，仅用训练集拟合整图 ZCA 白化。"""

import hashlib
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

import numpy as np
import torch

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def resolve_path(path: str | Path) -> Path:
    """将相对路径解析到当前实验目录，与命令启动目录无关。"""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def archive_is_valid(path: Path) -> bool:
    """分块检查官方归档的 MD5，用于发现缓存损坏，不作为安全签名。"""
    if not path.is_file():
        return False
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == CIFAR10_MD5


def get_archive(data_dir: str) -> Path:
    """复用校验通过的缓存；首次下载先写临时文件，校验成功后再替换。"""
    path = resolve_path(data_dir) / "cifar-10-binary.tar.gz"
    if path.exists():
        if not archive_is_valid(path):
            raise ValueError(f"CIFAR-10 缓存校验失败，请检查或移走该文件后重试：{path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        urlretrieve(CIFAR10_URL, temporary)
        if not archive_is_valid(temporary):
            raise ValueError("下载的 CIFAR-10 归档校验失败")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def read_binary_batch(archive: tarfile.TarFile, name: str) -> tuple[np.ndarray, np.ndarray]:
    """读取固定名称的 10000 条二进制记录，不解压到磁盘、不反序列化 pickle。"""
    member = archive.getmember(f"cifar-10-batches-bin/{name}")
    if not member.isfile() or member.size != 10000 * 3073:
        raise ValueError(f"CIFAR-10 批次大小或类型异常：{name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"无法读取批次：{name}")
    with handle:
        records = np.frombuffer(handle.read(), dtype=np.uint8).reshape(10000, 3073)
    labels = records[:, 0].astype(np.int64)
    if np.any(labels > 9):
        raise ValueError(f"标签超出 0–9：{name}")
    # 官方二进制依次保存 R、G、B 平面，直接重排为 NCHW，不是 NHWC。
    images = records[:, 1:].reshape(-1, 3, 32, 32).copy()
    return images, labels


def take_subset(
    images: np.ndarray, labels: np.ndarray, limit: int | None, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按固定种子从官方数据边界内抽样；None 或超过总量时保留全部。"""
    if limit is not None and limit <= 0:
        raise ValueError("样本上限必须为正整数或 None")
    if limit is None or limit >= len(labels):
        return images, labels
    indices = np.random.default_rng(seed).permutation(len(labels))[:limit]
    return images[indices], labels[indices]


def fit_zca(images: np.ndarray, epsilon: float, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """分块累计 3072 维均值和协方差，返回训练集拟合的均值与正则化 ZCA 矩阵。"""
    if len(images) < 2 or epsilon <= 0 or batch_size <= 0:
        raise ValueError("ZCA 需要至少两个训练样本、正 epsilon 和正批量大小")
    flat = images.reshape(len(images), -1)
    # 先按 [0,1] 像素尺度求均值；float64 累计避免协方差数值误差。
    mean = flat.mean(axis=0, dtype=np.float64) / 255.0
    covariance = np.zeros((flat.shape[1], flat.shape[1]), dtype=np.float64)
    for start in range(0, len(flat), batch_size):
        centered = flat[start:start + batch_size].astype(np.float64) / 255.0 - mean
        covariance += centered.T @ centered
    covariance /= len(flat)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # 小规模数据的协方差可能秩亏；截断负舍入误差，加 epsilon 限制放大。
    inverse_scale = 1.0 / np.sqrt(np.maximum(eigenvalues, 0.0) + epsilon)
    matrix = (eigenvectors * inverse_scale) @ eigenvectors.T
    if not np.isfinite(matrix).all():
        raise FloatingPointError("ZCA 矩阵含非有限值")
    return mean.astype(np.float32), matrix.astype(np.float32)


def transform_images(
    images: np.ndarray, mean: np.ndarray | None, matrix: np.ndarray | None,
    batch_size: int,
) -> torch.Tensor:
    """分批应用同一白化变换，返回 CPU 上连续的 N×3×32×32 float32 张量。"""
    result = np.empty((len(images), 3072), dtype=np.float32)
    flat = images.reshape(len(images), 3072)
    for start in range(0, len(images), batch_size):
        batch = flat[start:start + batch_size].astype(np.float32) / 255.0
        if matrix is not None:
            batch = (batch - mean) @ matrix
        if not np.isfinite(batch).all():
            raise FloatingPointError("预处理结果含非有限值")
        result[start:start + len(batch)] = batch
    return torch.from_numpy(result.reshape(-1, 3, 32, 32))


def load_cifar10(config: dict) -> tuple[dict, dict]:
    """先抽样和划分，再拟合白化；验证集和测试集绝不参与均值/协方差估计。"""
    path = get_archive(config["data_dir"])
    print("CIFAR-10 archive:", path.resolve(), flush=True)
    with tarfile.open(path, "r:gz") as archive:
        batches = [read_binary_batch(archive, f"data_batch_{i}.bin") for i in range(1, 6)]
        train_images = np.concatenate([images for images, _ in batches])
        train_labels = np.concatenate([labels for _, labels in batches])
        test_images, test_labels = read_binary_batch(archive, "test_batch.bin")
    del batches
    train_images, train_labels = take_subset(
        train_images, train_labels, config["max_train_samples"], config["seed"],
    )
    test_images, test_labels = take_subset(
        test_images, test_labels, config["max_test_samples"], config["seed"] + 1,
    )
    indices = np.random.default_rng(config["seed"] + 2).permutation(len(train_labels))
    n_validation = max(1, int(len(indices) * config["validation_fraction"]))
    validation_indices, train_indices = indices[:n_validation], indices[n_validation:]
    if len(train_indices) < 2:
        raise ValueError("划分后至少需要两个训练样本")
    raw = {
        "train": (train_images[train_indices], train_labels[train_indices]),
        "validation": (train_images[validation_indices], train_labels[validation_indices]),
        "test": (test_images, test_labels),
    }
    mean, matrix = None, None
    if config["whitening"] == "zca":
        print("Fitting 3072-dimensional ZCA on training split only...", flush=True)
        mean, matrix = fit_zca(
            raw["train"][0], config["zca_epsilon"], config["preprocessing_batch_size"],
        )
    elif config["whitening"] != "none":
        raise ValueError("whitening 只支持 zca 或 none")
    data = {
        name: (
            transform_images(images, mean, matrix, config["preprocessing_batch_size"]),
            torch.from_numpy(labels),
        )
        for name, (images, labels) in raw.items()
    }
    metadata = {
        "source_url": CIFAR10_URL,
        "archive_md5": CIFAR10_MD5,
        "sample_counts": {name: len(labels) for name, (_, labels) in data.items()},
        "whitening": config["whitening"],
        "zca_fit_samples": len(train_indices) if matrix is not None else 0,
        "zca_epsilon": config["zca_epsilon"] if matrix is not None else None,
    }
    return data, metadata
