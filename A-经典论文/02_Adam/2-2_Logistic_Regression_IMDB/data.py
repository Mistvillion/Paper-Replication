"""负责整个实验的数据准备，以及训练时使用的输入Dropout"""

from collections import Counter
import html
from pathlib import Path
import re
import tarfile
from urllib.request import urlopen
import shutil

import numpy as np
from scipy.sparse import csr_matrix

# 定义词的形式
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def resolve_path(value: str) -> Path:
    """解析数据文件路径"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def ensure_archive(config: dict) -> Path:
    """确保本地有数据压缩包，若没有则下载"""
    archive = resolve_path(config["data_dir"]) / config["archive_name"]  # 数据压缩包路径
    if archive.is_file():
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    print(f"下载 IMDB 官方完整压缩包到 {archive}", flush=True)
    try:
        with urlopen(config["dataset_url"], timeout=60) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        # 不解压到磁盘，避免压缩包成员路径参与文件写入。
        with tarfile.open(partial, "r:gz") as source:
            source.getmembers()
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    return archive


def tokenize(text: str) -> list[str]:
    """把影评转成词列表"""
    text = html.unescape(text)
    return TOKEN_PATTERN.findall(re.sub(r"<[^>]+>", " ", text).lower())


def balanced_sample(
        indices: np.ndarray, 
        labels: np.ndarray, 
        maximum: int | None,
        rng: np.random.Generator
    ) -> np.ndarray:
    """从候选样本中随机挑出一些样本，正类和负类各占一半，最后返回这些样本的索引"""
    # 确定抽样总数
    if maximum is None: total = len(indices)
    else: total = min(maximum, len(indices))
    selected = []

    # 正类样本label=1，负类样本label=0；给正负两类分配样本
    # 若抽样总数total为偶数，则正负类各占一半；若抽样总数total为奇数，则正类比负类多一个
    for label, count in ((0, total // 2), (1, total - total // 2)):
        # 每个类别内部随机抽样
        group = indices[labels[indices] == label]
        if len(group) < count:
            raise ValueError("数据类别数量不足以进行等额抽样")

        # 随机打乱后取前count个
        shuffled = rng.permutation(group)
        chosen = shuffled[:count]
        selected.append(chosen)

    # 最后合并正负类，再打乱一次
    return rng.permutation(np.concatenate(selected))


def read_reviews(archive: Path, config: dict) -> dict:
    """从压缩包中读取选中的影评样本"""
    rng = np.random.default_rng(config["seed"])
    reviews = {}
    with tarfile.open(archive, "r:gz") as source:
        members = [member for member in source.getmembers() if member.isfile()]
        for split, maximum in (("train", config["max_train_samples"]), ("test", config["max_test_samples"])):
            chosen_members = sorted(
                [member for member in members
                 if re.fullmatch(rf"aclImdb/{split}/(neg|pos)/[^/]+\.txt", member.name)],
                key=lambda member: member.name,
            )
            labels = np.array([int("/pos/" in member.name) for member in chosen_members], dtype=np.int64)
            if not len(labels) or set(labels.tolist()) != {0, 1}:
                raise ValueError(f"压缩包缺少 {split} 正负类影评，请检查缓存来源")
            indices = balanced_sample(np.arange(len(labels)), labels, maximum, rng)
            documents = [[] for _ in indices]
            # 按压缩包内偏移读取，避免随机回退导致 gzip 反复从头解压；
            # 写回抽样顺序，保持文档与 labels[indices] 一一对应。
            positions = sorted(range(len(indices)), key=lambda position: chosen_members[int(indices[position])].offset_data)
            for position in positions:
                member = chosen_members[int(indices[position])]
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError("无法读取影评成员")
                with stream:
                    documents[position] = tokenize(stream.read().decode("utf-8", errors="replace"))
            reviews[split] = (documents, labels[indices])
    return reviews


def make_bow(
        documents: list[list[str]], 
        vocabulary: dict[str, int],
        dimension: int, 
        binary: bool
    ) -> csr_matrix:
    """生成 N×d 的 CSR 词袋；未知词忽略，空文档保留为全零行。"""
    indptr, indices, values = [0], [], []
    for words in documents:
        counts = Counter(vocabulary[word] for word in words if word in vocabulary)
        for column, count in sorted(counts.items()):
            indices.append(column)
            values.append(1.0 if binary else float(count))
        indptr.append(len(indices))
    return csr_matrix((np.asarray(values, dtype=np.float64),
                       np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int32)), shape=(len(documents), dimension))


def load_imdb(config: dict) -> tuple[dict, dict]:
    """划分验证集后仅在训练集统计词频，返回稀疏数据及可记录的数据摘要。"""
    archive = ensure_archive(config)
    reviews = read_reviews(archive, config)
    documents, labels = reviews["train"]
    rng = np.random.default_rng(config["seed"] + 1)
    train_indices, validation_indices = [], []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = max(1, int(round(len(indices) * config["validation_fraction"])))
        count = min(count, len(indices) - 1)
        if count < 1:
            raise ValueError("每类至少需要两个训练池样本以划分验证集")
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    train_indices = rng.permutation(train_indices)
    validation_indices = rng.permutation(validation_indices)
    train_documents = [documents[index] for index in train_indices]
    frequencies = Counter(word for words in train_documents for word in words)
    # 同频词按字母序排序，避免词表受字典插入顺序影响。
    words = sorted(frequencies, key=lambda word: (-frequencies[word], word))[:config["vocabulary_size"]]
    if not words:
        raise ValueError("训练语料没有可用单词")
    vocabulary = {word: index for index, word in enumerate(words)}
    splits = {
        "train": (train_documents, labels[train_indices]),
        "validation": ([documents[index] for index in validation_indices], labels[validation_indices]),
        "test": reviews["test"],
    }
    data = {
        name: (make_bow(texts, vocabulary, config["vocabulary_size"], config["binary_bow"]), targets)
        for name, (texts, targets) in splits.items()
    }
    summary = {"vocabulary_words": len(vocabulary), "feature_dimension": config["vocabulary_size"],
               "archive": str(archive), "splits": {}}
    for name, (features, targets) in data.items():
        summary["splits"][name] = {
            "samples": len(targets), "positive_samples": int(targets.sum()),
            "density": features.nnz / (features.shape[0] * features.shape[1]),
            "empty_reviews": int(np.count_nonzero(np.diff(features.indptr) == 0)),
        }
    return data, summary


def apply_input_dropout(features: csr_matrix, rate: float,
                        rng: np.random.Generator) -> csr_matrix:
    """仅对非零词袋元素采样 inverted Dropout，不稠密化且不修改原始数据。"""
    result = features.copy()
    if rate:
        keep_probability = 1.0 - rate
        mask = rng.random(result.nnz) < keep_probability
        result.data *= mask / keep_probability
        result.eliminate_zeros()
    return result
