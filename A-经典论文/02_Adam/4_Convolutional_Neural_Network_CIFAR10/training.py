"""控制公平对照、设备、随机流及训练记录；测试集只在全部训练完成后使用。"""

from math import ceil, isfinite
import os
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from evaluation import evaluate
from model import ConvNet, l2_penalty
from optimizers import Optimizer, validate_settings


def validate_config(config: dict, regimes: dict, settings: dict) -> None:
    """在数据下载之前检查实验配置及三种优化器的超参数。"""
    for key in (
        "epochs", "batch_size", "evaluation_batch_size", "preprocessing_batch_size",
        "hidden_units", "torch_threads", "evaluations_per_early_epoch",
    ):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    for key in ("seed", "detail_epochs"):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} 必须为非负整数")
    for key in ("max_train_samples", "max_test_samples"):
        value = config[key]
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{key} 必须为正整数或 None")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction 必须位于 (0, 1)")
    if config["max_train_samples"] is not None:
        count = config["max_train_samples"]
        if count - max(1, int(count * config["validation_fraction"])) < 2:
            raise ValueError("按当前样本上限划分后，至少应保留两个训练样本")
    if len(config["channels"]) != 3 or any(
        not isinstance(width, int) or width <= 0 for width in config["channels"]
    ):
        raise ValueError("channels 必须为三个正整数")
    if config["whitening"] not in {"zca", "none"}:
        raise ValueError("whitening 只支持 zca 或 none")
    if not isfinite(config["zca_epsilon"]) or config["zca_epsilon"] <= 0:
        raise ValueError("zca_epsilon 必须为有限正数")
    if not isfinite(config["l2_strength"]) or config["l2_strength"] < 0:
        raise ValueError("l2_strength 必须为有限非负数")
    if config["device"] not in {"cpu", "cuda", "mps"}:
        raise ValueError("device 只支持 cpu、cuda、mps")
    if not regimes or not settings:
        raise ValueError("至少需要一种正则化方案和一种优化器")
    for name, regime in regimes.items():
        if set(regime) != {"input_dropout", "hidden_dropout"}:
            raise ValueError(f"{name}: Dropout 配置键不匹配")
        if any(not 0 <= rate < 1 for rate in regime.values()):
            raise ValueError(f"{name}: Dropout 丢弃率必须位于 [0, 1)")
    for name, optimizer_settings in settings.items():
        validate_settings(name, optimizer_settings)


def configure_runtime(config: dict) -> torch.device:
    """设置随机种子和确定性选项；用户指定的加速设备不可用时明确报错。"""
    if config["deterministic"]:
        # 在初始化 CUDA 上下文前设置；默认 CPU 运行不使用此环境变量。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置指定 CUDA，但当前 PyTorch 无可用 CUDA 设备")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("配置指定 MPS，但当前 PyTorch 无可用 MPS 设备")
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config["deterministic"]
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def synchronize(device: torch.device) -> None:
    """计时边界同步加速设备，避免把异步队列提交时间误当成训练耗时。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate_checkpoint(
    model: ConvNet, train: tuple, validation: tuple,
    *, epoch: float, updates: int, training_seconds: float, config: dict, device: torch.device,
) -> dict:
    """固定当前参数评估完整训练/验证集，返回一个无 Dropout 的曲线点。"""
    options = {
        "batch_size": config["evaluation_batch_size"],
        "device": device,
        "l2_strength": config["l2_strength"],
    }
    train_metrics = evaluate(model, train, **options)
    validation_metrics = evaluate(model, validation, **options)
    return {
        "epoch": epoch,
        "updates": updates,
        "training_seconds": training_seconds,
        "train_nll": train_metrics["nll"],
        "train_objective": train_metrics["objective"],
        "train_accuracy": train_metrics["accuracy"],
        "validation_nll": validation_metrics["nll"],
        "validation_accuracy": validation_metrics["accuracy"],
    }


def train_one_optimizer(
    model: ConvNet, train: tuple, validation: tuple,
    *, name: str, settings: dict, config: dict, device: torch.device,
) -> dict:
    """训练一个已加载公共初值的模型，前三轮加密评估，其余每轮末评估一次。"""
    optimizer = Optimizer(name, model, settings)
    order_rng = np.random.default_rng(config["seed"] + 100)
    dropout_rng = torch.Generator(device="cpu").manual_seed(config["seed"] + 200)
    images, labels = train
    batch_count = ceil(len(labels) / config["batch_size"])
    early_interval = max(1, ceil(batch_count / config["evaluations_per_early_epoch"]))
    training_seconds = 0.0
    model.train()
    synchronize(device)
    started = perf_counter()
    history = [evaluate_checkpoint(
        model, train, validation, epoch=0.0, updates=0, training_seconds=0.0,
        config=config, device=device,
    )]
    for epoch in range(1, config["epochs"] + 1):
        synchronize(device)
        segment_started = perf_counter()
        permutation = order_rng.permutation(len(labels))
        seen = 0
        for batch_index, start in enumerate(range(0, len(labels), config["batch_size"]), start=1):
            indices = torch.from_numpy(permutation[start:start + config["batch_size"]])
            batch_images, batch_labels = images[indices].to(device), labels[indices].to(device)
            model.zero_grad(set_to_none=True)
            # Nesterov 只临时移动参数；反向传播结束后恢复基点，再执行真实更新。
            with optimizer.gradient_point():
                logits = model(batch_images, dropout_rng)
                loss = F.cross_entropy(logits, batch_labels) + l2_penalty(model, config["l2_strength"])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"{name}, epoch {epoch}: 训练目标非有限")
                loss.backward()
            optimizer.step()
            seen += len(indices)
            early_checkpoint = epoch <= config["detail_epochs"] and batch_index % early_interval == 0
            if early_checkpoint or seen == len(labels):
                synchronize(device)
                training_seconds += perf_counter() - segment_started
                # 横轴按实际已看过的样本数计算，最后一个较小批次也正好结束一轮。
                record = evaluate_checkpoint(
                    model, train, validation,
                    epoch=epoch - 1 + seen / len(labels), updates=optimizer.step_count,
                    training_seconds=training_seconds, config=config, device=device,
                )
                history.append(record)
                print(
                    f"{name:>9} | epoch {record['epoch']:6.3f}/{config['epochs']} | "
                    f"train NLL {record['train_nll']:.4f} | "
                    f"val NLL {record['validation_nll']:.4f} | "
                    f"val acc {record['validation_accuracy']:.4f}",
                    flush=True,
                )
                synchronize(device)
                segment_started = perf_counter()
    synchronize(device)
    elapsed_seconds = perf_counter() - started
    return {
        "history": history,
        "training_seconds": training_seconds,
        "elapsed_seconds": elapsed_seconds,
        # 暂存到 CPU，待所有方法完成训练后统一评估测试集；不写入 JSON。
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }


def run_experiments(data: dict, config: dict, regimes: dict, settings: dict, device: torch.device) -> dict:
    """六组训练共享一份初始化，训练完成后才评估每组末轮模型的测试指标。"""
    torch.manual_seed(config["seed"])
    template = ConvNet(config["channels"], config["hidden_units"], next(iter(regimes.values())))
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    del template
    results = {}
    for regime_name, regime in regimes.items():
        results[regime_name] = {}
        for name, optimizer_settings in settings.items():
            print(f"\n=== {regime_name} / {name} ===", flush=True)
            model = ConvNet(config["channels"], config["hidden_units"], regime)
            model.load_state_dict(initial_state)
            model.to(device)
            results[regime_name][name] = train_one_optimizer(
                model, data["train"], data["validation"], name=name,
                settings=optimizer_settings, config=config, device=device,
            )
            del model
    for regime_name, methods in results.items():
        for result in methods.values():
            model = ConvNet(config["channels"], config["hidden_units"], regimes[regime_name])
            model.load_state_dict(result.pop("state_dict"))
            model.to(device)
            result["test"] = evaluate(
                model, data["test"], batch_size=config["evaluation_batch_size"],
                device=device, l2_strength=config["l2_strength"],
            )
            del model
    return results
