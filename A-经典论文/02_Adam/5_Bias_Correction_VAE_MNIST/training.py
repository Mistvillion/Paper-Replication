"""配对超参数扫描、非有限失败记录与仅依据验证集的学习率选择。"""

from itertools import product
from math import isfinite
import os
from time import perf_counter

import numpy as np
import torch

from evaluation import evaluate
from model import NumericalFailure, VAE, loss_components, sample_noise
from optimizers import Adam


def validate_config(config: dict, sweep: dict) -> None:
    """在下载前验证配置，保证末轮有评估点且每个网格坐标唯一。"""
    for key in (
        "hidden_units", "latent_dim", "epochs", "batch_size", "evaluation_batch_size",
        "train_mc_samples", "evaluation_mc_samples", "torch_threads",
    ):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    for key in ("max_train_samples", "max_test_samples"):
        value = config[key]
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{key} 必须为正整数或 None")
    if config["max_train_samples"] is not None and config["max_train_samples"] < 3:
        raise ValueError("max_train_samples 至少为 3")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction 必须位于 (0,1)")
    if not isfinite(config["initial_weight_std"]) or config["initial_weight_std"] <= 0:
        raise ValueError("initial_weight_std 必须为有限正数")
    checkpoints = config["checkpoint_epochs"]
    if not checkpoints or any(
        not isinstance(epoch, int) or not 1 <= epoch <= config["epochs"] for epoch in checkpoints
    ):
        raise ValueError("checkpoint_epochs 必须为训练预算内的非空正整数序列")
    if list(checkpoints) != sorted(set(checkpoints)) or checkpoints[-1] != config["epochs"]:
        raise ValueError("checkpoint_epochs 必须严格递增并包含最后一轮")
    if config["device"] not in {"cpu", "cuda", "mps"} or config["plot_split"] not in {"train", "validation"}:
        raise ValueError("不支持的 device 或 plot_split")
    for key in ("beta1_values", "beta2_values", "log10_learning_rates"):
        values = sweep[key]
        if not values or not all(isfinite(value) for value in values) or len(values) != len(set(values)):
            raise ValueError(f"{key} 必须非空、有限且无重复")
    if any(not 0 <= beta < 1 for key in ("beta1_values", "beta2_values") for beta in sweep[key]):
        raise ValueError("beta 必须位于 [0,1)")
    powers = sweep["log10_learning_rates"]
    if list(powers) != sorted(powers):
        raise ValueError("log10_learning_rates 必须递增")
    for power in powers:
        try:
            learning_rate = 10.0**power
        except OverflowError as error:
            raise ValueError("学习率指数过大") from error
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("学习率必须为可表示的有限正数")
    if not isfinite(sweep["epsilon"]) or sweep["epsilon"] <= 0:
        raise ValueError("epsilon 必须为有限正数")


def configure_runtime(config: dict) -> torch.device:
    """设置种子、线程与确定性策略；指定后端不可用时停止，不自动切换设备。"""
    if config["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 没有可用 CUDA 设备")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前 PyTorch 没有可用 MPS 设备")
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config["deterministic"]
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def synchronize(device: torch.device) -> None:
    """在计时边界同步加速器，CPU 无需处理。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def make_model(config: dict) -> VAE:
    """统一从配置创建 CPU 模型，随后由调用者加载公共初值或所选权重。"""
    return VAE(config["hidden_units"], config["latent_dim"], config["initial_weight_std"])


def train_trial(
    model: VAE, train_images: torch.Tensor, validation_images: torch.Tensor,
    *, trial: dict, config: dict, device: torch.device,
) -> dict:
    """训练单个网格点；只捕获实际非有限数值，其他异常原样抛出。"""
    optimizer = Adam(
        model, learning_rate=trial["learning_rate"], beta1=trial["beta1"], beta2=trial["beta2"],
        epsilon=trial["epsilon"], bias_correction=trial["bias_correction"],
    )
    order_rng = np.random.default_rng(config["seed"] + config["seed_offsets"]["batch_order"])
    noise_rng = torch.Generator(device="cpu").manual_seed(config["seed"] + config["seed_offsets"]["train_noise"])
    result = {
        "trial": dict(trial), "status": "completed", "failure": None,
        "completed_epochs": 0, "successful_updates": 0, "history": [], "checkpoints": {},
    }
    evaluation_options = {
        "batch_size": config["evaluation_batch_size"],
        "mc_samples": config["evaluation_mc_samples"], "device": device,
    }
    model.train()
    epoch, stage = 0, "training"
    synchronize(device)
    started = perf_counter()
    try:
        for epoch in range(1, config["epochs"] + 1):
            stage = "training"
            sums = {"negative_elbo": 0.0, "reconstruction_nll": 0.0, "kl": 0.0}
            permutation = order_rng.permutation(len(train_images))
            for start in range(0, len(train_images), config["batch_size"]):
                indices = torch.from_numpy(permutation[start:start + config["batch_size"]])
                batch = train_images[indices].to(device)
                noise = sample_noise(config["train_mc_samples"], len(batch), model.latent_dim, noise_rng, device)
                model.zero_grad(set_to_none=True)
                terms = loss_components(model, batch, noise)
                terms["negative_elbo"].backward()
                optimizer.step()
                result["successful_updates"] += 1
                for name, value in terms.items():
                    sums[name] += len(batch) * value.item()
            result["completed_epochs"] = epoch
            # 在线值使用不断变化的参数和训练采样，不能与下面的固定参数评估混用。
            result["history"].append({
                "epoch": epoch,
                **{f"online_{name}": value / len(train_images) for name, value in sums.items()},
            })
            if epoch in config["checkpoint_epochs"]:
                stage = "evaluation/train"
                train_metrics = evaluate(model, train_images, seed=config["seed"] + config["seed_offsets"]["eval_train"], **evaluation_options)
                stage = "evaluation/validation"
                validation_metrics = evaluate(
                    model, validation_images, seed=config["seed"] + config["seed_offsets"]["eval_validation"], **evaluation_options,
                )
                result["checkpoints"][str(epoch)] = {"train": train_metrics, "validation": validation_metrics}
                print(
                    f"  epoch {epoch:03d} | train -ELBO {train_metrics['negative_elbo']:.3f} | "
                    f"val -ELBO {validation_metrics['negative_elbo']:.3f}", flush=True,
                )
    except NumericalFailure as error:
        result["status"] = "nonfinite"
        result["failure"] = {
            "epoch": epoch, "stage": stage,
            "successful_updates": result["successful_updates"], "reason": str(error),
        }
        print(f"  NONFINITE at epoch {epoch}, {stage}: {error}", flush=True)
    synchronize(device)
    result["elapsed_seconds"] = perf_counter() - started
    return result


def run_sweep(data: dict, config: dict, sweep: dict, device: torch.device) -> tuple[list, list]:
    """扫描全部配对网格；每组仅用末轮验证损失选择学习率，扫描后再测试所选模型。"""
    torch.manual_seed(config["seed"])
    template = make_model(config)
    initial_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    del template
    results, best = [], {}
    grid = product(sweep["beta1_values"], sweep["beta2_values"], sweep["log10_learning_rates"], (True, False))
    for trial_id, (beta1, beta2, power, correction) in enumerate(grid, start=1):
        trial = {
            "id": trial_id, "beta1": beta1, "beta2": beta2,
            "log10_learning_rate": power, "learning_rate": 10.0**power,
            "epsilon": sweep["epsilon"], "bias_correction": correction,
        }
        print(
            f"\n=== trial {trial_id}: beta1={beta1}, beta2={beta2}, "
            f"alpha={trial['learning_rate']:g}, correction={correction} ===", flush=True,
        )
        model = make_model(config)
        model.load_state_dict(initial_state)
        model.to(device)
        result = train_trial(
            model, data["train"], data["validation"], trial=trial, config=config, device=device,
        )
        results.append(result)
        key = (beta1, beta2, correction)
        if result["status"] == "completed":
            validation = result["checkpoints"][str(config["epochs"])]["validation"]
            score = validation["negative_elbo"]
            if key not in best or (score, trial["learning_rate"]) < (
                best[key]["validation"]["negative_elbo"], best[key]["trial"]["learning_rate"],
            ):
                best[key] = {
                    "trial": dict(trial), "validation": validation,
                    "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                }
        del model

    selected = []
    for beta1, beta2, correction in product(sweep["beta1_values"], sweep["beta2_values"], (True, False)):
        key = (beta1, beta2, correction)
        if key not in best:
            selected.append({
                "beta1": beta1, "beta2": beta2, "bias_correction": correction,
                "status": "no_completed_trial", "trial": None, "test": None,
            })
            continue
        entry = best[key]
        model = make_model(config)
        model.load_state_dict(entry.pop("state_dict"))
        model.to(device)
        try:
            entry["test"] = evaluate(
                model, data["test"], batch_size=config["evaluation_batch_size"],
                mc_samples=config["evaluation_mc_samples"], seed=config["seed"] + config["seed_offsets"]["eval_test"], device=device,
            )
            entry["status"] = "tested"
        except NumericalFailure as error:
            # 不根据测试失败改选次优学习率，否则测试集会参与选择。
            entry.update(status="test_nonfinite", test=None, test_failure=str(error))
        selected.append(entry)
        del model
    return results, selected
