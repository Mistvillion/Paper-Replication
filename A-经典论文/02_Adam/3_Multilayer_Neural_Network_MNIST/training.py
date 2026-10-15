"""在相同初始化、样本顺序和 Dropout 掩码下比较优化器。"""

from time import perf_counter

import numpy as np

from evaluation import evaluate
from model import initialize_parameters, loss_and_gradients
from optimizers import Optimizer, validate_settings


def validate_config(config: dict, regimes: dict, optimizer_settings: dict) -> None:
    """检查可修改配置，尽量在下载数据或分配大数组之前报告错误。"""
    for key in ("epochs", "batch_size", "evaluation_batch_size"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    for key in ("max_train_samples", "max_test_samples"):
        value = config[key]
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{key} 必须为正整数或 None")
    sizes = config["hidden_sizes"]
    if len(sizes) != 2 or any(not isinstance(size, int) or size <= 0 for size in sizes):
        raise ValueError("hidden_sizes 必须为两个正整数")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction 必须位于 (0, 1)")
    if not regimes or not optimizer_settings:
        raise ValueError("至少需要一种正则化方案和一种优化器")
    for name, regime in regimes.items():
        rates, strength = regime["dropout_rates"], regime["l2_strength"]
        if len(rates) != 3 or any(not 0 <= rate < 1 for rate in rates):
            raise ValueError(f"{name}: dropout_rates 必须为三个 [0, 1) 内的概率")
        if not np.isfinite(strength) or strength < 0:
            raise ValueError(f"{name}: l2_strength 必须有限且非负")
    for name, settings in optimizer_settings.items():
        validate_settings(name, settings)


def train_one_optimizer(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    initial_parameters: dict[str, np.ndarray],
    *,
    name: str,
    settings: dict,
    regime: dict,
    config: dict,
) -> dict:
    """训练一个方法，记录 epoch 0 与各轮末指标，不读取测试集或提前停止。"""
    parameters = {key: value.copy() for key, value in initial_parameters.items()}
    optimizer = Optimizer(name, parameters, settings)
    # 排列与掩码使用独立随机流；各方法重新创建相同种子的随机流。
    order_rng = np.random.default_rng(config["seed"] + config["seed_offsets"]["batch_order"])
    dropout_rng = np.random.default_rng(config["seed"] + config["seed_offsets"]["dropout"])
    train_features, train_labels = train
    evaluation_options = {
        "batch_size": config["evaluation_batch_size"],
        "l2_strength": regime["l2_strength"],
    }
    history = []
    training_seconds = 0.0
    started = perf_counter()
    for epoch in range(config["epochs"] + 1):
        online_nll, online_objective = None, None
        if epoch > 0:
            update_started = perf_counter()
            permutation = order_rng.permutation(len(train_labels))
            nll_sum, objective_sum = 0.0, 0.0
            for start in range(0, len(train_labels), config["batch_size"]):
                indices = permutation[start:start + config["batch_size"]]
                objective, nll, gradients = loss_and_gradients(
                    train_features[indices], train_labels[indices],
                    optimizer.parameters_for_gradient(),
                    l2_strength=regime["l2_strength"],
                    dropout_rates=regime["dropout_rates"], rng=dropout_rng,
                )
                if not np.isfinite(objective) or any(
                    not np.all(np.isfinite(value)) for value in gradients.values()
                ):
                    raise FloatingPointError(f"{name} epoch {epoch}: 损失/梯度非有限，请降低学习率")
                optimizer.step(gradients)
                if any(not np.all(np.isfinite(value)) for value in parameters.values()):
                    raise FloatingPointError(f"{name} epoch {epoch}: 参数非有限，请降低学习率")
                nll_sum += len(indices) * nll
                objective_sum += len(indices) * objective
            training_seconds += perf_counter() - update_started
            online_nll = nll_sum / len(train_labels)
            online_objective = objective_sum / len(train_labels)

        # 这里固定参数并关闭 Dropout，不把在线加噪损失当成整训练集 NLL。
        train_metrics = evaluate(*train, parameters, **evaluation_options)
        validation_metrics = evaluate(*validation, parameters, **evaluation_options)
        history.append({
            "epoch": epoch,
            "updates": optimizer.step_count,
            "training_seconds": training_seconds,
            "train_nll": train_metrics["nll"],
            "train_objective": train_metrics["objective"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_nll": validation_metrics["nll"],
            "validation_accuracy": validation_metrics["accuracy"],
            "online_nll": online_nll,
            "online_objective": online_objective,
        })
        print(
            f"{name:>9} | epoch {epoch:03d}/{config['epochs']:03d} | "
            f"train NLL {train_metrics['nll']:.4f} | "
            f"val NLL {validation_metrics['nll']:.4f} | "
            f"val acc {validation_metrics['accuracy']:.4f}",
            flush=True,
        )
    return {
        "parameters": parameters,
        "history": history,
        "training_seconds": training_seconds,
        "elapsed_seconds": perf_counter() - started,
    }


def run_experiments(data: dict, config: dict, regimes: dict, settings: dict) -> dict:
    """训练所有正则化方案与优化器组合，然后统一进行独立测试集评估。"""
    validate_config(config, regimes, settings)
    initial_parameters = initialize_parameters(
        config["input_size"], config["hidden_sizes"], config["num_classes"], config["seed"]
    )
    results = {}
    for regime_name, regime in regimes.items():
        results[regime_name] = {}
        for name, optimizer_settings in settings.items():
            print(f"\n=== {regime_name} / {name} ===")
            results[regime_name][name] = train_one_optimizer(
                data["train"], data["validation"], initial_parameters,
                name=name, settings=optimizer_settings, regime=regime, config=config,
            )
    # 不根据测试集选学习率、epoch 或正则化方案。
    for regime_name, methods in results.items():
        for result in methods.values():
            result["test"] = evaluate(
                *data["test"], result["parameters"],
                batch_size=config["evaluation_batch_size"],
                l2_strength=regimes[regime_name]["l2_strength"],
            )
    return results
