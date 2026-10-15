"""固定数据顺序和输入 Dropout 随机序列，比较四种优化器。"""

from time import perf_counter

import numpy as np

from data import apply_input_dropout
from evaluation import evaluate
from model import initialize_parameters, loss_and_gradients
from optimizers import create_optimizer


def measure_epoch(data: dict, parameters: dict, config: dict, epoch: int,
                  learning_rate: float, seconds: float) -> dict:
    """记录无 Dropout 的训练与验证指标，不接触测试集。"""
    row = {"epoch": epoch, "learning_rate": learning_rate, "training_seconds": seconds}
    for split in ("train", "validation"):
        features, labels = data[split]
        metrics = evaluate(features, labels, parameters, config["l2_strength"], config["evaluation_batch_size"])
        row.update({f"{split}_{key}": value for key, value in metrics.items()})
    return row


def train_one_optimizer(name: str, settings: dict, initial: dict,
                        data: dict, config: dict) -> dict:
    """训练固定 epoch 数；各方法独立复制初始参数并重置相同随机种子。"""
    parameters = {key: value.copy() for key, value in initial.items()}
    optimizer = create_optimizer(name, parameters, settings)
    order_rng = np.random.default_rng(config["seed"] + 100)
    dropout_rng = np.random.default_rng(config["seed"] + 200)
    features, labels = data["train"]
    history = [measure_epoch(data, parameters, config, 0, settings["learning_rate"], 0.0)]
    history[0]["dropout_batch_nll"] = None
    history[0]["dropout_batch_objective"] = None
    elapsed = 0.0
    for epoch in range(1, config["epochs"] + 1):
        started = perf_counter()
        order = order_rng.permutation(len(labels))
        sum_nll, sum_objective = 0.0, 0.0
        for start in range(0, len(labels), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            batch = apply_input_dropout(features[indices], config["dropout_rate"], dropout_rng)
            # Nesterov 在前瞻位置求梯度；其余方法使用当前参数。
            objective, nll, gradients = loss_and_gradients(batch, labels[indices], optimizer.parameters_for_gradient(), config["l2_strength"])
            if not np.isfinite(objective) or not all(np.all(np.isfinite(g)) for g in gradients.values()):
                raise FloatingPointError(f"{name} 在 epoch={epoch} 出现非有限目标或梯度")
            step = optimizer.step_count + 1
            learning_rate = settings["learning_rate"]
            if config["learning_rate_decay"]:
                learning_rate /= np.sqrt(step)
            optimizer.step(gradients, learning_rate)
            if not all(np.all(np.isfinite(value)) for value in parameters.values()):
                raise FloatingPointError(f"{name} 在 step={step} 更新后参数非有限")
            sum_nll += nll * len(indices)
            sum_objective += objective * len(indices)
        elapsed += perf_counter() - started
        row = measure_epoch(data, parameters, config, epoch, float(learning_rate), elapsed)
        # 这些值来自变化中的参数及随机掩码，不等于 epoch 末完整训练集 NLL。
        row["dropout_batch_nll"] = sum_nll / len(labels)
        row["dropout_batch_objective"] = sum_objective / len(labels)
        history.append(row)
        print(f"{name:9s} epoch={epoch:3d} train NLL={row['train_nll']:.4f} "
              f"val NLL={row['validation_nll']:.4f} val acc={row['validation_accuracy']:.2%}", flush=True)
    return {"parameters": parameters, "history": history, "steps": optimizer.step_count}


def run_optimizer_suite(data: dict, config: dict, settings: dict) -> dict:
    """使用公共初始化完成所有训练，然后统一评估最终参数的测试集表现。"""
    initial = initialize_parameters(data["train"][0].shape[1])
    results = {name: train_one_optimizer(name, options, initial, data, config)
               for name, options in settings.items()}
    features, labels = data["test"]
    for result in results.values():
        result["test"] = evaluate(features, labels, result["parameters"],
                                  config["l2_strength"], config["evaluation_batch_size"])
    return results
