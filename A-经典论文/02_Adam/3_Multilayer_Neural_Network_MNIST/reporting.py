"""输出最终指标，保存可核查的配置、逐轮历史与对比曲线。"""

import json
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def print_final_metrics(results: dict) -> None:
    """打印各组末轮训练/验证指标及独立测试指标，不输出未经检验的胜负结论。"""
    for regime, methods in results.items():
        print(f"\n{regime}: final metrics (Dropout OFF)")
        print("optimizer   train NLL   val NLL   val acc   test NLL   test acc   train sec")
        for name, result in methods.items():
            last, test = result["history"][-1], result["test"]
            print(
                f"{name:10} {last['train_nll']:10.4f} {last['validation_nll']:9.4f} "
                f"{last['validation_accuracy']:9.4f} {test['nll']:10.4f} "
                f"{test['accuracy']:10.4f} {result['training_seconds']:11.2f}"
            )


def save_results(
    results: dict, config: dict, regimes: dict, settings: dict,
    sample_counts: dict[str, int], output_dir: Path,
) -> None:
    """运行时保存 JSON 与 PNG；不保存模型权重，重复运行覆盖同名输出。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        regime: {
            name: {key: value for key, value in result.items() if key != "parameters"}
            for name, result in methods.items()
        }
        for regime, methods in results.items()
    }
    payload = {
        "experiment": config,
        "regimes": regimes,
        "optimizers": settings,
        "sample_counts": sample_counts,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        },
        "metric_protocol": {
            "train_nll": "epoch-end whole-training-set NLL, dropout OFF, no L2",
            "online_nll": "sample-weighted pre-update minibatch NLL, dropout per regime",
            "online_objective": "online_nll plus minibatch-time coupled L2 penalty",
            "training_seconds": "cumulative minibatch loop only, excludes evaluation",
            "elapsed_seconds": "training plus epoch evaluations, excludes test/data/plots",
        },
        "results": serializable,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)

    figure, axes = plt.subplots(
        len(results), 3, figsize=(15, 4 * len(results)), squeeze=False,
    )
    for row, (regime, methods) in enumerate(results.items()):
        for name, result in methods.items():
            history = result["history"]
            epochs = [record["epoch"] for record in history]
            # 对数轴不能显示零；仅绘图时设置极小下界，不改变保存的指标。
            train_nll = np.maximum([record["train_nll"] for record in history], 1e-12)
            val_nll = np.maximum([record["validation_nll"] for record in history], 1e-12)
            axes[row, 0].semilogy(epochs, train_nll, label=name)
            axes[row, 1].semilogy(epochs, val_nll, label=name)
            axes[row, 2].semilogy(
                [record["training_seconds"] for record in history], train_nll, label=name,
            )
        for column, title in enumerate(("Train NLL", "Validation NLL", "Train NLL vs time")):
            axis = axes[row, column]
            axis.set_title(f"{regime}: {title} (Dropout OFF)")
            axis.set_xlabel("Training-loop seconds" if column == 2 else "Epoch")
            axis.set_ylabel("Mean NLL (no L2)")
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    print("\nSaved:", output_dir / "metrics.json")
    print("Saved:", output_dir / "training_curves.png")
    if config["show_plots"]:
        plt.show()
    plt.close(figure)
