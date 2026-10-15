"""打印末轮指标，保存六条对照曲线及可追溯的配置和评估记录。"""

import json
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def print_final_metrics(results: dict) -> None:
    """打印各方法的训练/验证/测试指标；不把未经调参的排序描述成论文结论。"""
    for regime, methods in results.items():
        print(f"\n{regime}: final metrics (Dropout OFF during evaluation)")
        print("optimizer   train NLL   val NLL   val acc   test NLL   test acc   train sec")
        for name, result in methods.items():
            last, test = result["history"][-1], result["test"]
            print(
                f"{name:10} {last['train_nll']:10.4f} {last['validation_nll']:9.4f} "
                f"{last['validation_accuracy']:9.4f} {test['nll']:10.4f} "
                f"{test['accuracy']:10.4f} {result['training_seconds']:11.2f}"
            )


def save_results(
    results: dict, config: dict, regimes: dict, settings: dict, metadata: dict,
    output_dir: Path,
) -> None:
    """保存 JSON 和 PNG，默认再显示图像；相同目录重复运行会覆盖旧输出。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": config,
        "regimes": regimes,
        "optimizers": settings,
        "data": metadata,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": str(torch.__version__),
            "matplotlib": matplotlib.__version__,
        },
        "metric_protocol": {
            "epoch": "cumulative training examples seen / training split size",
            "train_nll": "current-parameter whole-training-set mean NLL; dropout OFF; no L2",
            "train_objective": "train_nll plus coupled L2 penalty on non-bias weights",
            "training_seconds": "synchronized training-loop time, excludes preprocessing/evaluation",
            "elapsed_seconds": "training plus train/validation evaluation and progress output; no test",
            "selection": "fixed budget, final weights, no early stopping or test-based selection",
        },
        "results": results,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {name: f"C{index % 10}" for index, name in enumerate(settings)}
    line_styles = ("-", "--", "-.", ":")
    early_window = min(3.0, float(config["epochs"]))
    for regime_index, (regime, methods) in enumerate(results.items()):
        for name, result in methods.items():
            history = result["history"]
            epochs = [record["epoch"] for record in history]
            early = [record for record in history if record["epoch"] <= early_window]
            style = {
                "label": f"{name} / {regime}", "color": colors[name],
                "linestyle": line_styles[regime_index % len(line_styles)],
            }
            axes[0].plot(
                [record["epoch"] for record in early], [record["train_nll"] for record in early], **style,
            )
            # 仅对数绘图设置下界，原始 JSON 指标保持不变。
            axes[1].semilogy(epochs, np.maximum([r["train_nll"] for r in history], 1e-12), **style)
            axes[2].semilogy(epochs, np.maximum([r["validation_nll"] for r in history], 1e-12), **style)
    axes[0].set_xlim(0, early_window)
    for axis, title in zip(axes, ("Train: first 3 epochs", "Train: full budget", "Validation")):
        axis.set_title(f"CIFAR-10 CNN - {title}")
        axis.set_xlabel("Iterations over training split (epochs)")
        axis.set_ylabel("Mean NLL (Dropout OFF, no L2)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    print("\nSaved:", output_dir / "metrics.json")
    print("Saved:", output_dir / "training_curves.png")
    if config["show_plots"]:
        plt.show()
    plt.close(figure)
