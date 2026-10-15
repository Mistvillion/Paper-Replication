"""打印最终指标，保存配置、数据摘要、逐轮指标与优化器对比图。"""

import csv
from datetime import datetime
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import scipy

from data import resolve_path


def report_results(results: dict, config: dict, settings: dict, data_summary: dict) -> Path:
    """创建时间戳结果目录；记录环境与指标但不保存模型或原始影评。"""
    if not config["show_plots"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = resolve_path(config["output_dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment": config, "optimizers": settings, "data": data_summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }
    with (output / "config.json").open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2, allow_nan=False)
    rows = [{"optimizer": name, **row} for name, result in results.items() for row in result["history"]]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    print("\n最终指标（无 Dropout；时间仅统计训练循环，不含评估与数据处理）")
    for name, result in results.items():
        final = result["history"][-1]
        summary[name] = {"final": final, "test": result["test"], "steps": result["steps"]}
        print(f"{name:9s} train NLL={final['train_nll']:.4f} "
              f"val acc={final['validation_accuracy']:.2%} "
              f"test NLL={result['test']['nll']:.4f} test acc={result['test']['accuracy']:.2%} "
              f"train time={final['training_seconds']:.2f}s")
    with (output / "summary.json").open("w", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2, allow_nan=False)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, result in results.items():
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_nll"] for row in history], label=name)
        axes[1].plot(epochs, [row["validation_nll"] for row in history], label=name)
        axes[2].plot(epochs, [row["validation_accuracy"] for row in history], label=name)
    for axis, title, ylabel in zip(
        axes, ("Training NLL (dropout off)", "Validation NLL", "Validation accuracy"),
        ("NLL", "NLL", "Accuracy"),
    ):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("IMDB sparse BoW logistic regression")
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=160)
    if config["show_plots"]:
        plt.show()
    plt.close(figure)
    print(f"结果已保存到：{output}")
    return output
