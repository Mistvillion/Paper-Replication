"""记录全部扫描配置与非有限失败，绘制偏差修正的学习率敏感性面板。"""

import json
from pathlib import Path
import platform

import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch


def print_summary(results: list, selected: list) -> None:
    """汇总扫描状态，打印按验证集所选学习率及其测试负 ELBO，不宣称某方法必胜。"""
    completed = sum(result["status"] == "completed" for result in results)
    print(f"\nTrials: {len(results)}, completed: {completed}, nonfinite: {len(results) - completed}")
    print("Selections by final validation negative ELBO (one learning rate per beta/correction group)")
    for entry in selected:
        if entry["trial"] is None:
            print(
                f"beta1={entry['beta1']}, beta2={entry['beta2']}, "
                f"correction={entry['bias_correction']}: no completed trial"
            )
            continue
        trial = entry["trial"]
        prefix = (
            f"beta1={trial['beta1']}, beta2={trial['beta2']}, correction={trial['bias_correction']}, "
            f"alpha={trial['learning_rate']:g}, val -ELBO={entry['validation']['negative_elbo']:.3f}"
        )
        if entry["test"] is None:
            print(prefix + f", test NONFINITE: {entry['test_failure']}")
        else:
            test = entry["test"]
            print(
                prefix + f", test -ELBO={test['negative_elbo']:.3f}, "
                f"reconstruction={test['reconstruction_nll']:.3f}, KL={test['kl']:.3f}"
            )


def plot_checkpoint(results: list, config: dict, sweep: dict, epoch: int):
    """按 beta1 行、beta2 列画学习率曲线；缺失的失败点断线，并在轴顶标记而非伪造损失。"""
    beta1_values, beta2_values = sweep["beta1_values"], sweep["beta2_values"]
    powers = sweep["log10_learning_rates"]
    lookup = {
        (
            result["trial"]["beta1"], result["trial"]["beta2"],
            result["trial"]["log10_learning_rate"], result["trial"]["bias_correction"],
        ): result
        for result in results
    }
    figure, axes = plt.subplots(
        len(beta1_values), len(beta2_values), squeeze=False,
        figsize=(4.6 * len(beta2_values), 3.4 * len(beta1_values) + 0.9),
    )
    colors = {True: "#d62728", False: "#2ca02c"}
    for row, beta1 in enumerate(beta1_values):
        for column, beta2 in enumerate(beta2_values):
            axis = axes[row, column]
            any_finite = False
            for correction in (True, False):
                losses, failures = [], []
                for power in powers:
                    result = lookup[(beta1, beta2, power, correction)]
                    checkpoint = result["checkpoints"].get(str(epoch))
                    if checkpoint is None:
                        if result["status"] != "nonfinite":
                            raise ValueError("已完成的试验缺少指定评估点，不能当成数值失败绘制")
                        losses.append(np.nan)  # 只用于断开图线，JSON 不写入 NaN。
                        failures.append(power)
                    else:
                        losses.append(checkpoint[config["plot_split"]]["negative_elbo"])
                        any_finite = True
                axis.plot(powers, losses, color=colors[correction], marker="o", markersize=4)
                if failures:
                    axis.plot(
                        failures, [0.97 if correction else 0.91] * len(failures),
                        linestyle="none", marker="x", markersize=7, color=colors[correction],
                        transform=axis.get_xaxis_transform(),
                    )
            if not any_finite:
                axis.text(0.5, 0.5, "No finite checkpoint", transform=axis.transAxes, ha="center")
            axis.set_title(f"beta1={beta1:g}, beta2={beta2:g}")
            axis.set_xlim(min(powers) - 0.2, max(powers) + 0.2)
            axis.set_xticks(powers)
            axis.set_xlabel("log10(alpha)")
            axis.set_ylabel("MC negative ELBO (nats/image)")
            axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"VAE bias correction: epoch {epoch}, {config['plot_split']} split, "
        f"L={config['evaluation_mc_samples']} evaluation samples",
    )
    figure.legend(handles=[
        Line2D([], [], color=colors[True], marker="o", label="Adam: bias correction"),
        Line2D([], [], color=colors[False], marker="o", label="Adam: no correction"),
        Line2D([], [], color="black", marker="x", linestyle="none", label="Nonfinite; marker height is not loss"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    return figure


def save_results(
    results: list, selected: list, config: dict, sweep: dict, metadata: dict, output_dir: Path,
) -> None:
    """保存全部配置、历史、评估点、失败原因和选择结果；不保存模型权重。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": config, "sweep": sweep, "data": metadata,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": str(torch.__version__), "matplotlib": matplotlib.__version__,
        },
        "metric_protocol": {
            "negative_elbo": "reconstruction NLL + analytic KL; nats/image; not exact marginal NLL",
            "checkpoints": "fixed model; fixed split-specific MC noise at every checkpoint/trial",
            "online_history": "sample-weighted pre-update batch losses with changing parameters/noise",
            "failure": "observed NaN/Inf only; finite large losses are retained without clipping",
            "elapsed_seconds": "synchronized training plus checkpoint evaluation/logging; no data or test",
            "selection": "final validation negative ELBO, completed trials only, separately per beta/correction group",
        },
        "trials": results, "selected_by_validation": selected,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print("\nSaved:", output_dir / "metrics.json")
    figures = []
    for epoch in config["checkpoint_epochs"]:
        figure = plot_checkpoint(results, config, sweep, epoch)
        path = output_dir / f"bias_correction_epoch_{epoch:03d}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        print("Saved:", path)
        if config["show_plots"]:
            figures.append(figure)
        else:
            plt.close(figure)
    if config["show_plots"]:
        plt.show()
        for figure in figures:
            plt.close(figure)
