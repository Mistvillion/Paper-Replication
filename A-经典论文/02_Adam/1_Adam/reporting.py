"""图像绘制"""

import matplotlib.pyplot as plt
import numpy as np


def plot_result(history, updates):
    """每列展示一个参数；上行为参数轨迹，下行为被减去的更新量。"""
    steps = np.arange(len(history))
    dimensions = history.shape[1]
    fig, axes = plt.subplots(2, dimensions, figsize=(5 * dimensions, 6), squeeze=False)

    for i in range(dimensions):
        label = rf"$\theta_{{{i + 1}}}$"
        axes[0, i].plot(steps, history[:, i], marker="o")
        axes[0, i].set_title(f"Parameter {label}")
        axes[0, i].set_ylabel("Parameter value")
        axes[0, i].set_xticks(steps)
        axes[0, i].ticklabel_format(axis="y", style="plain", useOffset=False)

        axes[1, i].bar(steps[1:], updates[:, i], color="tab:orange")
        axes[1, i].axhline(0, color="gray", linewidth=0.8)
        axes[1, i].set_title(f"Update for {label}")
        axes[1, i].set_ylabel(r"$\Delta_t = \theta_{t-1} - \theta_t$")
        axes[1, i].set_xticks(steps[1:])

    for ax in axes.flat:
        ax.set_xlabel("Step")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Adam with bias correction")
    fig.tight_layout()
    return fig