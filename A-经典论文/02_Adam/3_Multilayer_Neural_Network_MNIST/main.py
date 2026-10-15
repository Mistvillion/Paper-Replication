"""实验 3：Adam 论文第 6.2 节的 MNIST 多层神经网络优化器对比，统一运行入口。"""

import matplotlib.pyplot as plt

from config import EXPERIMENT, OPTIMIZERS, REGIMES
from data import load_mnist, resolve_path
from reporting import print_final_metrics, save_results
from training import run_experiments, validate_config


def main() -> None:
    """校验配置，准备数据，训练两组优化器对照，再评估并保存结果。"""
    # 设置绘图风格
    plt.style.use("seaborn-v0_8-whitegrid")

    # 先校验配置，尽早报告错误
    validate_config(EXPERIMENT, REGIMES, OPTIMIZERS)

    # 加载并抽样 MNIST，划分训练/验证集，测试集保持独立
    data = load_mnist(EXPERIMENT)
    sample_counts = {name: len(labels) for name, (_, labels) in data.items()}
    print("Samples:", sample_counts)
    print(
        "Architecture:",
        (EXPERIMENT["input_size"], *EXPERIMENT["hidden_sizes"], EXPERIMENT["num_classes"]),
    )

    # 两组正则化方案 x 五种优化器，公共初值与批次顺序；训练结束后才评估测试集
    results = run_experiments(data, EXPERIMENT, REGIMES, OPTIMIZERS)

    # 打印最终指标，保存指标文件与曲线图
    print_final_metrics(results)
    save_results(
        results, EXPERIMENT, REGIMES, OPTIMIZERS, sample_counts,
        resolve_path(EXPERIMENT["output_dir"]),
    )


if __name__ == "__main__":
    main()
