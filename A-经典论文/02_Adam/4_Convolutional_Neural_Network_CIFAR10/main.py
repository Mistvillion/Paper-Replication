"""实验 4：Adam 论文第 6.3 节 CIFAR-10 卷积网络实验的统一入口。"""

from config import EXPERIMENT, OPTIMIZERS, REGIMES
from data import load_cifar10, resolve_path
from reporting import print_final_metrics, save_results
from training import configure_runtime, run_experiments, validate_config


def main() -> None:
    """校验配置和设备，准备 CIFAR-10 与 ZCA，运行六组对照并保存结果。"""
    # 先校验配置并确定设备，避免在下载数据之后才发现配置错误
    validate_config(EXPERIMENT, REGIMES, OPTIMIZERS)
    device = configure_runtime(EXPERIMENT)
    print("Device:", device)
    print("Convolution channels:", EXPERIMENT["channels"], "FC hidden units:", EXPERIMENT["hidden_units"])
    print("Teaching reproduction: reduced scale, no automatic hyperparameter search.")
    # 读取官方归档，抽样并划分训练/验证/测试集，仅用训练集拟合 ZCA
    data, metadata = load_cifar10(EXPERIMENT)
    print("Samples:", metadata["sample_counts"], flush=True)
    # 六组对照（有/无 Dropout × 三种优化器）共享划分与初始化，训练结束后才评估测试集
    results = run_experiments(data, EXPERIMENT, REGIMES, OPTIMIZERS, device)
    # 打印最终指标，保存 JSON 与曲线图
    print_final_metrics(results)
    save_results(
        results, EXPERIMENT, REGIMES, OPTIMIZERS, metadata,
        resolve_path(EXPERIMENT["output_dir"]),
    )


if __name__ == "__main__":
    main()
