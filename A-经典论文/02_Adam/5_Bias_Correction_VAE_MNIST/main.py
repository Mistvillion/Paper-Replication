"""实验 5：Adam 论文第 6.4 节 VAE 偏差修正消融的统一入口。"""

from config import EXPERIMENT, SWEEP
from data import load_mnist, resolve_path
from reporting import print_summary, save_results
from training import configure_runtime, run_sweep, validate_config


def main() -> None:
    """准备固定二值 MNIST，运行配对扫描，仅按验证集选学习率后再评估测试集。"""
    validate_config(EXPERIMENT, SWEEP)
    device = configure_runtime(EXPERIMENT)
    trials = 2 * len(SWEEP["beta1_values"]) * len(SWEEP["beta2_values"]) * len(SWEEP["log10_learning_rates"])
    print("Device:", device, "| Trials:", trials)
    print("VAE hidden units:", EXPERIMENT["hidden_units"], "| Latent dimension:", EXPERIMENT["latent_dim"])
    print("Teaching reproduction: reduced architecture/budget; losses are MC negative ELBO estimates.")
    data, metadata = load_mnist(EXPERIMENT)
    print("Samples:", metadata["sample_counts"], flush=True)
    results, selected = run_sweep(data, EXPERIMENT, SWEEP, device)
    print_summary(results, selected)
    save_results(results, selected, EXPERIMENT, SWEEP, metadata, resolve_path(EXPERIMENT["output_dir"]))


if __name__ == "__main__":
    main()
