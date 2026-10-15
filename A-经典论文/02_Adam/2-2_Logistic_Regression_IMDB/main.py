"""实验 2-2 统一入口"""

from config import EXPERIMENT, IMDB_OPTIMIZERS, validate_config
from data import load_imdb
from reporting import report_results
from training import run_optimizer_suite


def main() -> None:
    """检查配置、准备稀疏词袋、比较优化器并保存最终结果。"""
    validate_config(EXPERIMENT, IMDB_OPTIMIZERS)  # 检查配置
    data, summary = load_imdb(EXPERIMENT)  # 获取数据和数据摘要

    # 打印词表信息，训练集、验证集、测试集的统计信息
    print(f"IMDB：特征维度 {summary['feature_dimension']}，实际词表 {summary['vocabulary_words']}")
    for name, stats in summary["splits"].items():
        print(f"{name:10s} n={stats['samples']} positive={stats['positive_samples']} "
              f"density={stats['density']:.3%} empty={stats['empty_reviews']}")

    # 运行优化器对比实验
    print("比较 Adam / AdaGrad / RMSProp / Nesterov；曲线使用关闭 Dropout 的评估指标。")
    results = run_optimizer_suite(data, EXPERIMENT, IMDB_OPTIMIZERS)
    report_results(results, EXPERIMENT, IMDB_OPTIMIZERS, summary)


if __name__ == "__main__":
    main()
