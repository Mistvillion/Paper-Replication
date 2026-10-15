"""实验 3：家族树关系补全，用 (人物, 关系) 查询预测答案人物，统一运行入口"""

import numpy as np

from config import ADAM, EXPERIMENT
from data import PEOPLE, build_dataset, encode_queries
from model import forward, train


def main():
    """构造数据、训练网络、评估训练集与留出集，并打印演示查询的预测。"""
    np.set_printoptions(precision=6, suppress=True)

    # 构造数据并划分训练集（100 条）与留出集（4 条）
    person_inputs, relation_inputs, targets, train_indices, holdout_indices, answers_by_query = build_dataset()

    # 仅用训练集全批量训练
    parameters, data_losses = train(
        person_inputs[train_indices], relation_inputs[train_indices], targets[train_indices],
        experiment=EXPERIMENT, adam_settings=ADAM,
    )
    print(f"最终训练数据损失（不含 L2）: {data_losses[-1]:.6f}")

    # exact match：24 个输出全部与目标一致才算这条查询正确
    for name, indices in (("训练集", train_indices), ("留出集", holdout_indices)):
        output, _ = forward(parameters, person_inputs[indices], relation_inputs[indices])
        matched = np.all((output >= 0.5) == (targets[indices] == 1.0), axis=1)
        print(f"{name} exact match: {int(matched.sum())}/{len(indices)}")

    # 演示查询
    queries = EXPERIMENT["test_queries"]
    query_person, query_relation = encode_queries(queries)
    scores, _ = forward(parameters, query_person, query_relation)
    for query, row_scores in zip(queries, scores):
        # 得分不低于 0.5 的人物视为预测答案，按得分降序排列
        indices = np.flatnonzero(row_scores >= 0.5)
        indices = indices[np.argsort(row_scores[indices])[::-1]]
        shown = [f"{PEOPLE[i]} ({row_scores[i]:.4f})" for i in indices]
        expected = sorted(answers_by_query.get(query, set()))
        print(f"{query} -> 预测: {shown or ['无']}；已知答案: {expected or ['无']}")


if __name__ == "__main__":
    main()
