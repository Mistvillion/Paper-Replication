"""NumPy 手写二分类逻辑回归；输入保持 CSR，参数与梯度使用稠密向量。"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit


def initialize_parameters(dimension: int) -> dict[str, np.ndarray]:
    """返回零初始化的权重 w[d] 与偏置 b[1]；线性模型无需随机打破对称性。"""
    return {"w": np.zeros(dimension, dtype=np.float64), "b": np.zeros(1, dtype=np.float64)}


def logits(features: csr_matrix, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """计算每条影评的 logit：Xw+b，返回形状 [B] 的向量。"""
    return np.asarray(features @ parameters["w"]).ravel() + parameters["b"][0]


def negative_log_likelihood(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """稳定计算逐样本二分类 NLL，避免显式计算 log(sigmoid(z))。"""
    return np.logaddexp(0.0, (1.0 - 2.0 * labels) * scores)


def loss_and_gradients(features: csr_matrix, labels: np.ndarray,
                       parameters: dict[str, np.ndarray], l2_strength: float) -> tuple:
    """返回平均 NLL 加 L2 的目标、数据 NLL 与解析梯度；偏置不正则化。"""
    scores = logits(features, parameters)
    nll = float(negative_log_likelihood(scores, labels).mean())
    residual = (expit(scores) - labels) / len(labels)
    gradients = {
        "w": np.asarray(features.T @ residual).ravel() + l2_strength * parameters["w"],
        "b": np.array([residual.sum()]),
    }
    objective = nll + 0.5 * l2_strength * float(parameters["w"] @ parameters["w"])
    return objective, nll, gradients
