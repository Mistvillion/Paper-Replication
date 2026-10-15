"""五层分支网络 (24 -> 6) + (12 -> 6) -> 12 -> 6 -> 24 的参数、前向传播、手写反向传播、Adam 与训练"""

# 包含的函数与类：
# sigmoid()：数值稳定的 Sigmoid
# initialize_parameters()：参数初始化
# forward()：前向传播
# loss_and_gradients()：加权二元交叉熵 + L2 的损失与解析梯度
# Adam：手写 Adam 优化器
# train()：全批量训练循环

import numpy as np

# 各参数的形状：人物分支、关系分支、合并层、隐藏层、输出层
PARAMETER_SHAPES = {
    "W_person": (24, 6), "b_person": (6,),
    "W_relation": (12, 6), "b_relation": (6,),
    "W_central": (12, 12), "b_central": (12,),
    "W_hidden": (12, 6), "b_hidden": (6,),
    "W_output": (6, 24), "b_output": (24,),
}


def sigmoid(x):
    """数值稳定的 Sigmoid，输入输出形状相同。"""
    x = np.asarray(x, dtype=np.float64)
    output = np.empty_like(x)
    positive = x >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    output[~positive] = exp_x / (1.0 + exp_x)
    return output


def initialize_parameters(seed, init_scale):
    """在 [-init_scale, init_scale] 内均匀采样，返回全部权重与偏置的字典。"""
    rng = np.random.default_rng(seed)
    return {
        name: rng.uniform(-init_scale, init_scale, size=shape)
        for name, shape in PARAMETER_SHAPES.items()
    }


def forward(parameters, person_batch, relation_batch):
    """前向传播。

    输入：person_batch 形状 (B, 24)，relation_batch 形状 (B, 12)。
    返回：(output, cache)，output 形状 (B, 24)，cache 保存反向传播所需的各层激活。
    """
    person_batch = np.asarray(person_batch, dtype=np.float64)
    relation_batch = np.asarray(relation_batch, dtype=np.float64)
    if person_batch.ndim != 2 or person_batch.shape[1] != 24:
        raise ValueError("人物输入形状应为 (样本数, 24)")
    if relation_batch.ndim != 2 or relation_batch.shape[1] != 12:
        raise ValueError("关系输入形状应为 (样本数, 12)")

    # 人物与关系分别编码为 6 维，拼接后再经过三层
    person_code = sigmoid(person_batch @ parameters["W_person"] + parameters["b_person"])
    relation_code = sigmoid(relation_batch @ parameters["W_relation"] + parameters["b_relation"])
    combined = np.concatenate([person_code, relation_code], axis=1)
    central = sigmoid(combined @ parameters["W_central"] + parameters["b_central"])
    hidden = sigmoid(central @ parameters["W_hidden"] + parameters["b_hidden"])
    output = sigmoid(hidden @ parameters["W_output"] + parameters["b_output"])

    cache = {
        "person_input": person_batch, "relation_input": relation_batch,
        "person_code": person_code, "relation_code": relation_code,
        "combined": combined, "central": central, "hidden": hidden,
    }
    return output, cache


def loss_and_gradients(parameters, person_batch, relation_batch, target, *, positive_weight, l2_strength):
    """加权二元交叉熵 + L2：正例权重为 positive_weight，只惩罚权重不惩罚偏置。

    返回：(含 L2 的总损失, 不含 L2 的数据损失, 梯度字典)；target 形状 (B, 24)。
    """
    output, cache = forward(parameters, person_batch, relation_batch)
    weights = np.where(target == 1.0, positive_weight, 1.0)
    gradients = {}

    # Sigmoid 与交叉熵组合后，输出层误差信号化简为 (output - target) * weights / B
    delta_output = (output - target) * weights / len(target)
    gradients["W_output"] = cache["hidden"].T @ delta_output
    gradients["b_output"] = delta_output.sum(axis=0)

    # 逐层向前传播误差，乘以 Sigmoid 导数 a * (1 - a)
    delta_hidden = delta_output @ parameters["W_output"].T
    delta_hidden *= cache["hidden"] * (1.0 - cache["hidden"])
    gradients["W_hidden"] = cache["central"].T @ delta_hidden
    gradients["b_hidden"] = delta_hidden.sum(axis=0)

    delta_central = delta_hidden @ parameters["W_hidden"].T
    delta_central *= cache["central"] * (1.0 - cache["central"])
    gradients["W_central"] = cache["combined"].T @ delta_central
    gradients["b_central"] = delta_central.sum(axis=0)

    # 拼接层的误差信号按列拆回人物分支（前 6 列）与关系分支（后 6 列）
    delta_combined = delta_central @ parameters["W_central"].T
    delta_person = delta_combined[:, :6] * cache["person_code"] * (1.0 - cache["person_code"])
    delta_relation = delta_combined[:, 6:] * cache["relation_code"] * (1.0 - cache["relation_code"])
    gradients["W_person"] = cache["person_input"].T @ delta_person
    gradients["b_person"] = delta_person.sum(axis=0)
    gradients["W_relation"] = cache["relation_input"].T @ delta_relation
    gradients["b_relation"] = delta_relation.sum(axis=0)

    # L2 梯度只加在权重上
    for name in parameters:
        if name.startswith("W_"):
            gradients[name] += l2_strength * parameters[name]

    clipped = np.clip(output, 1e-12, 1.0 - 1e-12)
    data_loss = -np.mean(np.sum(
        positive_weight * target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped),
        axis=1,
    ))
    penalty = 0.5 * l2_strength * sum(
        np.sum(value ** 2) for name, value in parameters.items() if name.startswith("W_")
    )
    return float(data_loss + penalty), float(data_loss), gradients


class Adam:
    """Adam：一阶矩与二阶矩的指数滑动平均，带偏差修正。"""

    def __init__(self, parameters, *, learning_rate, beta1, beta2, epsilon):
        """保存参数字典（原地更新）并将两个矩估计初始化为零。"""
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.v = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_count = 0

    def step(self, gradients):
        """用梯度字典执行一次 Adam 更新。"""
        self.step_count += 1
        for name in self.parameters:
            gradient = gradients[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * gradient
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * gradient ** 2
            # 偏差修正，抵消零初始化带来的低估
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.step_count)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.step_count)
            self.parameters[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def train(person_inputs, relation_inputs, targets, *, experiment, adam_settings):
    """在训练集上做全批量训练，每个 epoch 更新一次参数。

    返回：(训练后的参数字典, 每个 epoch 的数据损失列表)。
    """
    parameters = initialize_parameters(experiment["seed"], experiment["init_scale"])
    optimizer = Adam(parameters, **adam_settings)
    data_losses = []
    for _ in range(experiment["epochs"]):
        _, data_loss, gradients = loss_and_gradients(
            parameters, person_inputs, relation_inputs, targets,
            positive_weight=experiment["positive_weight"],
            l2_strength=experiment["l2_strength"],
        )
        optimizer.step(gradients)
        data_losses.append(data_loss)
    return parameters, data_losses
