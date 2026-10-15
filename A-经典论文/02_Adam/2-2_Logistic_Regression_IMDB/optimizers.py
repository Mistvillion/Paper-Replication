"""显式实现四种优化器；所有状态每步更新，不采用 lazy sparse Adam。"""

import numpy as np


class Optimizer:
    """共享参数引用和时间步；学习率调度由训练循环负责。"""

    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        """保存待更新参数，各方法持有独立状态。"""
        self.parameters = parameters
        self.step_count = 0

    def parameters_for_gradient(self) -> dict[str, np.ndarray]:
        """返回梯度计算位置，Nesterov 会覆盖此前瞻接口。"""
        return self.parameters

    def step(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """由具体算法执行一次原地参数更新。"""
        raise NotImplementedError

    def zeros(self) -> dict[str, np.ndarray]:
        """创建与各参数同形的零状态。"""
        return {name: np.zeros_like(value) for name, value in self.parameters.items()}


class Adam(Optimizer):
    """论文 Algorithm 1：一、二阶矩均做零初始化偏差修正。"""

    def __init__(self, parameters: dict, beta1: float, beta2: float, epsilon: float) -> None:
        """保存衰减率并将两个矩估计初始化为零。"""
        super().__init__(parameters)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.first_moment, self.second_moment = self.zeros(), self.zeros()

    def step(self, gradients: dict, learning_rate: float) -> None:
        """更新所有坐标的矩，再用修正后的矩归一化更新。"""
        self.step_count += 1
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            self.first_moment[name] *= self.beta1
            self.first_moment[name] += (1 - self.beta1) * gradient
            self.second_moment[name] *= self.beta2
            self.second_moment[name] += (1 - self.beta2) * gradient**2
            first_hat = self.first_moment[name] / (1 - self.beta1**self.step_count)
            second_hat = self.second_moment[name] / (1 - self.beta2**self.step_count)
            parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)


class AdaGrad(Optimizer):
    """累计历史梯度平方，不遗忘稀有词之前的更新。"""

    def __init__(self, parameters: dict, epsilon: float) -> None:
        """初始化平方梯度累计量。"""
        super().__init__(parameters)
        self.epsilon = epsilon
        self.squared_sum = self.zeros()

    def step(self, gradients: dict, learning_rate: float) -> None:
        """按每个坐标累计尺度缩放当前梯度。"""
        self.step_count += 1
        for name, parameter in self.parameters.items():
            self.squared_sum[name] += gradients[name]**2
            parameter -= learning_rate * gradients[name] / (np.sqrt(self.squared_sum[name]) + self.epsilon)


class RMSProp(Optimizer):
    """无动量、非中心化、不做偏差修正的 RMSProp。"""

    def __init__(self, parameters: dict, beta2: float, epsilon: float) -> None:
        """初始化梯度平方的指数移动平均。"""
        super().__init__(parameters)
        self.beta2, self.epsilon = beta2, epsilon
        self.second_moment = self.zeros()

    def step(self, gradients: dict, learning_rate: float) -> None:
        """用近期梯度的均方根尺度归一化，epsilon 放在根号外。"""
        self.step_count += 1
        for name, parameter in self.parameters.items():
            self.second_moment[name] *= self.beta2
            self.second_moment[name] += (1 - self.beta2) * gradients[name]**2
            parameter -= learning_rate * gradients[name] / (np.sqrt(self.second_moment[name]) + self.epsilon)


class NesterovSGD(Optimizer):
    """在 theta + momentum * velocity 位置计算梯度的经典 Nesterov。"""

    def __init__(self, parameters: dict, momentum: float) -> None:
        """将表示参数位移的速度初始化为零。"""
        super().__init__(parameters)
        self.momentum = momentum
        self.velocity = self.zeros()

    def parameters_for_gradient(self) -> dict[str, np.ndarray]:
        """构造前瞻位置，不提前修改真实参数。"""
        return {name: value + self.momentum * self.velocity[name]
                for name, value in self.parameters.items()}

    def step(self, gradients: dict, learning_rate: float) -> None:
        """先更新速度，再把位移加到真实参数。"""
        self.step_count += 1
        for name, parameter in self.parameters.items():
            self.velocity[name] *= self.momentum
            self.velocity[name] -= learning_rate * gradients[name]
            parameter += self.velocity[name]


def create_optimizer(name: str, parameters: dict, settings: dict) -> Optimizer:
    """根据已验证的配置构造优化器；不将基础学习率传给构造函数。"""
    classes = {"Adam": Adam, "AdaGrad": AdaGrad, "RMSProp": RMSProp, "Nesterov": NesterovSGD}
    arguments = {key: value for key, value in settings.items() if key != "learning_rate"}
    return classes[name](parameters, **arguments)
