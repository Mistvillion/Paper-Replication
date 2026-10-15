"""Adam、AdaGrad 和经典前瞻形式的 Nesterov SGD"""

from __future__ import annotations
import numpy as np


class Optimizer:
    """本实验所有优化器共享的最小接口。"""

    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        """保存参数和更新次数"""
        self.parameters = parameters
        self.step_count = 0

    def parameters_for_gradient(self) -> dict[str, np.ndarray]:
        """返回应当计算当前梯度的位置。只有NesterovSGD会重写这个函数"""
        return self.parameters

    def step(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """执行一次参数更新"""
        raise NotImplementedError


class NesterovSGD(Optimizer):
    """在 theta + momentum * velocity 的前瞻位置计算梯度"""

    def __init__(self, parameters: dict[str, np.ndarray], *, momentum: float = 0.9) -> None:
        """初始化动量速度；速度表示上一步的参数位移。"""
        super().__init__(parameters)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum 必须位于 [0, 1)")
        self.momentum = float(momentum)
        self.velocity = {
            name: np.zeros_like(parameter)
            for name, parameter in parameters.items()
        }

    def parameters_for_gradient(self) -> dict[str, np.ndarray]:
        """构造前瞻参数，不提前修改真实参数。"""
        return {
            name: parameter + self.momentum * self.velocity[name]
            for name, parameter in self.parameters.items()
        }

    def step(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """用前瞻位置的梯度更新速度，再把速度加到真实参数。"""
        self.step_count += 1
        for name in self.parameters:
            self.velocity[name] = (self.momentum * self.velocity[name] - learning_rate * gradients[name])
            self.parameters[name] += self.velocity[name]


class AdaGrad(Optimizer):
    """累计所有历史梯度平方的 AdaGrad"""

    def __init__(self, parameters: dict[str, np.ndarray], *, epsilon: float = 1e-8) -> None:
        """把逐坐标平方梯度累计量初始化为零。"""
        super().__init__(parameters)
        if epsilon <= 0.0:
            raise ValueError("epsilon 必须大于 0")
        self.epsilon = float(epsilon)
        self.squared_sum = {
            name: np.zeros_like(parameter)
            for name, parameter in parameters.items()
        }

    def step(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """累加梯度平方，并执行逐坐标归一化更新。"""
        # 累计量不会遗忘历史，所以有效学习率通常持续减小。
        self.step_count += 1
        for name in self.parameters:
            self.squared_sum[name] += gradients[name] ** 2
            self.parameters[name] -= (learning_rate * gradients[name] / (np.sqrt(self.squared_sum[name]) + self.epsilon))


class Adam(Optimizer):
    """论文 Algorithm 1 的标准 Adam，含一阶矩和二阶原始矩的偏差修正"""

    def __init__(self, parameters: dict[str, np.ndarray], *, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        """初始化矩估计及偏差修正所需时间步。"""
        super().__init__(parameters)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 和 beta2 必须位于 [0, 1)")
        if epsilon <= 0.0:
            raise ValueError("epsilon 必须大于 0")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.first_moment = {
            name: np.zeros_like(parameter)
            for name, parameter in parameters.items()
        }
        self.second_moment = {
            name: np.zeros_like(parameter)
            for name, parameter in parameters.items()
        }

    def step(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """更新并修正两个矩估计，然后执行 Adam 参数更新。"""
        # 时间步必须先加一，偏差修正的指数才从 t=1 开始。
        self.step_count += 1
        for name in self.parameters:
            gradient = gradients[name]
            self.first_moment[name] = (self.beta1 * self.first_moment[name] + (1.0 - self.beta1) * gradient)
            self.second_moment[name] = (self.beta2 * self.second_moment[name] + (1.0 - self.beta2) * gradient**2)
            first_hat = self.first_moment[name] / (1.0 - self.beta1**self.step_count)
            second_hat = self.second_moment[name] / (1.0 - self.beta2**self.step_count)
            self.parameters[name] -= (learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon))


def create_optimizer(name: str, parameters: dict[str, np.ndarray], settings: dict[str, float]) -> Optimizer:
    """按实验配置创建优化器；基础学习率由训练循环读取和调度。"""
    if name == "Adam":
        return Adam(
            parameters,
            beta1=settings.get("beta1", 0.9),
            beta2=settings.get("beta2", 0.999),
            epsilon=settings.get("epsilon", 1e-8),
        )
    if name == "AdaGrad":
        return AdaGrad(parameters, epsilon=settings.get("epsilon", 1e-8))
    if name == "Nesterov":
        return NesterovSGD(parameters, momentum=settings.get("momentum", 0.9))
    raise ValueError(f"未知优化器: {name!r}")
