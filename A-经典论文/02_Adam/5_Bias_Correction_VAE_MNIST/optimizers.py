"""标准 Adam 与仅去掉两个偏差修正因子的消融，共享其余更新代码。"""

from math import isfinite

import torch
from torch import nn

from model import require_finite


class Adam:
    """不调用 torch.optim；bias_correction=False 不修正一阶或二阶矩的零初始化偏差。"""

    def __init__(
        self, model: nn.Module, *, learning_rate: float, beta1: float, beta2: float,
        epsilon: float, bias_correction: bool,
    ):
        """逐参数创建零矩估计，每次训练重新计步，不在 epoch 边界重置。"""
        if not all(isfinite(value) for value in (learning_rate, beta1, beta2, epsilon)):
            raise ValueError("Adam 超参数必须有限")
        if learning_rate <= 0 or epsilon <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("学习率和 epsilon 必须为正，beta 必须位于 [0,1)")
        self.parameters = list(model.parameters())
        self.first_moments = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second_moments = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.step_count = 0

    @torch.no_grad()
    def step(self) -> None:
        """更新矩后按开关决定是否除以 1-beta^t，稳定项始终在根号外。"""
        for parameter in self.parameters:
            if parameter.grad is None:
                raise RuntimeError("参数梯度缺失：这属于实现错误，不作为数值发散跳过")
            require_finite("gradient", parameter.grad)
        self.step_count += 1
        for parameter, first, second in zip(self.parameters, self.first_moments, self.second_moments):
            gradient = parameter.grad
            first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
            require_finite("moment estimates", first, second)
            if self.bias_correction:
                numerator = first / (1.0 - self.beta1**self.step_count)
                denominator = (second / (1.0 - self.beta2**self.step_count)).sqrt() + self.epsilon
            else:
                numerator = first
                denominator = second.sqrt() + self.epsilon
            require_finite("update terms", numerator, denominator)
            parameter.addcdiv_(numerator, denominator, value=-self.learning_rate)
            require_finite("updated parameter", parameter)
