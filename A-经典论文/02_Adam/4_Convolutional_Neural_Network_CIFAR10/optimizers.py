"""不调用 torch.optim，直接实现 Adam、AdaGrad 和前瞻形式 Nesterov SGD。"""

from contextlib import contextmanager
from math import isfinite

import torch
from torch import nn


def validate_settings(name: str, settings: dict) -> None:
    """检查优化器名称、必需键与超参数范围，防止误拼配置被静默忽略。"""
    required = {
        "Adam": {"learning_rate", "beta1", "beta2", "epsilon"},
        "AdaGrad": {"learning_rate", "epsilon"},
        "Nesterov": {"learning_rate", "momentum"},
    }
    if name not in required or set(settings) != required[name]:
        raise ValueError(f"未知方法或配置键不匹配：{name}")
    for key, value in settings.items():
        if not isfinite(value):
            raise ValueError(f"{name}.{key} 必须为有限数")
        if key in {"learning_rate", "epsilon"} and value <= 0:
            raise ValueError(f"{name}.{key} 必须大于零")
        if key in {"beta1", "beta2", "momentum"} and not 0 <= value < 1:
            raise ValueError(f"{name}.{key} 必须位于 [0, 1)")


class Optimizer:
    """最小更新接口；每次训练创建新实例，学习率恒定，状态不跨方法共享。"""

    def __init__(self, name: str, model: nn.Module, settings: dict):
        """保存可训练参数并创建所选方法需要的零状态，步数从零开始。"""
        validate_settings(name, settings)
        self.name = name
        self.settings = dict(settings)
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.step_count = 0
        self.first = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second = (
            [torch.zeros_like(parameter) for parameter in self.parameters]
            if name == "Adam" else []
        )

    @contextmanager
    def gradient_point(self):
        """Nesterov 临时移到前瞻点，退出时精确恢复；前向和 backward 必须都在上下文内。"""
        if self.name != "Nesterov":
            yield
            return
        with torch.no_grad():
            originals = [parameter.detach().clone() for parameter in self.parameters]
            for parameter, velocity in zip(self.parameters, self.first):
                parameter.add_(velocity, alpha=self.settings["momentum"])
        try:
            yield
        finally:
            # 用副本恢复，而不是减去位移，避免多次加减产生浮点漂移。
            with torch.no_grad():
                for parameter, original in zip(self.parameters, originals):
                    parameter.copy_(original)

    @torch.no_grad()
    def step(self) -> None:
        """使用已计算的梯度更新参数；Adam 在偏差修正后把 epsilon 放在根号外。"""
        for parameter in self.parameters:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"{self.name}: 梯度缺失或含非有限值")
        self.step_count += 1
        alpha = self.settings["learning_rate"]
        for index, parameter in enumerate(self.parameters):
            gradient = parameter.grad
            first = self.first[index]
            if self.name == "Adam":
                beta1, beta2 = self.settings["beta1"], self.settings["beta2"]
                second = self.second[index]
                first.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                first_hat = first / (1 - beta1**self.step_count)
                second_hat = second / (1 - beta2**self.step_count)
                parameter.addcdiv_(
                    first_hat, second_hat.sqrt() + self.settings["epsilon"], value=-alpha,
                )
            elif self.name == "AdaGrad":
                # 此时 first 是历史平方梯度累计量，不是动量。
                first.addcmul_(gradient, gradient)
                parameter.addcdiv_(gradient, first.sqrt() + self.settings["epsilon"], value=-alpha)
            else:
                # 梯度已经在 theta + mu * velocity 计算，这里不再做额外前瞻修正。
                first.mul_(self.settings["momentum"]).add_(gradient, alpha=-alpha)
                parameter.add_(first)
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError(f"{self.name}: 参数含非有限值，请检查学习率")
