"""显式实现五种一阶优化器；所有状态均属于当前一次训练。"""

import numpy as np


def validate_settings(name: str, settings: dict[str, float]) -> None:
    """在加载数据前检查方法名称、必需配置及超参数取值范围。"""
    required = {
        "Adam": {"learning_rate", "beta1", "beta2", "epsilon"},
        "AdaGrad": {"learning_rate", "epsilon"},
        "RMSProp": {"learning_rate", "rho", "epsilon"},
        "Nesterov": {"learning_rate", "momentum"},
        "AdaDelta": {"learning_rate", "rho", "epsilon"},
    }
    if name not in required:
        raise ValueError(f"未知优化器：{name}")
    if set(settings) != required[name]:
        raise ValueError(f"{name} 需要且仅接受这些配置：{sorted(required[name])}")
    for key, value in settings.items():
        if not np.isfinite(value):
            raise ValueError(f"{name}.{key} 必须有限")
        if key in {"learning_rate", "epsilon"} and value <= 0:
            raise ValueError(f"{name}.{key} 必须大于零")
        if key in {"beta1", "beta2", "rho", "momentum"} and not 0 <= value < 1:
            raise ValueError(f"{name}.{key} 必须位于 [0, 1)")


class Optimizer:
    """用名称选择更新公式，原地更新参数；不包含学习率调度与正则项。"""

    def __init__(self, name: str, parameters: dict[str, np.ndarray], settings: dict):
        """为所选优化器创建独立的零初始状态；时间步从零开始。"""
        validate_settings(name, settings)
        self.name = name
        self.parameters = parameters
        self.settings = dict(settings)
        self.step_count = 0
        # first：Adam 一阶矩或 Nesterov 位移；second：平方梯度累计量。
        self.first = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.second = {key: np.zeros_like(value) for key, value in parameters.items()}

    def parameters_for_gradient(self) -> dict[str, np.ndarray]:
        """仅 Nesterov 返回 theta + mu * velocity，其余方法直接在当前参数处求导。"""
        if self.name == "Nesterov":
            momentum = self.settings["momentum"]
            return {
                key: value + momentum * self.first[key]
                for key, value in self.parameters.items()
            }
        return self.parameters

    def step(self, gradients: dict[str, np.ndarray]) -> None:
        """执行一次恒定基础学习率的更新；Adam 的偏差修正跨 epoch 连续计数。"""
        self.step_count += 1
        settings = self.settings
        alpha = settings["learning_rate"]
        for key, parameter in self.parameters.items():
            gradient = gradients[key]
            if self.name == "Adam":
                beta1, beta2 = settings["beta1"], settings["beta2"]
                self.first[key] *= beta1
                self.first[key] += (1 - beta1) * gradient
                self.second[key] *= beta2
                self.second[key] += (1 - beta2) * gradient**2
                first_hat = self.first[key] / (1 - beta1**self.step_count)
                second_hat = self.second[key] / (1 - beta2**self.step_count)
                parameter -= alpha * first_hat / (np.sqrt(second_hat) + settings["epsilon"])
            elif self.name == "AdaGrad":
                self.second[key] += gradient**2
                parameter -= alpha * gradient / (np.sqrt(self.second[key]) + settings["epsilon"])
            elif self.name == "RMSProp":
                # 非 centered、无动量版本；epsilon 位于根号外。
                rho = settings["rho"]
                self.second[key] *= rho
                self.second[key] += (1 - rho) * gradient**2
                parameter -= alpha * gradient / (np.sqrt(self.second[key]) + settings["epsilon"])
            elif self.name == "Nesterov":
                # gradient 已在前瞻点计算，不能再额外做一次 Nesterov 修正。
                self.first[key] *= settings["momentum"]
                self.first[key] -= alpha * gradient
                parameter += self.first[key]
            else:  # AdaDelta：first 保存未乘外部学习率的更新平方 EMA。
                rho, epsilon = settings["rho"], settings["epsilon"]
                self.second[key] *= rho
                self.second[key] += (1 - rho) * gradient**2
                delta = -np.sqrt(self.first[key] + epsilon) / np.sqrt(self.second[key] + epsilon) * gradient
                parameter += alpha * delta
                self.first[key] *= rho
                self.first[key] += (1 - rho) * delta**2
