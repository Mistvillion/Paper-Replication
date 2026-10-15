"""三段卷积—ReLU—池化网络；自动微分负责梯度，优化器由实验显式实现。"""

import torch
from torch import nn
from torch.nn import functional as F


def inverted_dropout(
    values: torch.Tensor, rate: float, training: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """训练时用独立 CPU 随机流生成逐元素掩码；推理时不加噪、不再缩放。"""
    if not training or rate == 0.0:
        return values
    if generator is None:
        raise ValueError("Dropout 训练需要显式随机数生成器")
    # 掩码从同一 CPU 随机流生成，再传到计算设备；不受模型初始化/评估影响。
    mask = torch.rand(values.shape, generator=generator, device="cpu") >= rate
    return values * mask.to(device=values.device, dtype=values.dtype) / (1.0 - rate)


class ConvNet(nn.Module):
    """32×32 彩色图像经三次下采样变为 4×4，再进入全连接隐藏层与十类输出层。"""

    def __init__(self, channels: tuple[int, int, int], hidden_units: int, regime: dict):
        """创建 5×5 卷积和 3×3/stride=2 池化，padding 选择使空间尺寸逐次减半。"""
        super().__init__()
        if len(channels) != 3 or any(width <= 0 for width in channels) or hidden_units <= 0:
            raise ValueError("需要三个正卷积通道数与正隐藏层宽度")
        self.input_dropout = regime["input_dropout"]
        self.hidden_dropout = regime["hidden_dropout"]
        if any(not 0 <= rate < 1 for rate in (self.input_dropout, self.hidden_dropout)):
            raise ValueError("Dropout 丢弃率必须在 [0,1) 内")
        sizes = (3, *channels)
        self.convolutions = nn.ModuleList([
            nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=5, stride=1, padding=2)
            for i in range(3)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.hidden = nn.Linear(channels[-1] * 4 * 4, hidden_units)
        self.output = nn.Linear(hidden_units, 10)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """统一采用 He 正态权重与零偏置；它是本实现的选择，不是论文明确设置。"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(
        self, images: torch.Tensor, dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """返回 B×10 logits；Dropout 只在白化后的输入及全连接隐藏层 ReLU 后使用。"""
        values = inverted_dropout(images, self.input_dropout, self.training, dropout_generator)
        for convolution in self.convolutions:
            values = self.pool(F.relu(convolution(values)))
        values = F.relu(self.hidden(torch.flatten(values, start_dim=1)))
        values = inverted_dropout(values, self.hidden_dropout, self.training, dropout_generator)
        return self.output(values)


def l2_penalty(model: nn.Module, strength: float) -> torch.Tensor:
    """返回 lambda/2 乘卷积及线性权重平方和，不惩罚偏置；默认系数为零。"""
    if strength == 0:
        return next(model.parameters()).new_zeros(())
    return 0.5 * strength * sum(
        parameter.square().sum() for parameter in model.parameters() if parameter.ndim > 1
    )
