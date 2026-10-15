"""单隐藏层 Softplus VAE，显式重参数化及 Bernoulli 重构项、解析 KL。"""

import torch
from torch import nn
from torch.nn import functional as F


class NumericalFailure(FloatingPointError):
    """仅表示实际观测到的非有限数值，供扫描流程记录失败而不掩盖其他异常。"""


def require_finite(name: str, *values: torch.Tensor) -> None:
    """遇到 NaN/Inf 抛出数值失败；不根据任意损失阈值把有限的大损失判为发散。"""
    if any(not torch.isfinite(value).all().item() for value in values):
        raise NumericalFailure(f"{name} contains NaN or Inf")


def sample_noise(
    samples: int, batch_size: int, latent_dim: int,
    generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    """从独立 CPU 随机流抽取 L×B×d 标准正态噪声，再转到计算设备。"""
    return torch.randn(
        samples, batch_size, latent_dim, generator=generator, device="cpu", dtype=torch.float32,
    ).to(device)


class VAE(nn.Module):
    """编码器与解码器各有一个 Softplus 隐藏层；先验为标准正态，后验为对角高斯。"""

    def __init__(self, hidden_units: int, latent_dim: int, initial_weight_std: float):
        """创建 784-h-(mu,logvar) 编码器及 d-h-784 Bernoulli 解码器。"""
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_hidden = nn.Linear(784, hidden_units)
        self.mean_head = nn.Linear(hidden_units, latent_dim)
        self.log_variance_head = nn.Linear(hidden_units, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_units)
        self.decoder_output = nn.Linear(hidden_units, 784)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=initial_weight_std)
                nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor, noise: torch.Tensor) -> tuple:
        """输入 B×784 和 L×B×d 噪声，返回 L×B×784 logits 及 B×d 后验参数。"""
        hidden = F.softplus(self.encoder_hidden(images))
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden)
        require_finite("posterior parameters", mean, log_variance)
        standard_deviation = torch.exp(0.5 * log_variance)
        require_finite("posterior standard deviation", standard_deviation)
        latent = mean.unsqueeze(0) + standard_deviation.unsqueeze(0) * noise
        logits = self.decoder_output(F.softplus(self.decoder_hidden(latent)))
        require_finite("decoder logits", logits)
        return logits, mean, log_variance


def loss_components(model: VAE, images: torch.Tensor, noise: torch.Tensor) -> dict[str, torch.Tensor]:
    """返回按图像平均的负 ELBO、重构 NLL 和 KL；像素/潜变量维度求和而非求平均。"""
    logits, mean, log_variance = model(images, noise)
    targets = images.unsqueeze(0).expand_as(logits)
    # 对 784 个像素求和，再平均 L 次重参数化样本和 B 幅图像。
    reconstruction = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    reconstruction = reconstruction.sum(dim=-1).mean()
    kl = 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance).sum(dim=-1).mean()
    terms = {"negative_elbo": reconstruction + kl, "reconstruction_nll": reconstruction, "kl": kl}
    require_finite("VAE objective", *terms.values())
    return terms
