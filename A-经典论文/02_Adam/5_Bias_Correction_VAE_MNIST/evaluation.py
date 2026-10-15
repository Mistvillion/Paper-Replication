"""固定 Monte Carlo 噪声评估负 ELBO，避免把采样变化误判为优化器差异。"""

from math import isfinite

import torch

from model import NumericalFailure, VAE, loss_components, sample_noise


@torch.no_grad()
def evaluate(
    model: VAE, images: torch.Tensor, *, batch_size: int, mc_samples: int,
    seed: int, device: torch.device,
) -> dict[str, float]:
    """按样本加权评估重构项与 KL；每次重置专用随机流，不改变训练采样序列。"""
    if len(images) == 0 or batch_size <= 0 or mc_samples <= 0:
        raise ValueError("评估需要非空数据、正批量大小和正 Monte Carlo 样本数")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    sums = {"negative_elbo": 0.0, "reconstruction_nll": 0.0, "kl": 0.0}
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device)
            noise = sample_noise(mc_samples, len(batch), model.latent_dim, generator, device)
            terms = loss_components(model, batch, noise)
            for name, value in terms.items():
                sums[name] += len(batch) * value.item()
        metrics = {name: value / len(images) for name, value in sums.items()}
        if not all(isfinite(value) for value in metrics.values()):
            raise NumericalFailure("evaluation aggregate contains NaN or Inf")
        return metrics
    finally:
        model.train(was_training)
