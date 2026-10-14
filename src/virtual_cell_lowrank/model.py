import torch
from torch import nn
from torch.nn import functional as F


class LowRank(nn.Module):
    """Historical architecture; a target-specific gate scales all gene effects."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.coords = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, rank))
        self.gate = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, embeddings, basis):
        return (self.coords(embeddings) @ basis) * torch.sigmoid(self.gate(embeddings))


def effect_loss(pred, truth, gate):
    # This threshold is a magnitude heuristic, not a statistical DE test.
    de = truth.abs() >= 0.05
    return (
        F.smooth_l1_loss(pred, truth, beta=0.05)
        + 0.25 * (F.softplus(-truth.sign() * pred / 0.05) * de).sum() / de.sum().clamp_min(1)
        + 0.5 * (pred.abs() * (~de)).sum() / (~de).sum().clamp_min(1)
        + 0.01 * gate.mean()
    )


def nmae_proxy(pred, truth):
    if truth.abs().mean() <= 1e-8:
        raise ValueError("NMAE proxy is undefined for an all-zero validation effect matrix")
    return float((pred - truth).abs().mean() / truth.abs().mean())
