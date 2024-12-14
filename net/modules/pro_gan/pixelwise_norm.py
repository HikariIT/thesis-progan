import torch.nn as nn
import torch


class PixelwiseNormalization(nn.Module):
    # Pixelwise Normalization, described in https://arxiv.org/pdf/1710.10196v3, p. 5

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
