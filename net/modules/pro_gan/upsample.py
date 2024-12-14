import torch.nn as nn
import torch


class Upsample(nn.Module):
    # Upsample the input tensor by a factor of 2 in Generator

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=2, mode='nearest')