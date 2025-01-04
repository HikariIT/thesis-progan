import torch.nn as nn
import torch.nn.functional as F


class VAEEncoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.conv(x)), 0.2)
        return x