import torch.nn as nn
import torch.nn.functional as F


class VAEDecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.activation(self.bn(self.conv(x)))
        return x

