import torch.nn as nn
import torch.nn.functional as F


class VAEDecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size, 1, kernel_size // 2)
        self.bn_1 = nn.BatchNorm2d(in_channels // 2)

        self.conv_2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.LeakyReLU(0.2)


    def forward(self, x):
        x = self.upsample(x)
        residual = self.residual_conv(x)
        x = self.activation(self.bn_1(self.conv(x)))
        x = self.conv_2(x)
        return self.activation(self.bn_2(x + residual))

