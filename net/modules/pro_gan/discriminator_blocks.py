import torch.nn as nn
import torch

from net.modules.pro_gan.weighted_modules import WeightedConv2d, WeightedLinear
from net.modules.pro_gan.minibatch_stddev import MiniBatchStdDev


class DiscriminatorBlock(nn.Module):
    # Abstract class for the discriminator block

    block: nn.Module

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DiscriminatorConvBlock(DiscriminatorBlock):
    # Convolutional building block for the discriminator

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.block.add_module('weighted_conv_2d', WeightedConv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.block.add_module('leaky_relu', nn.LeakyReLU(0.2))


class DiscriminatorInitialBlock(DiscriminatorBlock):
    # Initial block of the discriminator

    def __init__(self, input_dim: int, out_channels: int):
        super().__init__()

        self.block.add_module('conv_block', DiscriminatorConvBlock(input_dim, out_channels, 1, 1, 0))


class DiscriminatorMiddleBlock(DiscriminatorBlock):
    # Middle block of the discriminator

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block.add_module('conv_block_1', DiscriminatorConvBlock(in_channels, out_channels, 3, 1, 1))
        self.block.add_module('conv_block_2', DiscriminatorConvBlock(out_channels, out_channels, 3, 1, 1))
        self.block.add_module('avg_pool', nn.AvgPool2d(2))


class DiscriminatorFinalBlock(nn.Module):
    # Final block of the discriminator

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            WeightedConv2d(in_channels + 1, out_channels, 3, 1, 1),
            WeightedConv2d(out_channels, out_channels, 4, 1, 0),
        )
        self.minibatch_stddev = MiniBatchStdDev()
        self.linear = WeightedLinear(out_channels, 1)
        self.flatten = nn.Flatten()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.minibatch_stddev(x)
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x