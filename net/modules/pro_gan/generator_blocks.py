import torch.nn as nn
import torch

from net.modules.pro_gan.pixelwise_norm import PixelwiseNormalization
from net.modules.pro_gan.upsample import Upsample
from net.modules.pro_gan.weighted_modules import WeightedConv2d


class GeneratorBlock(nn.Module):
    # Abstract class for the generator block

    block: nn.Module

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeneratorConvBlock(GeneratorBlock):
    # Convolutional building block for the generator

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.block.add_module('weighted_conv_2d', WeightedConv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.block.add_module('leaky_relu', nn.LeakyReLU(0.2))
        self.block.add_module('pixelwise_norm', PixelwiseNormalization())


class GeneratorInitialBlock(GeneratorBlock):
    # Accept latent space vector as input and outputs feature map

    def __init__(self, latent_size: int, fmap_size: int):
        super().__init__()

        self.block.add_module('conv_transpose', nn.ConvTranspose2d(latent_size, fmap_size, kernel_size=4, stride=1, padding=0))
        self.block.add_module('leaky_relu', nn.LeakyReLU(0.2))
        self.block.add_module('pixelwise_norm', PixelwiseNormalization())
        self.block.add_module('conv_block', GeneratorConvBlock(fmap_size, fmap_size, 3, 1, 1))


class GeneratorMiddleBlock(GeneratorBlock):
    # Middle block of the generator

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block.add_module('upsample', Upsample())
        self.block.add_module('conv_block_1', GeneratorConvBlock(in_channels, out_channels, 3, 1, 1))
        self.block.add_module('conv_block_2', GeneratorConvBlock(out_channels, out_channels, 3, 1, 1))


class GeneratorFinalBlock(GeneratorBlock):
    # Final block of the generator

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block.add_module('weighted_conv_2d', WeightedConv2d(in_channels, out_channels, 1, 1, 0))