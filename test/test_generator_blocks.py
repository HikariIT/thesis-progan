import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from ..net.modules.pro_gan.generator_blocks import GeneratorConvBlock, GeneratorInitialBlock, GeneratorFinalBlock

class TestGeneratorBlocks:

    def test_generator_conv_block(self):
        block = GeneratorConvBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        x = torch.randn(1, 3, 64, 64)
        y = block(x)
        assert y.shape == (1, 64, 64, 64)

    def test_generator_initial_block(self):
        block = GeneratorInitialBlock(latent_size=128, fmap_size=64)
        x = torch.randn(1, 128, 1, 1)
        y = block(x)
        assert y.shape == (1, 64, 4, 4)

    def test_generator_final_block(self):
        block = GeneratorFinalBlock(in_channels=64, out_channels=3)
        x = torch.randn(1, 64, 64, 64)
        y = block(x)
        assert y.shape == (1, 3, 64, 64)