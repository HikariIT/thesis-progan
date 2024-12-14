import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from net.modules.pro_gan.discriminator_blocks import DiscriminatorConvBlock, DiscriminatorInitialBlock, DiscriminatorFinalBlock

class TestDiscriminatorBlocks:
    def test_discriminator_conv_block(self):
        block = DiscriminatorConvBlock(3, 64, 3, 1, 1)
        x = torch.randn(1, 3, 64, 64)
        y = block(x)
        assert y.shape == (1, 64, 64, 64)

    def test_discriminator_initial_block(self):
        block = DiscriminatorInitialBlock(64, 3)
        x = torch.randn(1, 3, 64, 64)
        y = block(x)
        assert y.shape == (1, 64, 64, 64)

    def test_discriminator_final_block(self):
        block = DiscriminatorFinalBlock(512, 256)
        x = torch.randn(32, 512, 4, 4)
        y = block(x)
        assert y.shape == (32, 1)