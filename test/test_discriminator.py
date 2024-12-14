import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from net.gan.discriminator import Discriminator
from config.gan_config import GANConfig


class TestDiscriminator(unittest.TestCase):

    def setUp(self):
        gan_config = GANConfig(
            latent_dim=128,
            image_channels=3,
            num_channels=64,
            max_channels=512,
            depth=4
        )
        self.discriminator = Discriminator(gan_config)

    def test_initialization(self):
        self.assertEqual(self.discriminator.latent_dim, 128)
        self.assertEqual(self.discriminator.input_dim, 3)
        self.assertEqual(self.discriminator.num_channels, 64)
        self.assertEqual(self.discriminator.max_channels, 512)
        self.assertEqual(self.discriminator.depth, 4)
        self.assertEqual(self.discriminator.current_depth, 0)

    def test_grow(self):
        self.discriminator.grow(1)
        self.assertEqual(self.discriminator.current_depth, 1)
        self.discriminator.grow(2)
        self.assertEqual(self.discriminator.current_depth, 3)
        with self.assertRaises(ValueError):
            self.discriminator.grow(2)

    def test_input_image_size_grows_with_depth(self):
        x = torch.randn(1, 3, 4, 4)
        output = self.discriminator(x)
        self.assertEqual(output.shape, (1, 1))

        self.discriminator.grow(1)
        x = torch.randn(1, 3, 8, 8)
        output = self.discriminator(x)
        self.assertEqual(output.shape, (1, 1))

        self.discriminator.grow(2)
        x = torch.randn(1, 3, 32, 32)
        output = self.discriminator(x)
        self.assertEqual(output.shape, (1, 1))

    def test_get_layers(self):
        layers = self.discriminator.get_layers()
        self.assertEqual(len(layers), self.discriminator.depth * 2 + 2)

    def test_forward(self):
        x = torch.randn(1, 3, 4, 4)
        output = self.discriminator(x)
        self.assertEqual(output.shape, (1, 1))

        self.discriminator.grow(1)
        x = torch.randn(1, 3, 8, 8)
        output = self.discriminator(x)
        self.assertEqual(output.shape, (1, 1))

    def test_device(self):
        device = self.discriminator.device()
        self.assertEqual(device, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()