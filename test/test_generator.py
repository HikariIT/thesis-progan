import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from net.gan.generator import Generator
from config.gan_config import GANConfig


class TestGenerator(unittest.TestCase):

    def setUp(self):
        gan_config = GANConfig(
            latent_dim=128,
            image_channels=3,
            num_channels=64,
            max_channels=512,
            depth=4
        )
        self.generator = Generator(gan_config)

    def test_initialization(self):
        self.assertEqual(self.generator.latent_dim, 128)
        self.assertEqual(self.generator.output_dim, 3)
        self.assertEqual(self.generator.num_channels, 64)
        self.assertEqual(self.generator.max_channels, 512)
        self.assertEqual(self.generator.depth, 4)
        self.assertEqual(self.generator.current_depth, 0)

    def test_grow(self):
        self.generator.grow(1)
        self.assertEqual(self.generator.current_depth, 1)
        self.generator.grow(2)
        self.assertEqual(self.generator.current_depth, 3)
        with self.assertRaises(ValueError):
            self.generator.grow(2)

    def test_output_image_size_grows_with_depth(self):
        latent_points = self.generator.generate_latent_points(1).view(-1, self.generator.latent_dim, 1, 1)
        output = self.generator(latent_points)
        self.assertEqual(output.shape[2], 4)
        self.generator.grow(1)
        output = self.generator(latent_points)
        self.assertEqual(output.shape[2], 8)
        self.generator.grow(2)
        output = self.generator(latent_points)
        self.assertEqual(output.shape[2], 32)

    def test_generate_latent_points(self):
        latent_points = self.generator.generate_latent_points(10)
        self.assertEqual(latent_points.shape, (10, 128))

    def test_forward(self):
        latent_points = self.generator.generate_latent_points(1).view(-1, self.generator.latent_dim, 1, 1)
        output = self.generator(latent_points)
        self.assertEqual(output.shape[1], 3)

        self.generator.grow(1)
        output = self.generator(latent_points)
        self.assertEqual(output.shape[1], 3)

        self.generator.grow(3)
        output = self.generator(latent_points)
        self.assertEqual(output.shape[1], 3)

    def test_get_layers(self):
        layers = self.generator.get_layers()
        self.assertEqual(len(layers), self.generator.depth * 2 + 2)


if __name__ == '__main__':
    unittest.main()