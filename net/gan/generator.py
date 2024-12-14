import torch.nn as nn
import torch

from config.gan_config import GANConfig
from net.modules.pro_gan.generator_blocks import GeneratorInitialBlock, GeneratorMiddleBlock, GeneratorFinalBlock


class Generator(nn.Module):

    def __init__(self, gan_config: GANConfig):
        super().__init__()

        self.latent_dim = gan_config.latent_dim
        self.output_dim = gan_config.image_channels
        self.num_channels = gan_config.num_channels
        self.max_channels = gan_config.max_channels

        self.depth = gan_config.depth
        self.current_depth = 0

        self.init_layers()

    def init_layers(self):
        self.initial_layer = GeneratorInitialBlock(self.latent_dim, min(self.max_channels, self.num_channels * 2 ** self.depth))
        self.middle_blocks = nn.ModuleList()

        for i in reversed(range(self.depth)):
            input_channels = min(self.max_channels, self.num_channels * 2 ** (i + 1))
            output_channels = min(self.max_channels, self.num_channels * 2 ** i)
            self.middle_blocks.append(GeneratorMiddleBlock(input_channels, output_channels))

        self.final_blocks = nn.ModuleList()

        for i in reversed(range(self.depth + 1)):
            input_channels = min(self.max_channels, self.num_channels * 2 ** i)
            output_channels = self.output_dim
            self.final_blocks.append(GeneratorFinalBlock(input_channels, output_channels))

    def grow(self, factor: int = 1):
        if self.current_depth + factor > self.depth:
            raise ValueError("Requested depth is greater than the maximum depth allowed")

        self.current_depth += factor

    def get_layers(self):
        return [self.initial_layer] + list(self.middle_blocks) + list(self.final_blocks)

    def forward(self, x: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        x = self.initial_layer(x)

        if self.current_depth != 0:
            for i in range(self.current_depth - 1):
                x = self.middle_blocks[i](x)

            residual = self.final_blocks[self.current_depth - 1](x)
            x = self.middle_blocks[self.current_depth - 1](x)

            new_res = self.final_blocks[self.current_depth](x)
            old_res = nn.functional.interpolate(residual, scale_factor=2)
            return alpha * new_res + (1 - alpha) * old_res

        else:
            return self.final_blocks[0](x)

    def device(self):
        return next(self.parameters()).device

    def generate_latent_points(self, num_points: int) -> torch.Tensor:
        return torch.randn(num_points, self.latent_dim, device=self.device())