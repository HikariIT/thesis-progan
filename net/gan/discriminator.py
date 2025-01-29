import torch.nn as nn
import torch

from config.gan_config import GANConfig
from net.modules.pro_gan.discriminator_blocks import DiscriminatorInitialBlock, DiscriminatorMiddleBlock, DiscriminatorFinalBlock


class Discriminator(nn.Module):

    def __init__(self, gan_config: GANConfig):
        super().__init__()

        self.latent_dim = gan_config.latent_dim
        self.input_dim = gan_config.image_channels
        self.num_channels = gan_config.num_channels
        self.max_channels = gan_config.max_channels

        self.depth = gan_config.depth
        self.current_depth = 0

        self.init_layers()

    def init_layers(self):
        self.initial_layers = nn.ModuleList()

        for i in reversed(range(self.depth + 1)):
            input_channels = self.input_dim
            output_channels = min(self.max_channels, self.num_channels * 2 ** i)
            self.initial_layers.append(DiscriminatorInitialBlock(input_channels, output_channels))

        self.middle_layers = nn.ModuleList()

        for i in reversed(range(self.depth)):
            input_channels = min(self.max_channels, self.num_channels * 2 ** i)
            output_channels = min(self.max_channels, self.num_channels * 2 ** (i + 1))
            self.middle_layers.append(DiscriminatorMiddleBlock(input_channels, output_channels))

        output_channels = min(self.max_channels, self.num_channels * 2**self.depth)
        self.final_layer = DiscriminatorFinalBlock(output_channels, output_channels)

    def grow(self, factor: int = 1):
        if self.current_depth + factor > self.depth:
            raise ValueError("Requested depth is greater than the maximum depth allowed")

        self.current_depth += factor

    def get_layers(self):
        return list(self.initial_layers) + list(self.middle_layers) + [self.final_layer]

    def forward(self, x: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        if self.current_depth != 0:
            old_res = nn.functional.avg_pool2d(x, 2)
            old_res = self.initial_layers[self.current_depth - 1](old_res)

            new_res = self.initial_layers[self.current_depth](x)
            new_res = self.middle_layers[self.current_depth - 1](new_res)

            x = alpha * new_res + (1 - alpha) * old_res

            for i in reversed(range(self.current_depth - 1)):
                x = self.middle_layers[i](x)
        else:
            x = self.initial_layers[self.current_depth](x)

        x = self.final_layer(x)
        return x

    def device(self):
        return next(self.parameters()).device