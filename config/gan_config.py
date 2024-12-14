from dataclasses import dataclass

@dataclass
class GANConfig:
    depth: int
    latent_dim: int
    num_channels: int
    max_channels: int
    image_channels: int
