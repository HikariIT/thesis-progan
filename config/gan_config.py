from dataclasses import dataclass

@dataclass
class GANConfig:
    depth: int = 5
    latent_dim: int = 256
    num_channels: int = 16
    max_channels: int = 256
    image_channels: int = 3
