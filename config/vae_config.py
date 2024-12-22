from dataclasses import dataclass

@dataclass
class VAEConfig:
    image_channels: int = 3
    latent_dim: int = 256