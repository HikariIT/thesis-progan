import torch.nn as nn
import torch

from config.vae_config import VAEConfig
from net.modules.vae.encoder_blocks import VAEEncoderBlock


class VAE(nn.Module):

    def __init__(self, vae_config: VAEConfig):
        super().__init__()

        self.latent_dim = vae_config.latent_dim
        self.image_channels = vae_config.image_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.mean_layer = nn.ModuleList()
        self.logvar_layer = nn.ModuleList()

        self.init_layers()

    def init_layers(self):
        self.encoder.append(VAEEncoderBlock(self.image_channels, 32, 4, 2, 1))
        self.encoder.append(VAEEncoderBlock(32, 64, 4, 2, 1))
        self.encoder.append(VAEEncoderBlock(64, 128, 4, 2, 1))
        self.encoder.append(VAEEncoderBlock(128, 256, 4, 2, 1))
        self.encoder.append(VAEEncoderBlock(256, 512, 4, 2, 1))
        self.encoder.append(nn.Flatten())

        self.mean_layer = nn.Linear(512 * 4 * 4, self.latent_dim)
        self.logvar_layer = nn.Linear(512 * 4 * 4, self.latent_dim)

        self.decoder.append(nn.Linear(self.latent_dim, 512 * 4 * 4))
        self.decoder.append(nn.Unflatten(1, (512, 4, 4)))

        self.decoder.append(nn.ConvTranspose2d(512, 256, 4, 2, 1))
        self.decoder.append(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.decoder.append(nn.ConvTranspose2d(128, 64, 4, 2, 1))
        self.decoder.append(nn.ConvTranspose2d(64, 32, 4, 2, 1))
        self.decoder.append(nn.ConvTranspose2d(32, self.image_channels, 4, 2, 1))

    def encode(self, x: torch.Tensor):
        for layer in self.encoder:
            x = layer(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor):
        eps = torch.randn_like(logvar).to(logvar.device)
        return mean + logvar * eps

    def decode(self, z: torch.Tensor):
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward(self, x: torch.Tensor):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
