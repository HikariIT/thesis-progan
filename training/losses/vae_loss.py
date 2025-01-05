import torch.nn.functional as F
import torch

from torch.optim.optimizer import Optimizer

from net.vae.model import VAE


class VAELoss:

    def __init__(self, vae_optimizer: Optimizer):
        self.optimizer = vae_optimizer

    def __call__(self, vae: VAE, x: torch.Tensor):
        x_hat, mean, log_var = vae(x)
        reproduction_loss = F.mse_loss(x_hat, x)
        kl_divergence = torch.mean(- 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))

        # print(f"Reconstruction Loss: {reproduction_loss.item()}, KL Divergence: {kl_divergence.item()}")

        loss = reproduction_loss + kl_divergence

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, reproduction_loss, kl_divergence