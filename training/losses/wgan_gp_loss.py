import torch
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
from config.gan_loss_config import GANLossConfig
from net.gan.discriminator import Discriminator


class WGANGPLoss:
    # Wasserstein GAN with Gradient Penalty Loss

    def __init__(self, discriminator_optimizer: Optimizer, generator_optimizer: Optimizer, gan_loss_config: GANLossConfig):
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.lambda_gp = gan_loss_config.lambda_gp

    def gradient_penalty(self, discriminator: Discriminator, real_batch: torch.Tensor, fake_batch: torch.Tensor, alpha: float):
        batch_size = real_batch.shape[0]
        beta = torch.rand((batch_size, 1, 1, 1)).to(real_batch.device)
        interpolated_img = real_batch * beta + fake_batch * (1 - beta)
        interpolated_img = Variable(interpolated_img, requires_grad=True)

        probability_interpolated = discriminator(interpolated_img, alpha)

        gradient = torch.autograd.grad(outputs=probability_interpolated, inputs=interpolated_img,
                                        grad_outputs=torch.ones_like(probability_interpolated).to(real_batch.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient = gradient.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)
        penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return penalty


class DiscriminatorLoss:

    def __init__(self, wgan_gp_loss: WGANGPLoss):
        self.wgan_gp_loss = wgan_gp_loss

    def __call__(self, discriminator: Discriminator, real_img: torch.Tensor, fake_img: torch.Tensor, alpha: float):
        real_probability = discriminator(real_img, alpha).view(-1)
        fake_probability = discriminator(fake_img, alpha).view(-1)

        gp_loss = self.wgan_gp_loss.gradient_penalty(discriminator, real_img, fake_img, alpha)
        discriminator_loss = torch.mean(fake_probability) - torch.mean(real_probability) + gp_loss

        self.wgan_gp_loss.discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        self.wgan_gp_loss.discriminator_optimizer.step()

        return discriminator_loss


class GeneratorLoss:

    def __init__(self, wgan_gp_loss: WGANGPLoss):
        self.wgan_gp_loss = wgan_gp_loss

    def __call__(self, discriminator: Discriminator, fake_img: torch.Tensor, alpha: float):
        fake_critic = discriminator(fake_img, alpha).view(-1)
        generator_loss = -torch.mean(fake_critic)

        self.wgan_gp_loss.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.wgan_gp_loss.generator_optimizer.step()

        return generator_loss
