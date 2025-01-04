import torch
import torchsummary

from config.gan_config import GANConfig
from net.gan.model import ProGAN

DATASET_PATH = './dataset/terrain/scaled'
LEARNING_RATE = 1e-3
LAMBDA_GP = 10

if __name__ =="__main__":
    gan_config = GANConfig(image_channels=1)
    generator, discriminator = ProGAN(gan_config).get_models()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # generator(torch.randn((256, 1, 1)).to(device))
    discriminator(torch.randn((1, 1, 4, 4)).to(device))

    for i in range(1, 6):
        discriminator.grow()

        # generator(torch.randn((256, 1, 1)).to(device))
        discriminator(torch.randn((1, 1, 4 * 2 ** i, 4 * 2 ** i)).to(device))