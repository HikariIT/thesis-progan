from config.gan_config import GANConfig
from net.gan.generator import Generator
from net.gan.discriminator import Discriminator


class ProGAN:

    def __init__(self, gan_config: GANConfig):
        self.gan_config = gan_config

    def get_models(self) -> tuple[Generator, Discriminator]:
        return Generator(self.gan_config), Discriminator(self.gan_config)
