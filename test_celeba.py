import torch
import torchvision

from config.gan_config import GANConfig
from config.gan_loss_config import GANLossConfig
from config.gan_training_config import GANTrainingConfig
from torchvision import transforms
from torch.optim.adam import Adam
from net.gan.model import ProGAN
from training.gan_training import GANTraining
from training.losses.wgan_gp_loss import DiscriminatorLoss, GeneratorLoss, WGANGPLoss

DATASET_PATH = './dataset/terrain/scaled'
LEARNING_RATE = 1e-3
LAMBDA_GP = 10

if __name__ =="__main__":
    transform_celeba = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])

    train_dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=transform_celeba)

    print(train_dataset[0][0].shape)

    gan_config = GANConfig(image_channels=1)
    generator, discriminator = ProGAN(gan_config).get_models()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    discriminator_optimizer = Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0, 0.99))
    generator_optimizer = Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0, 0.99))

    loss_config = GANLossConfig(LAMBDA_GP)
    wgan_gp_loss = WGANGPLoss(discriminator_optimizer, generator_optimizer, loss_config)
    discriminator_loss = DiscriminatorLoss(wgan_gp_loss)
    generator_loss = GeneratorLoss(wgan_gp_loss)

    training_config = GANTrainingConfig(save_interval=1000, img_generation_interval=500, checkpoint_images=16, num_workers=4)
    trainer = GANTraining(generator, discriminator, generator_loss, discriminator_loss, train_dataset, training_config)
    trainer.train()