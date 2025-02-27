import torchvision.transforms as transforms
import torchvision
import torch

from config.vae_training_config import VAETrainingConfig
from training.vae_training import VAETraining
from training.losses.vae_loss import VAELoss
from config.vae_config import VAEConfig
from torch.utils.data import Dataset
from torch.optim.adam import Adam
from net.vae.model import VAE

BATCH_SIZE = 16
LATENT_DIM = 512
IMAGE_CHANNELS = 1
LEARNING_RATE = 5e-4
DATASET_PATH = './dataset/terrain/scaled'


def train(train_dataset: Dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae_config = VAEConfig(latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS)
    vae = VAE(vae_config).to(device)

    vae_optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
    vae_loss = VAELoss(vae_optimizer)

    training_config = VAETrainingConfig(epochs=2048, batch_size=BATCH_SIZE, log_interval=1, image_interval=10, save_interval=50, num_workers=4)
    trainer = VAETraining(vae, vae_loss, training_config)
    trainer.train(train_dataset)


if __name__ == "__main__":
    transform_celeba = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])

    train_dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=transform_celeba)
    train_dataset.samples = train_dataset.samples[:1000]
    train_dataset.imgs = train_dataset.imgs[:1000]

    train(train_dataset)

# File for VAE training