import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.optim.adam import Adam
from torch.utils.data import Dataset
from config.vae_training_config import VAETrainingConfig
from training.vae_training import VAETraining
from training.losses.vae_loss import VAELoss
from config.vae_config import VAEConfig
from net.vae.model import VAE

BATCH_SIZE = 16
LATENT_DIM = 100
IMAGE_CHANNELS = 1
LEARNING_RATE = 1e-4
DATASET_PATH = './dataset/terrain/scaled'

def check(train_dataset: Dataset):
    vae_config = VAEConfig(latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS)
    vae = VAE(vae_config)
    vae.load_state_dict(torch.load('saved_models/1736007219_1900/model.pth'))
    vae.eval()

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    batch = next(iter(data_loader))

    x = batch[0] / 2 + 0.5
    print(x.shape)
    print(x.min(), x.max())
    x_hat, mean, log_var = vae(x)
    print(x_hat.shape)
    print(x_hat.min(), x_hat.max())

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x[0].squeeze().detach().numpy(), cmap='gray')
    ax[1].imshow(x_hat[0].squeeze().detach().numpy(), cmap='gray')
    plt.show()

if __name__ == "__main__":
    transform_celeba = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])

    train_dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=transform_celeba)
    check(train_dataset)