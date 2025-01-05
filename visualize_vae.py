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
from torchvision.utils import make_grid

LATENT_DIM = 100
IMAGE_CHANNELS = 1
DATASET_PATH = './dataset/terrain/scaled'

def interpolate_points(p1, p2, n_steps=10):
    ratios = torch.linspace(0, 1, n_steps)
    vectors = ratios.view(-1, 1, 1) * (p2 - p1)
    return vectors + p1

def check(train_dataset: Dataset):
    vae_config = VAEConfig(latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS)
    vae = VAE(vae_config)
    vae.load_state_dict(torch.load('saved_models/1736077620_1000/model.pth'))
    vae.eval()

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(data_loader))

    # Get two images
    x = batch[0] / 2 + 0.5

    img_1 = x[0].unsqueeze(0)
    img_2 = x[1].unsqueeze(0)

    # Encode images
    mu_1, log_var_1 = vae.encode(img_1)
    mu_2, log_var_2 = vae.encode(img_2)

    # Sample latent vectors
    z_1 = vae.reparameterize(mu_1, log_var_1)
    z_2 = vae.reparameterize(mu_2, log_var_2)

    # Interpolate between images
    interpolated_z = interpolate_points(z_1, z_2, 10)

    # Decode interpolated images

    images = []
    for i in range(10):
        interpolated_image = vae.decode(interpolated_z[i])
        images.append(interpolated_image)

    # Display images
    fig, ax = plt.subplots(1, 12, figsize=(24, 2))

    ax[0].imshow(img_1.squeeze().detach().numpy(), cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Obraz 1')

    for i in range(1, 11):
        ax[i].imshow(images[i - 1].squeeze().detach().numpy(), cmap='gray')
        ax[i].axis('off')

    ax[11].imshow(img_2.squeeze().detach().numpy(), cmap='gray')
    ax[11].axis('off')
    ax[11].set_title('Obraz 2')


    fig.suptitle('Interpolacja między dwoma obrazami przy użyciu autoenkodera VAE')

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