import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from sklearn.decomposition import PCA
from torch.optim.adam import Adam
from torch.utils.data import Dataset
from config.vae_training_config import VAETrainingConfig
from training.vae_training import VAETraining
from training.losses.vae_loss import VAELoss
from config.vae_config import VAEConfig
from net.vae.model import VAE
from torchvision.utils import make_grid

LATENT_DIM = 512
IMAGE_CHANNELS = 1
DATASET_PATH = './dataset/terrain/scaled/test'

def interpolate_points(p1, p2, n_steps=10):
    ratios = torch.linspace(0, 1, n_steps)
    vectors = ratios.view(-1, 1, 1) * (p2 - p1)
    return vectors + p1

def check(train_dataset: Dataset):
    vae_config = VAEConfig(latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS)
    vae = VAE(vae_config)
    vae.load_state_dict(torch.load('saved_models/1736374594_2000/model.pth'))
    vae.eval()

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(data_loader))[0]

    img_1 = batch[0].unsqueeze(0)
    img_2 = batch[1].unsqueeze(0)

    # Encode images
    mu_1, log_var_1 = vae.encode(img_1)
    mu_2, log_var_2 = vae.encode(img_2)

    # Sample latent vectors
    z_1 = vae.reparameterize(mu_1, log_var_1)
    z_2 = vae.reparameterize(mu_2, log_var_2)

    # Interpolate between images
    NO_IMAGES = 6
    interpolated_z = interpolate_points(z_1, z_2, NO_IMAGES - 2)

    # Decode interpolated images
    images = []
    for i in range(NO_IMAGES - 2):
        interpolated_image = vae.decode(interpolated_z[i])
        images.append(interpolated_image)

    # Display images
    fig, ax = plt.subplots(1, NO_IMAGES, figsize=(2 * NO_IMAGES, 2))

    ax[0].imshow(img_1.squeeze().detach().numpy(), cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Obraz 1')

    for i in range(1, NO_IMAGES - 1):
        ax[i].imshow(images[i - 1].squeeze().detach().numpy(), cmap='gray')
        ax[i].axis('off')

    ax[NO_IMAGES - 1].imshow(img_2.squeeze().detach().numpy(), cmap='gray')
    ax[NO_IMAGES - 1].axis('off')
    ax[NO_IMAGES - 1].set_title('Obraz 2')


    fig.suptitle('Interpolacja między dwoma obrazami przy użyciu autoenkodera VAE')

    plt.show()

    plt.imshow(vae.decode(z_1).squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()


def latent_space_visualization(train_dataset: Dataset):
    vae_config = VAEConfig(latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS)
    vae = VAE(vae_config)
    vae.load_state_dict(torch.load('saved_models/1736374594_2000/model.pth'))
    vae.eval()

    # Get all images from ImageFolder with their labels
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    batch = next(iter(data_loader))

    data = batch[0]
    labels = batch[1]

    # Encode images
    mu, log_var = vae.encode(data)
    z = vae.reparameterize(mu, log_var)

    print(z.shape)

    pca = PCA(n_components=3)
    pca.fit(z.detach().numpy())

    z_pca = pca.transform(z.detach().numpy())
    print(z_pca.shape)

    # Create 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    classes = ['desert', 'high_lands', 'lake_lands', 'low_lands', 'mountains']
    colors = ['tab:brown', 'tab:orange', 'tab:blue', 'tab:green', 'tab:gray']

    for i in range(len(classes)):
        indices = labels == i
        ax.scatter(z_pca[indices, 0], z_pca[indices, 1], z_pca[indices, 2], c=colors[i], label=classes[i])

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('Przestrzeń ukryta autoenkodera VAE zredukowana do 3 wymiarów przy użyciu PCA')

    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=classes[i], markerfacecolor=c, markersize=10) for i, c in enumerate(colors)], title='Rodzaj terenu')

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
    latent_space_visualization(train_dataset)