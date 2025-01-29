import torch
import matplotlib.pyplot as plt

from net.gan.model import ProGAN
from config.gan_config import GANConfig


LEARNING_RATE = 1e-3
LAMBDA_GP = 10


def interpolate_points(p1, p2, n_steps=10):
    ratios = torch.linspace(0, 1, n_steps + 2)
    vectors = []
    for ratio in ratios[1:-1]:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return vectors

if __name__ =="__main__":
    gan_config = GANConfig(image_channels=1)
    generator, discriminator = ProGAN(gan_config).get_models()

    path = 'saved_models/1735035936_depth_5_iteration_359000.pth'
    model_data = torch.load(path)

    generator.load_state_dict(model_data['generator'])
    discriminator.load_state_dict(model_data['discriminator'])

    generator.current_depth = model_data['depth']
    discriminator.current_depth = model_data['depth']

    print(generator.current_depth)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Create latent vector

    latent_vector = generator.generate_latent_points(4)
    latent_vector = latent_vector.to(device)

    # Generate image
    with torch.no_grad():
        images = generator(latent_vector).detach().cpu()
        images = images.clamp(-1, 1)
        images_normalized = (images + 1) / 2

    # Display image in 5x2 grid
    fig, ax = plt.subplots(1, 4, figsize=(8, 3))

    for i in range(4):
        img = images_normalized[i].squeeze().numpy()
        print(img.shape)
        print(img.min(), img.max())
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')

    fig.suptitle('Zdjęcia wygenerowane przez generator GAN po douczeniu sieci')
    fig.tight_layout()

    plt.show()

    plt.imshow(images_normalized[0].squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

    # Create latent vector
    latent_vector = generator.generate_latent_points(2)
    latent_vector_1 = latent_vector[0]
    latent_vector_2 = latent_vector[1]

    latent_vector_1 = latent_vector_1.to(device)
    latent_vector_2 = latent_vector_2.to(device)

    # Interpolate between images
    NO_IMAGES = 6

    interpolated_z = interpolate_points(latent_vector_1, latent_vector_2, NO_IMAGES - 2)

    # Decode interpolated images
    images = []
    for i in range(NO_IMAGES - 2):
        interpolated_image = generator(interpolated_z[i])
        images.append(interpolated_image)

    # Display images
    fig, ax = plt.subplots(1, NO_IMAGES, figsize=(2 * NO_IMAGES, 2))

    ax[0].imshow(generator(latent_vector_1).squeeze().detach().cpu().numpy(), cmap='gray')
    ax[0].axis('off')

    for i in range(1, NO_IMAGES - 1):
        ax[i].imshow(images[i - 1].squeeze().detach().cpu().numpy(), cmap='gray')
        ax[i].axis('off')

    ax[NO_IMAGES - 1].imshow(generator(latent_vector_2).squeeze().detach().cpu().numpy(), cmap='gray')
    ax[NO_IMAGES - 1].axis('off')

    plt.show()
