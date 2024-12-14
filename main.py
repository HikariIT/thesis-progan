import matplotlib.pyplot as plt
import numpy as np
import torch

from config.gan_config import GANConfig
from net.gan.discriminator import Discriminator
from net.gan.generator import Generator

# Create a generator object
gan_cfg = GANConfig(
    latent_dim=256,
    image_channels=1,
    depth=5,
    num_channels=16,
    max_channels=256
)

generator = Generator(gan_cfg)
discriminator = Discriminator(gan_cfg)

print(discriminator.get_layers())

latent_vector = generator.generate_latent_points(1)
latent_vector_as_tensor_batch = torch.tensor(latent_vector, dtype=torch.float32)
latent_vector_as_tensor_batch = latent_vector_as_tensor_batch.to(generator.device())
latent_vector_as_tensor_batch = latent_vector_as_tensor_batch.view(-1, generator.latent_dim, 1, 1)

NO_STEPS = 3

fig, ax = plt.subplots(1, NO_STEPS, figsize=(15, 5))

for i in range(NO_STEPS):
    generated_image = generator(latent_vector_as_tensor_batch, alpha=i/NO_STEPS)
    generated_image_numpy = generated_image.squeeze().detach().cpu().numpy()
    ax[i].imshow(generated_image_numpy, cmap='gray')
    ax[i].axis('off')

    predicted = discriminator(generated_image)
    print(f"Prediction: {predicted}")

    generator.grow()
    discriminator.grow()


plt.show()