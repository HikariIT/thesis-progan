import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

from training.losses.wgan_gp_loss import DiscriminatorLoss, GeneratorLoss
from config.gan_training_config import GANTrainingConfig
from net.gan.discriminator import Discriminator
from net.gan.generator import Generator


class GANHistory:

    def __init__(self):
        self.generator_loss = []
        self.discriminator_loss = []
        self.discriminator_acc_real = []
        self.discriminator_acc_fake = []


class GANTraining:

    def __init__(self, generator: Generator, discriminator: Discriminator, generator_loss: GeneratorLoss, discriminator_loss: DiscriminatorLoss,
                 dataset: Dataset, config: GANTrainingConfig):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.dataset = dataset
        self.config = config

        self.id = str(round(time.time()))
        self.iterator = None
        self.iteration = 0
        self.current_depth_iterations = 0
        self.progress = None

        self.history = GANHistory()
        self.writer = SummaryWriter(log_dir=f"{self.config.log_dir}/{self.id}")
        self.depth_breakpoints = np.cumsum(self.config.steps_for_depth)

        if self.config.path_to_saved_model is not None:
            self.load_model(self.config.path_to_saved_model)

        self.current_batch_size = self.config.batch_sizes_for_depth[self.generator.current_depth]
        self.latent_space = self.generator.generate_latent_points(self.config.checkpoint_images)

        if self.config.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.update_ema(self.generator, self.generator_ema, 0)

    def train(self):
        self.init_progress_bar()
        self.data_loader = DataLoader(self.dataset, batch_size=self.current_batch_size, num_workers=self.config.num_workers, pin_memory=self.config.pin_memory)
        self.iterator = iter(self.data_loader)

        while self.iteration < self.depth_breakpoints[-1]:
            alpha = self.get_alpha()

            try:
                real_batch, _ = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.data_loader)
                real_batch, _ = next(self.iterator)

            real_batch = real_batch.to(self.generator.device())
            real_batch = nn.functional.interpolate(real_batch, size=(4 * 2 ** self.generator.current_depth, 4 * 2 ** self.generator.current_depth)) # NOTE: This is a temporary fix

            latent_points = self.generator.generate_latent_points(real_batch.shape[0])
            fake_batch = self.generator(latent_points, alpha)

            real_accuracy, fake_accuracy = self.compute_discriminator_accuracy(real_batch, fake_batch, alpha)

            discriminator_loss = self.discriminator_loss(self.discriminator, real_batch, fake_batch, alpha)
            generator_loss = self.generator_loss(self.discriminator, fake_batch, alpha)

            if self.config.use_ema:
                self.update_ema(self.generator, self.generator_ema, self.config.ema_decay)

            self.save_values(generator_loss, discriminator_loss, real_accuracy, fake_accuracy)
            self.step()

    def save_values(self, generator_loss: torch.Tensor, discriminator_loss: torch.Tensor, real_accuracy: float, fake_accuracy: float):
        if self.iteration % self.config.log_interval == 0:
            self.save_losses(generator_loss, discriminator_loss)
            self.save_accuracies(real_accuracy, fake_accuracy)
        if self.iteration % self.config.img_generation_interval == 0:
            self.save_images()
        if self.iteration % self.config.save_interval == 0:
            self.save_model()

    def save_losses(self, generator_loss: torch.Tensor, discriminator_loss: torch.Tensor):
        self.history.generator_loss.append(generator_loss.item())
        self.history.discriminator_loss.append(discriminator_loss.item())

        self.writer.add_scalar("Loss/Discriminator", discriminator_loss.item(), self.iteration)
        self.writer.add_scalar("Loss/Generator", generator_loss.item(), self.iteration)

    def save_accuracies(self, real_accuracy: float, fake_accuracy: float):
        self.history.discriminator_acc_real.append(real_accuracy)
        self.history.discriminator_acc_fake.append(fake_accuracy)

        self.writer.add_scalar("Accuracy/Real", real_accuracy, self.iteration)
        self.writer.add_scalar("Accuracy/Fake", fake_accuracy, self.iteration)

    def save_images(self):
        with torch.no_grad():
            images = self.generator(self.latent_space).detach().cpu()
            images = images.clamp(-1, 1)
            images_normalized = (images + 1) / 2

            grid = make_grid(images_normalized, nrow=8, normalize=True)
            self.writer.add_image("Generated Images", grid, self.iteration)

    def save_model(self):
        path = f"{self.config.save_dir}/{self.id}_depth_{self.generator.current_depth}_iteration_{self.iteration}.pth"
        os.makedirs(self.config.save_dir, exist_ok=True)
        saves = len(list(filter(lambda x: x.startswith(self.id), os.listdir(self.config.save_dir))))

        if saves >= self.config.no_concurrent_saves:
            # Oldest (sorted by lowest id)
            oldest = sorted(list(filter(lambda x: x.startswith(self.id), os.listdir(self.config.save_dir))))[0]
            os.remove(f"{self.config.save_dir}/{oldest}")

        os.makedirs(self.config.save_dir, exist_ok=True)

        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "generator_optimizer": self.generator_loss.wgan_gp_loss.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_loss.wgan_gp_loss.discriminator_optimizer.state_dict(),
            "depth": self.generator.current_depth,
            "iteration": self.iteration,
            "current_depth_iterations": self.current_depth_iterations,
            "current_batch_size": self.current_batch_size,
        }, path)

    def step(self):
        self.iteration += 1
        self.current_depth_iterations += 1

        if self.progress is not None:
            self.progress.update(1)
            self.progress.set_postfix(alpha=self.get_alpha(), iteration=self.iteration)

        if self.generator.current_depth <= self.generator.depth and self.current_depth_iterations >= self.config.steps_for_depth[self.generator.current_depth]:
            self.current_depth_iterations = 0
            self.generator.grow()
            self.discriminator.grow()

            self.progress = None
            self.init_progress_bar()

            if self.current_batch_size != self.config.batch_sizes_for_depth[self.generator.current_depth]:
                self.current_batch_size = self.config.batch_sizes_for_depth[self.generator.current_depth]
                self.data_loader = DataLoader(self.dataset, batch_size=self.current_batch_size, num_workers=self.config.num_workers, pin_memory=self.config.pin_memory)
                self.iterator = iter(self.data_loader)

    def init_progress_bar(self):
        if self.progress is None:
            self.progress = tqdm(
                desc=f"Training depth {self.generator.current_depth} - image size {4 * 2 ** self.generator.current_depth}x{4 * 2 ** self.generator.current_depth}",
                total=self.config.steps_for_depth[self.generator.current_depth],
                mininterval=1.0,
                initial=self.current_depth_iterations
            )
        else:
            self.progress.close()

    def get_alpha(self) -> float:
        if self.generator.current_depth == 0:
            return 1.0

        numerator = min(self.current_depth_iterations, self.config.transition_steps_for_depth[self.generator.current_depth - 1])
        denominator = self.config.transition_steps_for_depth[self.generator.current_depth - 1]
        return numerator / denominator

    def load_model(self, path: str):
        model_data = torch.load(path)

        print(f"Loading model from {path}...")

        self.generator.load_state_dict(model_data["generator"])
        self.discriminator.load_state_dict(model_data["discriminator"])
        self.generator_loss.wgan_gp_loss.generator_optimizer.load_state_dict(model_data["generator_optimizer"])
        self.discriminator_loss.wgan_gp_loss.discriminator_optimizer.load_state_dict(model_data["discriminator_optimizer"])

        self.generator.current_depth = model_data["depth"]
        self.discriminator.current_depth = model_data["depth"]
        self.iteration = model_data["iteration"]
        self.current_depth_iterations = model_data["current_depth_iterations"]
        self.current_batch_size = model_data["current_batch_size"]

        print(f"Loaded model at depth {self.generator.current_depth} and iteration {self.iteration}.")

    def update_ema(self, generator_orig: Generator, generator_ema: Generator, beta: float):
        with torch.no_grad():
            orig_params = dict(generator_orig.named_parameters())

            for param_name, ema_param in generator_ema.named_parameters():
                orig_param = orig_params[param_name]
                ema_param.copy_(beta * ema_param + (1.0 - beta) * orig_param)

    def compute_discriminator_accuracy(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, alpha: float) -> tuple[float, float]:
        self.discriminator.eval()

        with torch.no_grad():
            real_preds = self.discriminator(real_batch, alpha).view(-1)
            fake_preds = self.discriminator(fake_batch, alpha).view(-1)

            real_accuracy = (real_preds > 0.5).float().mean().item()
            fake_accuracy = (fake_preds < 0.5).float().mean().item()

        self.discriminator.train()  # Reset to training mode

        return real_accuracy, fake_accuracy
