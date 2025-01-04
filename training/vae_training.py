
import os
import time
import torch
import shutil

from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.vae_training_config import VAETrainingConfig
from training.losses.vae_loss import VAELoss
from torch.optim.optimizer import Optimizer
from net.vae.model import VAE


class VAETraining:
    def __init__(self, vae: VAE, vae_loss: VAELoss, config: VAETrainingConfig):
        self.vae = vae
        self.vae_loss = vae_loss
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0

        self.id = str(round(time.time()))
        self.writer = SummaryWriter(log_dir=f"{self.config.log_dir}/{self.id}")

    def train(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers, pin_memory=self.config.pin_memory)
        for epoch in range(self.start_epoch, self.config.epochs):
            self.vae.train()
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for batch_idx, (data, _) in enumerate(progress_bar):
                data = data.to(self.device)
                data = data / 2 + 0.5
                loss, reproduction_loss, kl_divergence_loss = self.vae_loss(self.vae, data)

                if batch_idx % self.config.log_interval == 0:
                    self.writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + batch_idx)

                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'recon': reproduction_loss.item(),
                    'kl': kl_divergence_loss.item()
                })

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_model(epoch + 1)

    def save_model(self, epoch):
        dir_path = os.path.join(self.config.save_dir, f"{self.id}_{epoch}")
        os.makedirs(dir_path, exist_ok=True)

        dirs = os.listdir(self.config.save_dir)
        dirs = [d for d in dirs if d.startswith(self.id)]

        MAX_DIRS = 5

        if len(dirs) > MAX_DIRS:
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            for i in range(len(dirs) - MAX_DIRS):
                shutil.rmtree(os.path.join(self.config.save_dir, dirs[i]))
                # Remove the oldest directories

        model_path = os.path.join(dir_path, "model.pth")
        torch.save(self.vae.state_dict(), model_path)
        optim_path = os.path.join(dir_path, "optimizer.pth")
        torch.save(self.vae_loss.optimizer.state_dict(), optim_path)\

        print(f"Model saved to {dir_path}")

    def load_model(self, model_path: str, optimizer_path: str):
        self.vae.load_state_dict(torch.load(model_path))
        self.vae_loss.optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"Model loaded from {model_path}")
        self.start_epoch = int(model_path.split("/")[1].split("_")[1])