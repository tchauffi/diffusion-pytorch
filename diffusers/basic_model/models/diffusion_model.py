from typing import Any, Callable
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from tqdm import tqdm

from timm.utils import ModelEmaV3
from .unet import UNet
from ..schedulers import (
    offset_cosine_diffusion_scheduler,
    cosine_diffusion_scheduler,
    linear_diffusion_scheduler,
)

generator = torch.manual_seed(0)


class DiffusionModel(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_filters, num_steps, lr=1e-3):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, num_filters)
        self.num_steps = num_steps

        steps = torch.arange(num_steps, dtype=torch.float32) / num_steps
        self.alphas, self.betas = offset_cosine_diffusion_scheduler(steps)
        self.criterion = torch.nn.L1Loss(reduction="mean")
        self.ema = ModelEmaV3(self.unet, decay=0.999)

        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x = batch["image"].to(self.device)
        t = torch.randint(0, self.num_steps, (x.shape[0],))
        noise = torch.randn(x.shape, generator=generator, requires_grad=False).to(
            x.device
        )

        alpha = self.alphas[t].reshape(x.shape[0], 1, 1, 1).to(x.device)
        beta = self.betas[t].reshape(x.shape[0], 1, 1, 1).to(x.device)
        noisy_image = alpha * x + beta * noise
        pred_noises = self.unet(noisy_image, t)
        loss = self.criterion(pred_noises, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def reverse_diffusion(self, initial_noises, diffusion_steps):
        num_images = initial_noises.shape[0]
        current_images = initial_noises
        steps = torch.arange(0, diffusion_steps, device=self.device)
        alphas, betas = offset_cosine_diffusion_scheduler(steps)

        for step in tqdm(reversed(steps), desc="Reversing Diffusion"):
            alpha = alphas[step].repeat(num_images).reshape(num_images, 1, 1, 1)
            beta = betas[step].repeat(num_images).reshape(num_images, 1, 1, 1)
            with torch.no_grad():
                t = torch.full((num_images,), step, device=self.device)
                self.ema.eval()
                pred_noise = self.ema(current_images, t)

                pred_image = (current_images - pred_noise * beta) / alpha

                alpha_minus_one = (
                    alphas[step - 1].repeat(num_images).reshape(num_images, 1, 1, 1)
                )
                beta_minus_one = (
                    betas[step - 1].repeat(num_images).reshape(num_images, 1, 1, 1)
                )

                current_images = (
                    alpha_minus_one * pred_image + beta_minus_one * pred_noise
                )

        return pred_image

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.unet)
