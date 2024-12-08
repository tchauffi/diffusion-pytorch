import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from einops import rearrange

import numpy as np


class DiffusionImageLogger(Callback):
    def __init__(self, num_images=4, every_n_epochs=1, nb_steps=50):
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.nb_steps = nb_steps

        self.input_noise = torch.randn(self.num_images, 3, 64, 64, requires_grad=False)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.every_n_epochs == 0:
            init_noises = self.input_noise.to(pl_module.device)

            final_images = pl_module.reverse_diffusion(init_noises, self.nb_steps)

            final_images = final_images * 0.5 + 0.5

            final_images = torch.clamp(final_images, 0.0, 1.0)
            # make_grid expects images in CHW format
            final_images = rearrange(
                final_images, "b c h w -> h (b w) c", b=self.num_images
            )

            pl_module.logger.experiment.add_image(
                f"final_sample",
                final_images,
                global_step=trainer.global_step,
                dataformats="HWC",
            )
