import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
from datasets import load_dataset
from diffusers.vae.model import VAE, VAEModel

from torchvision.transforms import functional as F
import torchvision


class CenterCrop(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            # Size must match the ratio while cropping to the edge of the image
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w >= h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return torchvision.transforms.functional.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


# ============================================================================
# VAE Configuration Options (uncomment one):
# ============================================================================

# OPTION 1: Lightweight VAE (~5.6M params) - Fast training, decent quality
vae = VAE(
    latent_dim=4,
    base_channels=64,           # Reduced from 128
    channel_multipliers=(1, 2, 2, 4),  # Lighter multipliers
    num_res_blocks=2,           # Single ResNet block per level
    attention_resolutions=(16,),   # No attention (saves memory)
    dropout=0.0,
    input_resolution=128
)

# OPTION 2: Medium VAE (~22M params) - Good balance
# vae = VAE(
#     latent_dim=4,
#     base_channels=64,
#     channel_multipliers=(1, 2, 4, 4),
#     num_res_blocks=2,
#     attention_resolutions=(16,),
#     dropout=0.0,
#     input_resolution=128
# )

# OPTION 3: Full VAE (~89M params) - Best quality, slower
# vae = VAE(
#     latent_dim=4,
#     base_channels=128,
#     channel_multipliers=(1, 2, 4, 4),
#     num_res_blocks=2,
#     attention_resolutions=(16,),
#     dropout=0.0,
#     input_resolution=128
# )

# Print model size
total_params = sum(p.numel() for p in vae.parameters())
print(f"VAE parameters: {total_params:,}")

# Improved loss configuration:
# - use_lpips=True: Multi-layer perceptual loss (reduces artifacts)
# - use_l1=True: L1 loss produces sharper results than MSE
# - use_gan=False: Set to True for even sharper results (but harder to train)
# - perceptual_weight=0.5: Balanced perceptual weight
# - use_ema=True: Exponential Moving Average for smoother weights
model = VAEModel(
    vae, 
    lr=1e-4,                 # Lower LR for stability (was 5e-4)
    kl_weight=0.000001,      # Even lower KL to prevent explosion
    perceptual_weight=0.1,   # Lower perceptual weight for stability
    ssim_weight=0.0,         # SSIM can help but may slow convergence
    kl_anneal_epochs=20,     # Slower KL annealing (was 10)
    use_lpips=True,          # Use improved multi-layer perceptual loss
    use_l1=True,             # L1 loss for sharper results
    use_gan=True,            # Enable for sharper results
    gan_weight=0.05,         # Lower GAN weight for stability (was 0.1)
    disc_start_epoch=10,     # Start GAN later (was 5)
    use_ema=True,            # Enable EMA for smoother inference
    ema_decay=0.999         # EMA decay rate (higher = more stable)
)

# Setup logger
logger = TensorBoardLogger("logs", name="vae")

checkpoint_callback = ModelCheckpoint(
    dirpath="logs/vae/checkpoints",
    filename="vae-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True
)

trainer = pl.Trainer(
    max_epochs=100, 
    precision="16-mixed",
    logger=logger,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
)

dataset = load_dataset("huggan/AFHQv2")

# Split dataset into train/val/test since only train split is available
full_train_dataset = dataset["train"]

# Split: 80% train, 10% validation, 10% test
train_test_split = full_train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
temp_test = train_test_split["test"]

# Split the remaining 20% into 50/50 for validation and test (10% each of original)
val_test_split = temp_test.train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

transfrom = transforms.Compose(
    [
        CenterCrop(),
        transforms.Resize((128, 128), Image.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

val_transform = transforms.Compose(
    [
        CenterCrop(),
        transforms.Resize((128, 128), Image.LANCZOS),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def transforms_fn(examples):
    examples["image"] = [transfrom(image.convert("RGB")) for image in examples["image"]]
    return examples


def val_transforms_fn(examples):
    examples["image"] = [val_transform(image.convert("RGB")) for image in examples["image"]]
    return examples


train_dataset.set_transform(transforms_fn)
val_dataset.set_transform(val_transforms_fn)
test_dataset.set_transform(val_transforms_fn)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=7, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=7, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=7, pin_memory=True)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

# Save EMA model separately for inference
if model.use_ema and model.ema is not None:
    ema_vae = model.get_ema_vae()
    ema_checkpoint_path = "logs/vae/checkpoints/vae-ema-final.pt"
    torch.save({
        'vae_state_dict': ema_vae.state_dict(),
        'config': {
            'latent_dim': vae.latent_dim,
            'input_resolution': vae.input_resolution,
        }
    }, ema_checkpoint_path)
    print(f"EMA VAE saved to {ema_checkpoint_path}")