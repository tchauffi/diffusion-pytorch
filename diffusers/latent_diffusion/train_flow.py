"""
Training script for Flow Matching Latent Diffusion Model.

This script:
1. Loads a pretrained VAE checkpoint
2. Freezes the VAE and trains only the UNet for flow matching
3. Saves checkpoints and logs to TensorBoard

Flow Matching learns to predict the velocity v = z_1 - z_0 between
data (z_0) and noise (z_1), enabling efficient ODE-based sampling.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from PIL import Image
from datasets import load_dataset

from diffusers.vae.model import VAE
from diffusers.latent_diffusion.flow_latent_diff import FlowLatentDiffusionModel

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set up cuda deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# allow tf32 on ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# Configuration
# ============================================================================

# VAE checkpoint path (from your trained VAE)
VAE_CHECKPOINT_PATH = "logs/vae/checkpoints/vae-ema-final.pt"  # or your best checkpoint

# Image settings (should match VAE training)
IMAGE_SIZE = 128

# Latent Diffusion settings
LATENT_CHANNELS = 4
BASE_CHANNELS = 256         # UNet base channels
CHANNEL_MULTIPLIERS = (1, 2, 2, 4)  # UNet channel scaling
NUM_DIFFUSION_STEPS = 1000
LEARNING_RATE = 4e-4        # Increased LR for faster convergence
USE_EMA = True
EMA_DECAY = 0.995          # Higher decay for more stable EMA
NUM_CLASSES = 3            # Number of classes in the dataset

# Training settings
BATCH_SIZE = 64             # Larger batch for more stable gradients
MAX_EPOCHS = 200            # More epochs for better convergence
NUM_WORKERS = 7


# ============================================================================
# Load pretrained VAE
# ============================================================================

def load_vae(checkpoint_path: str, device='cpu') -> VAE:
    """Load a pretrained VAE from checkpoint."""
    
    # Create VAE with same architecture as training
    vae = VAE(
        latent_dim=LATENT_CHANNELS,
        base_channels=64,
        channel_multipliers=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0,
        input_resolution=IMAGE_SIZE
    )
    
    # Load weights
    try:
        # Try loading EMA checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'vae_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['vae_state_dict'])
            print(f"Loaded VAE from EMA checkpoint: {checkpoint_path}")
        elif 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = {k.replace('vae.', ''): v for k, v in checkpoint['state_dict'].items() 
                          if k.startswith('vae.')}
            vae.load_state_dict(state_dict)
            print(f"Loaded VAE from Lightning checkpoint: {checkpoint_path}")
        else:
            vae.load_state_dict(checkpoint)
            print(f"Loaded VAE from state dict: {checkpoint_path}")
    except FileNotFoundError:
        print(f"WARNING: VAE checkpoint not found at {checkpoint_path}")
        print("Training with randomly initialized VAE (not recommended for production)")
    
    return vae


# ============================================================================
# Data transforms (same as VAE training)
# ============================================================================

class CenterCrop(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
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
        import torchvision
        return torchvision.transforms.functional.center_crop(img, size)


train_transform = transforms.Compose([
    CenterCrop(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    CenterCrop(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def train_transforms_fn(examples):
    examples["image"] = [train_transform(image.convert("RGB")) for image in examples["image"]]
    return examples


def val_transforms_fn(examples):
    examples["image"] = [val_transform(image.convert("RGB")) for image in examples["image"]]
    return examples


# ============================================================================
# Main training
# ============================================================================

if __name__ == "__main__":
    # Load pretrained VAE
    print("Loading pretrained VAE...")
    vae = load_vae(VAE_CHECKPOINT_PATH)
    
    # Create Flow Matching Latent Diffusion Model
    print("Creating Flow Matching Latent Diffusion Model...")
    model = FlowLatentDiffusionModel(
        vae=vae,
        latent_channels=LATENT_CHANNELS,
        base_channels=BASE_CHANNELS,
        channel_multipliers=CHANNEL_MULTIPLIERS,
        num_steps=NUM_DIFFUSION_STEPS,
        lr=LEARNING_RATE,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
        num_classes=NUM_CLASSES
    )
    
    # Print model info
    unet_params = sum(p.numel() for p in model.unet.parameters())
    vae_params = sum(p.numel() for p in model.vae.parameters())
    print(f"UNet parameters: {unet_params:,}")
    print(f"VAE parameters (frozen): {vae_params:,}")
    
    # Setup logger
    logger = TensorBoardLogger("logs", name="latent_diffusion")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/latent_diffusion/checkpoints",
        filename="ldm-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("huggan/AFHQ")
    
    # Split dataset
    full_train_dataset = dataset["train"]
    train_test_split = full_train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]
    
    # Apply transforms
    train_dataset.set_transform(train_transforms_fn)
    val_dataset.set_transform(val_transforms_fn)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    print("Saving final model...")
    torch.save({
        'unet_state_dict': model.unet.state_dict(),
        'ema_state_dict': model.ema.module.state_dict() if model.ema else None,
        'config': {
            'latent_channels': LATENT_CHANNELS,
            'base_channels': BASE_CHANNELS,
            'channel_multipliers': CHANNEL_MULTIPLIERS,
            'num_steps': NUM_DIFFUSION_STEPS,
            'num_classes': NUM_CLASSES
        }
    }, "logs/latent_diffusion/checkpoints/ldm-final.pt")
    
    print("Training complete!")
