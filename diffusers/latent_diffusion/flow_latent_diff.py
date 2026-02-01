import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from timm.utils import ModelEmaV3

from ..vae.model import VAE
from .latent_diffusion import LatentUNet


class FlowLatentDiffusionModel(pl.LightningModule):
    """Latent Flow Matching Model combining VAE and flow matching in latent space.
    
    This model implements Conditional Flow Matching (CFM) / Rectified Flow:
    1. Encodes images to latent space using a pretrained VAE
    2. Learns to predict the velocity field v = z_1 - z_0 (noise - data)
    3. Samples by integrating the ODE from noise to data
    
    Flow Matching formulation:
    - z_t = (1 - t) * z_0 + t * z_1  (linear interpolation)
    - v_target = z_1 - z_0  (constant velocity along the path)
    - The model predicts v(z_t, t) ≈ v_target
    
    Args:
        vae: Pretrained VAE model (will be frozen)
        latent_channels: Number of latent channels (should match VAE)
        base_channels: Base channels for latent UNet
        channel_multipliers: Channel scaling at each UNet level
        num_steps: Number of diffusion timesteps (for discretization)
        lr: Learning rate
        use_ema: Use exponential moving average for UNet weights
        ema_decay: EMA decay rate
        num_classes: Number of classes for conditional generation
        cfg_dropout_prob: Probability of dropping class for CFG training
    """
    
    def __init__(
        self,
        vae: VAE,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: tuple = (1, 2, 4),
        num_steps: int = 1000,
        lr: float = 1e-4,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        num_classes: int = None,
        cfg_dropout_prob: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        
        # Freeze VAE - we only train the UNet
        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Class conditioning
        self.num_classes = num_classes
        self.cfg_dropout_prob = cfg_dropout_prob  # Probability of dropping class for CFG training
        
        # Latent UNet for velocity prediction
        self.unet = LatentUNet(
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_classes=num_classes
        )
        
        # Number of steps (for discretization during sampling)
        self.num_steps = num_steps
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # EMA
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = None
        
        self.lr = lr
        
        # VAE scaling factor - normalize latents to have roughly unit variance
        # Your VAE produces latents with std ~2.6, so we scale by 1/std ≈ 0.38
        # This helps the flow model as it expects inputs close to unit variance
        self.scale_factor = 0.38  # Adjusted for your VAE (was 0.18215 for SD)
    
    def on_fit_start(self):
        """Initialize EMA when training starts."""
        if self.use_ema:
            self.ema = ModelEmaV3(self.unet, decay=self.ema_decay)
            print(f"EMA initialized with decay={self.ema_decay}")
    
    @torch.no_grad()
    def encode(self, x):
        """Encode images to latent space."""
        self.vae.eval()
        mu, logvar = self.vae.encode(x)
        # Use mean for encoding (no sampling during diffusion training)
        z = mu * self.scale_factor
        return z
    
    @torch.no_grad()
    def decode(self, z):
        """Decode latents to images."""
        self.vae.eval()
        z = z / self.scale_factor
        return self.vae.decode(z)
    
    def training_step(self, batch, batch_idx):
        x = batch["image"]
        
        # Encode to latent space (z_0 = data)
        z_0 = self.encode(x)
        
        # Sample random timesteps t ∈ [0, 1]
        t = torch.rand(z_0.shape[0], device=self.device)

        # Get class labels if available
        c = batch.get("label", None)
        
        # Classifier-free guidance: randomly drop class labels during training
        if c is not None and self.num_classes is not None:
            # Create dropout mask - set dropped labels to num_classes (null class)
            drop_mask = torch.rand(z_0.shape[0], device=self.device) < self.cfg_dropout_prob
            c = c.clone()
            c[drop_mask] = self.num_classes  # null class index
        
        # Sample noise (z_1 = noise)
        z_1 = torch.randn_like(z_0)
        
        # Flow Matching: linear interpolation between data and noise
        # z_t = (1 - t) * z_0 + t * z_1
        t_expanded = t.view(-1, 1, 1, 1)
        z_t = (1 - t_expanded) * z_0 + t_expanded * z_1
        
        # Target velocity: v = z_1 - z_0 (constant along the path)
        v_target = z_1 - z_0
        
        # Model predicts the velocity
        v_pred = self.unet(z_t, t, c)
        
        # MSE loss on velocity prediction
        loss = self.criterion(v_pred, v_target)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training step."""
        if self.use_ema and self.ema is not None:
            self.ema.update(self.unet)
    
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        
        # Encode to latent space (z_0 = data)
        z_0 = self.encode(x)
        
        # Sample random timesteps t ∈ [0, 1]
        t = torch.rand(z_0.shape[0], device=self.device)
        
        # Get class labels if available
        c = batch.get("label", None)
        
        # Sample noise (z_1 = noise)
        z_1 = torch.randn_like(z_0)
        
        # Flow Matching: linear interpolation
        t_expanded = t.view(-1, 1, 1, 1)
        z_t = (1 - t_expanded) * z_0 + t_expanded * z_1
        
        # Target velocity
        v_target = z_1 - z_0
        
        # Predict velocity (use EMA if available)
        if self.use_ema and self.ema is not None:
            self.ema.eval()
            v_pred = self.ema.module(z_t, t, c)
        else:
            v_pred = self.unet(z_t, t, c)
        
        loss = self.criterion(v_pred, v_target)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        
        # Generate samples on first batch
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_samples(x.shape[0])
        
        return loss
    
    @torch.no_grad()
    def _log_samples(self, num_samples=4):
        """Generate and log sample images."""
        num_samples = min(num_samples, 4)
        
        # Get latent shape from a dummy encoding
        dummy = torch.zeros(1, 3, self.vae.input_resolution, self.vae.input_resolution, device=self.device)
        z_shape = self.encode(dummy).shape[1:]
        
        # Generate class labels if class-conditional
        if self.num_classes is not None:
            # Sample different classes (cycle through if num_samples > num_classes)
            class_labels = torch.arange(num_samples, device=self.device) % self.num_classes
            samples = self.sample(num_samples, z_shape, num_inference_steps=40, 
                                  class_labels=class_labels, cfg_scale=3.0, sampler="euler")
        else:
            samples = self.sample(num_samples, z_shape, num_inference_steps=40, sampler="euler")
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        import torchvision
        grid = torchvision.utils.make_grid(samples, nrow=num_samples)
        
        if self.logger:
            self.logger.experiment.add_image("samples", grid, self.current_epoch)
    
    @torch.no_grad()
    def sample(self, num_samples, latent_shape, num_inference_steps=None,
               class_labels=None, cfg_scale=1.0, sampler="euler"):
        """Generate samples using Flow Matching ODE integration.
        
        Flow Matching samples by integrating the ODE:
            dz/dt = v(z, t)
        from t=1 (noise) to t=0 (data).
        
        Args:
            num_samples: Number of samples to generate
            latent_shape: Shape of latent tensors (C, H, W)
            num_inference_steps: Number of integration steps
            class_labels: Class labels for conditional generation
            cfg_scale: Classifier-free guidance scale (>1.0 for stronger conditioning)
            sampler: Integration method - "euler" or "heun"
            
        Returns:
            Generated images in pixel space
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_steps
        
        # Start with pure noise at t=1
        z = torch.randn(num_samples, *latent_shape, device=self.device)
        
        # Use EMA model if available
        model = self.ema.module if (self.use_ema and self.ema is not None) else self.unet
        model.eval()
        
        # Prepare class labels for CFG
        use_cfg = self.num_classes is not None and class_labels is not None and cfg_scale > 1.0
        if class_labels is not None and self.num_classes is not None:
            if isinstance(class_labels, int):
                class_labels = torch.full((num_samples,), class_labels, device=self.device, dtype=torch.long)
            else:
                class_labels = class_labels.to(self.device)
            # Null class for unconditional prediction in CFG
            null_labels = torch.full((num_samples,), self.num_classes, device=self.device, dtype=torch.long)
        
        # Time steps from t=1 (noise) to t=0 (data)
        # We integrate backwards: t goes from 1 -> 0
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device)
        
        # Helper function to compute velocity with optional CFG (batched for efficiency)
        def get_velocity(z_t, t_scalar):
            t_batch = torch.full((num_samples,), t_scalar, device=self.device, dtype=torch.float32)
            
            if use_cfg:
                # Batch conditional and unconditional together for single forward pass
                z_in = torch.cat([z_t, z_t], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)
                labels_in = torch.cat([class_labels, null_labels], dim=0)
                
                v_both = model(z_in, t_in, labels_in)
                v_cond, v_uncond = v_both.chunk(2, dim=0)
                # CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                c = class_labels if (class_labels is not None and self.num_classes is not None) else None
                v = model(z_t, t_batch, c)
            return v
        
        # Choose sampler
        if sampler == "euler":
            z = self._euler_sample(z, timesteps, get_velocity)
        elif sampler == "heun":
            z = self._heun_sample(z, timesteps, get_velocity)
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'euler' or 'heun'.")
        
        # Decode to image space
        images = self.decode(z)
        
        return images
    
    def _euler_sample(self, z, timesteps, get_velocity):
        """Euler method for ODE integration.
        
        The simplest first-order method:
            z_{i+1} = z_i + dt * v(z_i, t_i)
            
        where dt = t_{i+1} - t_i (negative since we go from 1 to 0)
        
        Args:
            z: Initial noise tensor at t=1
            timesteps: Array of time points from 1 to 0
            get_velocity: Function to compute velocity v(z, t)
            
        Returns:
            Final z at t=0
        """
        for i in tqdm(range(len(timesteps) - 1), desc="Euler Sampling"):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t  # Negative step (going from 1 to 0)
            
            # Compute velocity at current point
            v = get_velocity(z, t)
            
            # Euler step: z = z + dt * v
            z = z + dt * v
        
        return z
    
    def _heun_sample(self, z, timesteps, get_velocity):
        """Heun's method (improved Euler / 2nd order Runge-Kutta).
        
        A second-order method that uses a predictor-corrector approach:
            1. Predict: z_pred = z_i + dt * v(z_i, t_i)
            2. Correct: z_{i+1} = z_i + dt/2 * (v(z_i, t_i) + v(z_pred, t_{i+1}))
        
        This provides better accuracy than Euler with ~2x the compute.
        
        Args:
            z: Initial noise tensor at t=1
            timesteps: Array of time points from 1 to 0
            get_velocity: Function to compute velocity v(z, t)
            
        Returns:
            Final z at t=0
        """
        for i in tqdm(range(len(timesteps) - 1), desc="Heun Sampling"):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t  # Negative step (going from 1 to 0)
            
            # Compute velocity at current point
            v1 = get_velocity(z, t)
            
            # Predictor: Euler step to estimate z at t_next
            z_pred = z + dt * v1
            
            # Compute velocity at predicted point
            v2 = get_velocity(z_pred, t_next)
            
            # Corrector: average the two velocities
            z = z + dt * 0.5 * (v1 + v2)
        
        return z
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),  # Lower beta2 for diffusion models
            weight_decay=1e-2
        )
        
        # Cosine decay with linear warmup - the standard for diffusion models
        # Warmup: linearly increase LR from 0 to lr over warmup_steps
        # Then: cosine decay from lr to min_lr
        warmup_steps = 1000
        min_lr = self.lr * 0.01
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, self.trainer.estimated_stepping_batches - warmup_steps))
                progress = min(progress, 1.0)
                return max(min_lr / self.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update per step, not epoch
                "frequency": 1
            }
        }
    
    @classmethod
    def from_pretrained(cls, vae: VAE, unet_checkpoint: str):
        """Load a LatentDiffusionModel from pretrained UNet weights."""
        import json
        from safetensors.torch import load_file

        config_path = unet_checkpoint.replace('.safetensors', '.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = cls(
            vae=vae,
            **config
        )
        model.unet.load_state_dict(load_file(unet_checkpoint))
        return model