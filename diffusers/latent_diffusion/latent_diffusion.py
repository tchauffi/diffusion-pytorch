"""
Latent Diffusion Model

Combines a VAE (for encoding/decoding images to/from latent space) with a 
diffusion model that operates in the compressed latent space. This is the 
architecture used by Stable Diffusion.

Benefits:
- Much faster training and inference (smaller latent space)
- Lower memory usage
- Can generate higher resolution images
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from timm.utils import ModelEmaV3

try:
    from xformers.ops import memory_efficient_attention
    # Check if xformers CUDA kernels are actually usable
    # They may be installed but not support the current GPU
    import xformers
    if hasattr(xformers, '_has_cpp_library') and not xformers._has_cpp_library:
        XFORMERS_AVAILABLE = False
    else:
        # Try a small test to verify it works
        try:
            _test = torch.randn(1, 1, 1, 16, device='cuda' if torch.cuda.is_available() else 'cpu')
            memory_efficient_attention(_test, _test, _test)
            XFORMERS_AVAILABLE = True
        except (NotImplementedError, RuntimeError):
            XFORMERS_AVAILABLE = False
            print("xformers installed but not supported on this GPU, using PyTorch SDPA")
except ImportError:
    XFORMERS_AVAILABLE = False

from ..vae.vae import VAE
from ..basic_model.schedulers import offset_cosine_diffusion_scheduler


def timestep_embedding(timesteps, dim, max_period=10000.0):
    """Generate sinusoidal timestep embeddings."""
    half_dim = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period)) * torch.arange(half_dim, device=timesteps.device) / half_dim
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock(nn.Module):
    """Residual block with time embedding for the latent UNet."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for latent UNet."""
    
    def __init__(self, channels, num_heads=4, use_xformers=True):
        super().__init__()
        self.num_heads = num_heads
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        if self.use_xformers:
            # xformers expects (B, N, H, D) format
            q = q.permute(0, 3, 1, 2).contiguous()  # (B, H*W, heads, dim)
            k = k.permute(0, 3, 1, 2).contiguous()
            v = v.permute(0, 3, 1, 2).contiguous()
            out = memory_efficient_attention(q, k, v)
            out = out.permute(0, 2, 3, 1).reshape(b, c, h, w)
        else:
            # PyTorch SDPA expects (B, H, N, D) format
            q = q.permute(0, 1, 3, 2).contiguous()  # (B, heads, H*W, dim)
            k = k.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2).contiguous()
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        
        out = self.proj(out)
        
        return x + out


class DownBlock(nn.Module):
    """Downsampling block for latent UNet."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False, dropout=0.1):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim, dropout)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim, dropout)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for latent UNet."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False, dropout=0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.res1 = ResBlock(in_channels + out_channels, out_channels, time_emb_dim, dropout)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim, dropout)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
    
    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return x


class LatentUNet(nn.Module):
    """UNet architecture for diffusion in latent space.
    
    Designed to work with VAE latent representations (typically 4 channels).
    """
    
    def __init__(
        self, 
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_levels: tuple = (1, 2),  # Which levels get attention
        dropout: float = 0.1,
        time_emb_dim: int = 256,
        cond_classes: int = None
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        if cond_classes is not None:
            self.label_emb = nn.Embedding(cond_classes + 1, time_emb_dim)  # +1 for null class
        else:
            self.label_emb = None
        
        # Initial convolution
        self.conv_in = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels
        
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            has_attn = level in attention_levels
            self.down_blocks.append(DownBlock(in_ch, out_ch, time_emb_dim, has_attn, dropout))
            channels.append(out_ch)
            in_ch = out_ch
        
        # Middle blocks
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_res1 = ResBlock(mid_channels, mid_channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_channels)
        self.mid_res2 = ResBlock(mid_channels, mid_channels, time_emb_dim, dropout)
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            has_attn = (len(channel_multipliers) - 1 - level) in attention_levels
            self.up_blocks.append(UpBlock(in_ch, out_ch, time_emb_dim, has_attn, dropout))
            in_ch = out_ch
        
        # Output
        self.norm_out = nn.GroupNorm(min(32, base_channels), base_channels)
        self.conv_out = nn.Conv2d(base_channels, latent_channels, 3, padding=1)
        
        # Initialize output conv to zero for stable training
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x, t, c=None):
        """
        Args:
            x: Noisy latent tensor [B, latent_channels, H, W]
            t: Normalized timesteps [B] in range [0, 1]
            c: Class labels [B] (optional, for class-conditional generation)
        
        Returns:
            Predicted noise [B, latent_channels, H, W]
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Add class embedding if available
        if self.label_emb is not None and c is not None:
            c_emb = self.label_emb(c)
            t_emb = t_emb + c_emb
        
        # Initial conv
        x = self.conv_in(x)
        
        # Downsampling
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb)
            skips.append(skip)
        
        # Middle
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)
        
        # Upsampling
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, t_emb)
        
        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x


class LatentDiffusionModel(pl.LightningModule):
    """Latent Diffusion Model combining VAE and diffusion in latent space.
    
    This model:
    1. Encodes images to latent space using a pretrained VAE
    2. Runs diffusion (noising/denoising) in the latent space
    3. Decodes denoised latents back to images
    
    Args:
        vae: Pretrained VAE model (will be frozen)
        latent_channels: Number of latent channels (should match VAE)
        base_channels: Base channels for latent UNet
        channel_multipliers: Channel scaling at each UNet level
        num_steps: Number of diffusion timesteps
        lr: Learning rate
        use_ema: Use exponential moving average for UNet weights
        ema_decay: EMA decay rate
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
        
        # Latent UNet for denoising
        self.unet = LatentUNet(
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_classes=num_classes
        )
        
        # Diffusion schedule
        self.num_steps = num_steps
        steps = torch.arange(num_steps, dtype=torch.float32)
        self.register_buffer('alphas', None)
        self.register_buffer('betas', None)
        alphas, betas = offset_cosine_diffusion_scheduler(steps / num_steps)
        self.alphas = alphas
        self.betas = betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # EMA
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = None
        
        self.lr = lr
        
        # VAE scaling factor - normalize latents to have roughly unit variance
        # Your VAE produces latents with std ~2.6, so we scale by 1/std ≈ 0.38
        # This helps the diffusion model as it expects inputs close to unit variance
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
        
        # Encode to latent space
        z = self.encode(x)
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (z.shape[0],), device=self.device)

        # Get class labels if available
        c = batch.get("label", None)
        
        # Classifier-free guidance: randomly drop class labels during training
        if c is not None and self.num_classes is not None:
            # Create dropout mask - set dropped labels to num_classes (null class)
            drop_mask = torch.rand(z.shape[0], device=self.device) < self.cfg_dropout_prob
            c = c.clone()
            c[drop_mask] = self.num_classes  # null class index
        
        # Sample noise
        noise = torch.randn_like(z)
        
        # Get diffusion parameters for these timesteps
        alpha = self.alphas[t].view(-1, 1, 1, 1).to(self.device)
        beta = self.betas[t].view(-1, 1, 1, 1).to(self.device)
        
        # Create noisy latents: z_t = beta * z + alpha * noise
        z_noisy = beta * z + alpha * noise

        # Compute SNR-based weighting
        snr = self.compute_snr(t)
        base_weight = (
            torch.stack([snr, 5.0 * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
            )

        base_weight[snr == 0] = 1.0
        
        # Predict noise (with class conditioning)
        noise_pred = self.unet(z_noisy, t.float() / self.num_steps, c)
        
        # Compute loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='none')
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * base_weight
        loss = loss.mean()
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training step."""
        if self.use_ema and self.ema is not None:
            self.ema.update(self.unet)
    
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        
        # Encode to latent space
        z = self.encode(x)
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (z.shape[0],), device=self.device)
        
        # Get class labels if available
        c = batch.get("label", None)
        
        # Sample noise
        noise = torch.randn_like(z)
        
        # Get diffusion parameters
        alpha = self.alphas[t].view(-1, 1, 1, 1).to(self.device)
        beta = self.betas[t].view(-1, 1, 1, 1).to(self.device)
        
        # Create noisy latents
        z_noisy = beta * z + alpha * noise
        
        # Predict noise (use EMA if available)
        if self.use_ema and self.ema is not None:
            self.ema.eval()
            noise_pred = self.ema.module(z_noisy, t.float() / self.num_steps, c)
        else:
            noise_pred = self.unet(z_noisy, t.float() / self.num_steps, c)
        
        loss = self.criterion(noise_pred, noise)
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
            samples = self.sample(num_samples, z_shape, 40, class_labels=class_labels, cfg_scale=3.0)
        else:
            samples = self.sample(num_samples, z_shape, 40)
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        import torchvision
        grid = torchvision.utils.make_grid(samples, nrow=num_samples)
        
        if self.logger:
            self.logger.experiment.add_image("samples", grid, self.current_epoch)
    
    @torch.no_grad()
    def sample(self, num_samples, latent_shape, num_inference_steps=None, eta=0.0, 
               class_labels=None, cfg_scale=1.0):
        """Generate samples using DDIM reverse diffusion in latent space.
        
        DDIM (Denoising Diffusion Implicit Models) allows for:
        - Deterministic sampling (eta=0) - same noise -> same image
        - Faster sampling with fewer steps while maintaining quality
        - Optional stochasticity (eta>0) for diversity
        - Class-conditional generation with classifier-free guidance
        
        Args:
            num_samples: Number of images to generate
            latent_shape: Shape of latent (C, H, W)
            num_inference_steps: Number of denoising steps (can be much less than training steps)
            eta: DDIM stochasticity parameter (0=deterministic, 1=DDPM-like stochastic)
            class_labels: Class labels for conditional generation [num_samples] or single int
            cfg_scale: Classifier-free guidance scale (1.0=no guidance, >1.0=stronger guidance)
        
        Returns:
            Generated images [num_samples, 3, H, W]
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_steps
        
        # Start with pure noise
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
        
        # Create a subsequence of timesteps for faster sampling
        # E.g., if training used 1000 steps but we sample with 50, we skip steps
        step_ratio = self.num_steps / num_inference_steps
        timesteps = (torch.arange(num_inference_steps, device=self.device) * step_ratio).long()
        timesteps = torch.flip(timesteps, [0])  # Reverse: from high noise to low
        
        # Get alphas and betas for all training steps
        all_steps = torch.arange(self.num_steps, device=self.device)
        all_alphas, all_betas = offset_cosine_diffusion_scheduler(all_steps / self.num_steps)
        all_alphas = all_alphas.to(self.device)  # noise rate (sin) = sqrt(1 - alpha_bar)
        all_betas = all_betas.to(self.device)    # signal rate (cos) = sqrt(alpha_bar)
        
        # DDIM sampling loop
        for i in tqdm(range(len(timesteps)), desc="DDIM Sampling"):
            t = timesteps[i]
            t_input = torch.full((num_samples,), t.item(), device=self.device, dtype=torch.float32)
            
            # Current noise and signal rates
            alpha_t = all_alphas[t].view(1, 1, 1, 1)  # noise rate at t
            beta_t = all_betas[t].view(1, 1, 1, 1)    # signal rate at t
            
            # Predict noise with classifier-free guidance
            if use_cfg:
                # Conditional prediction
                noise_pred_cond = model(z, t_input / self.num_steps, class_labels)
                # Unconditional prediction
                noise_pred_uncond = model(z, t_input / self.num_steps, null_labels)
                # CFG: combine predictions
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No CFG - just conditional or unconditional prediction
                c = class_labels if (class_labels is not None and self.num_classes is not None) else None
                noise_pred = model(z, t_input / self.num_steps, c)
            
            # DDIM: predict x0 from xt and predicted noise
            # z_t = beta_t * z_0 + alpha_t * noise
            # z_0 = (z_t - alpha_t * noise_pred) / beta_t
            z0_pred = (z - alpha_t * noise_pred) / beta_t
            
            if i < len(timesteps) - 1:
                # Get previous timestep
                t_prev = timesteps[i + 1]
                alpha_prev = all_alphas[t_prev].view(1, 1, 1, 1)
                beta_prev = all_betas[t_prev].view(1, 1, 1, 1)
                
                # DDIM update rule (deterministic when eta=0)
                # Direction pointing to x_t
                pred_noise_direction = alpha_prev * noise_pred
                
                # Deterministic DDIM step
                z = beta_prev * z0_pred + pred_noise_direction
                
                # Optional stochasticity (eta > 0 adds randomness like DDPM)
                if eta > 0:
                    # Variance for stochastic sampling
                    sigma_t = eta * torch.sqrt((1 - beta_prev**2) / (1 - beta_t**2) * (1 - beta_t**2 / beta_prev**2))
                    noise = torch.randn_like(z)
                    z = z + sigma_t * noise
            else:
                # Final step: use predicted clean latent
                z = z0_pred
        
        # Decode to image space
        images = self.decode(z)
        
        return images
    
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
    
    def compute_snr(self, t):
        """Compute signal-to-noise ratio at timestep t."""
        alpha_cumprod_t = self.alpha_cumprod.to(t.device)[t]
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)

        alpha = sqrt_alpha_cumprod
        sigma = sqrt_one_minus_alpha_cumprod

        snr = (alpha / sigma) ** 2
        return snr

if __name__ == "__main__":
    # Test the latent diffusion model
    print("Testing Latent Diffusion Model...")
    
    # Create a small VAE for testing
    vae = VAE(
        latent_dim=4,
        base_channels=32,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        input_resolution=64
    )
    
    # Create latent diffusion model
    ldm = LatentDiffusionModel(
        vae=vae,
        latent_channels=4,
        base_channels=64,
        channel_multipliers=(1, 2, 4),
        num_steps=100,
        use_ema=False
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 64)
    z = ldm.encode(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    
    # Test UNet
    t = torch.randint(0, 100, (2,))
    noise = torch.randn_like(z)
    noise_pred = ldm.unet(z, t.float() / 100)
    print(f"Noise prediction shape: {noise_pred.shape}")
    
    # Test sampling
    print("\nTesting sampling...")
    samples = ldm.sample(2, z.shape[1:], num_inference_steps=10)
    print(f"Sample shape: {samples.shape}")
    
    # Count parameters
    unet_params = sum(p.numel() for p in ldm.unet.parameters())
    vae_params = sum(p.numel() for p in ldm.vae.parameters())
    print(f"\nUNet parameters: {unet_params:,}")
    print(f"VAE parameters (frozen): {vae_params:,}")
    
    print("\n✓ All tests passed!")
