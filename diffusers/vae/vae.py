import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.models import vgg16, VGG16_Weights
import copy


class EMA:
    """Exponential Moving Average for model weights.
    
    Maintains a shadow copy of model parameters that is updated with
    exponential moving average. Used for more stable inference.
    
    Args:
        model: The model to track
        decay: EMA decay rate (higher = slower update, more stable)
        device: Device to store EMA weights on
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (backup current weights first)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class MultiScalePerceptualLoss(nn.Module):
    """Multi-scale perceptual loss using multiple VGG layers.
    
    Uses features from multiple depths to capture both low-level textures
    and high-level semantics, reducing artifacts from single-layer perceptual loss.
    """
    
    def __init__(self, layers: tuple = (3, 8, 15, 22), weights: tuple = (1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Extract features at different depths
        self.layers = layers
        self.weights = weights
        
        # Split VGG into blocks
        self.blocks = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev_layer:layer+1]))
            prev_layer = layer + 1
        
        # Freeze all parameters
        for block in self.blocks:
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
        
        # Register normalization buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize from [-1, 1] to ImageNet normalization."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        
        loss = 0.0
        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            # Use L1 instead of L2 for sharper results
            loss += weight * F.l1_loss(x, y)
        
        return loss


class LPIPSLoss(nn.Module):
    """LPIPS-style perceptual loss with learned weights.
    
    This is a simplified version that uses fixed weights optimized for 
    perceptual similarity. More robust than raw VGG features.
    """
    
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Use specific layers known to work well for perceptual similarity
        # These correspond to relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23]) # relu4_3
        self.slice5 = nn.Sequential(*list(vgg.children())[23:30]) # relu5_3
        
        # Learned linear weights (approximating LPIPS)
        # These are tuned weights that work well for image reconstruction
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        
        loss = 0.0
        
        # Extract and compare features at each level
        x1, y1 = self.slice1(x), self.slice1(y)
        loss += self.weights[0] * F.l1_loss(x1, y1)
        
        x2, y2 = self.slice2(x1), self.slice2(y1)
        loss += self.weights[1] * F.l1_loss(x2, y2)
        
        x3, y3 = self.slice3(x2), self.slice3(y2)
        loss += self.weights[2] * F.l1_loss(x3, y3)
        
        x4, y4 = self.slice4(x3), self.slice4(y3)
        loss += self.weights[3] * F.l1_loss(x4, y4)
        
        x5, y5 = self.slice5(x4), self.slice5(y4)
        loss += self.weights[4] * F.l1_loss(x5, y5)
        
        return loss


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial training.
    
    Classifies overlapping patches as real/fake, which helps with 
    local texture quality and reduces artifacts.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 3):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        ch_mult = 1
        for i in range(1, num_layers):
            prev_ch = base_channels * ch_mult
            ch_mult = min(2 ** i, 8)
            curr_ch = base_channels * ch_mult
            layers += [
                nn.Conv2d(prev_ch, curr_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, curr_ch), curr_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Final layers
        prev_ch = base_channels * ch_mult
        ch_mult = min(2 ** num_layers, 8)
        curr_ch = base_channels * ch_mult
        
        layers += [
            nn.Conv2d(prev_ch, curr_ch, kernel_size=4, stride=1, padding=1),
            nn.GroupNorm(min(32, curr_ch), curr_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(curr_ch, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def hinge_loss_d(real_pred, fake_pred):
    """Hinge loss for discriminator."""
    real_loss = F.relu(1.0 - real_pred).mean()
    fake_loss = F.relu(1.0 + fake_pred).mean()
    return (real_loss + fake_loss) / 2


def hinge_loss_g(fake_pred):
    """Hinge loss for generator."""
    return -fake_pred.mean()


# Keep old PerceptualLoss for backward compatibility
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained VGG16 and extract features
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        # Normalize from [-1, 1] to ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        eps = 1e-6
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        y = (y + 1) / 2
        x = (x - mean) / (std + eps)
        y = (y - mean) / (std + eps)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return torch.mean((x_features - y_features) ** 2)


def get_activation(name: str):
    """Get activation function by name."""
    if name == "swish" or name == "silu":
        return nn.SiLU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


class ResNetBlock(nn.Module):
    """ResNet block with GroupNorm for better generative model training."""
    
    def __init__(self, in_channels: int, out_channels: int = None, 
                 num_groups: int = 32, activation: str = "swish", dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.activation = get_activation(activation)
        
        # Skip connection with 1x1 conv if channels differ
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""
    
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, b, heads, hw, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # b, heads, hw, head_dim
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return out + residual


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer using interpolation + convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Encoder(nn.Module):
    """ResNet-based encoder for VAE with attention at lower resolutions."""
    
    def __init__(self, latent_dim: int = 4, in_channels: int = 3, 
                 base_channels: int = 64, channel_multipliers: tuple = (1, 2, 4, 4),
                 num_res_blocks: int = 2, attention_resolutions: tuple = (8,),
                 dropout: float = 0.0, num_groups: int = 32, input_resolution: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_resolutions = len(channel_multipliers)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        curr_channels = base_channels
        curr_resolution = input_resolution
        
        for level, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            block_list = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                block_list.append(ResNetBlock(curr_channels, out_channels, num_groups, dropout=dropout))
                curr_channels = out_channels
                
                # Add attention at specified resolutions
                if curr_resolution in attention_resolutions:
                    block_list.append(AttentionBlock(curr_channels, num_groups=num_groups))
            
            self.down_blocks.append(block_list)
            
            # Downsample (except for last level)
            if level < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(curr_channels)]))
                curr_resolution //= 2
        
        # Middle blocks
        self.mid_block1 = ResNetBlock(curr_channels, curr_channels, num_groups, dropout=dropout)
        self.mid_attn = AttentionBlock(curr_channels, num_groups=num_groups)
        self.mid_block2 = ResNetBlock(curr_channels, curr_channels, num_groups, dropout=dropout)
        
        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=min(num_groups, curr_channels), num_channels=curr_channels, eps=1e-6)
        self.conv_mu = nn.Conv2d(curr_channels, latent_dim, kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(curr_channels, latent_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        h = self.conv_in(x)
        
        # Downsampling
        for block_list in self.down_blocks:
            for block in block_list:
                h = block(h)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        
        return mu, logvar


class Decoder(nn.Module):
    """ResNet-based decoder for VAE with attention at lower resolutions."""
    
    def __init__(self, latent_dim: int = 4, out_channels: int = 3,
                 base_channels: int = 64, channel_multipliers: tuple = (1, 2, 4, 4),
                 num_res_blocks: int = 2, attention_resolutions: tuple = (8,),
                 dropout: float = 0.0, num_groups: int = 32, input_resolution: int = 64):
        super().__init__()
        
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_resolutions = len(channel_multipliers)
        
        # Compute the channel size at the bottleneck
        bottleneck_channels = base_channels * channel_multipliers[-1]
        
        # Initial convolution from latent space
        self.conv_in = nn.Conv2d(latent_dim, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        
        # Middle blocks
        self.mid_block1 = ResNetBlock(bottleneck_channels, bottleneck_channels, num_groups, dropout=dropout)
        self.mid_attn = AttentionBlock(bottleneck_channels, num_groups=num_groups)
        self.mid_block2 = ResNetBlock(bottleneck_channels, bottleneck_channels, num_groups, dropout=dropout)
        
        # Upsampling blocks (reverse order of encoder)
        self.up_blocks = nn.ModuleList()
        curr_channels = bottleneck_channels
        curr_resolution = input_resolution // (2 ** (len(channel_multipliers) - 1))  # Start at bottleneck resolution
        
        reversed_mults = list(reversed(channel_multipliers))
        for level, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            block_list = nn.ModuleList()
            
            for i in range(num_res_blocks + 1):  # +1 for the extra block in decoder
                block_list.append(ResNetBlock(curr_channels, out_ch, num_groups, dropout=dropout))
                curr_channels = out_ch
                
                # Add attention at specified resolutions
                if curr_resolution in attention_resolutions:
                    block_list.append(AttentionBlock(curr_channels, num_groups=num_groups))
            
            self.up_blocks.append(block_list)
            
            # Upsample (except for last level)
            if level < len(reversed_mults) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(curr_channels)]))
                curr_resolution *= 2
        
        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=min(num_groups, curr_channels), num_channels=curr_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(curr_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z):
        h = self.conv_in(z)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Upsampling
        for block_list in self.up_blocks:
            for block in block_list:
                h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return torch.tanh(h)


class VAE(nn.Module):
    """Variational Autoencoder with ResNet architecture for latent diffusion.
    
    Args:
        latent_dim: Number of latent channels (default 4 for SD-style VAE)
        base_channels: Base channel count (scaled by multipliers)
        channel_multipliers: Channel scaling at each resolution level
        num_res_blocks: Number of ResNet blocks per resolution
        attention_resolutions: Resolutions at which to apply self-attention
        dropout: Dropout rate for ResNet blocks
        input_resolution: Input image resolution (assumed square)
    """
    
    def __init__(self, latent_dim: int = 4, base_channels: int = 64,
                 channel_multipliers: tuple = (1, 2, 4, 4), num_res_blocks: int = 2,
                 attention_resolutions: tuple = (8,), dropout: float = 0.0,
                 input_resolution: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_resolution = input_resolution
        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            input_resolution=input_resolution
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            input_resolution=input_resolution
        )
        
        # Scaling factor for latent space (similar to Stable Diffusion)
        self.scale_factor = 0.18215

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """Encode image to latent distribution parameters."""
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent to image."""
        return self.decoder(z)
    
    def sample(self, mu, logvar):
        """Sample from the latent distribution."""
        z = self.reparameterize(mu, logvar)
        return z * self.scale_factor
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
class VAEModel(pl.LightningModule):
    """VAE training module with improved loss functions.
    
    Args:
        vae: The VAE model to train
        lr: Learning rate
        kl_weight: Weight for KL divergence loss
        perceptual_weight: Weight for perceptual loss
        ssim_weight: Weight for SSIM loss
        kl_anneal_epochs: Number of epochs to anneal KL weight
        num_log_images: Number of images to log during validation
        use_lpips: Use LPIPS-style multi-layer perceptual loss (recommended)
        use_gan: Enable adversarial training with PatchGAN discriminator
        gan_weight: Weight for adversarial loss (if use_gan=True)
        disc_start_epoch: Epoch to start discriminator training
        use_l1: Use L1 loss instead of MSE for sharper results
        use_ema: Enable Exponential Moving Average for VAE weights
        ema_decay: EMA decay rate (higher = slower update)
    """
    
    def __init__(self, vae: VAE, lr: float = 1e-4, kl_weight: float = 0.00001, 
                 perceptual_weight: float = 0.1, ssim_weight: float = 0.0,
                 kl_anneal_epochs: int = 10, num_log_images: int = 8,
                 use_lpips: bool = True, use_gan: bool = False, 
                 gan_weight: float = 0.1, disc_start_epoch: int = 5,
                 use_l1: bool = True, use_ema: bool = True, ema_decay: float = 0.999):
        super().__init__()
        self.automatic_optimization = not use_gan  # Manual optimization for GAN
        
        self.vae = vae
        self.lr = lr
        self.kl_weight_max = kl_weight
        self.kl_anneal_epochs = kl_anneal_epochs
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.num_log_images = num_log_images
        self.use_gan = use_gan
        self.gan_weight = gan_weight
        self.disc_start_epoch = disc_start_epoch
        self.use_l1 = use_l1
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = None  # Will be initialized in on_fit_start
        
        # Reconstruction loss
        if use_l1:
            self.recon_criterion = nn.L1Loss()
        else:
            self.recon_criterion = nn.MSELoss()
        
        # Perceptual loss - use improved LPIPS-style by default
        if use_lpips:
            self.perceptual_loss = LPIPSLoss()
        else:
            self.perceptual_loss = PerceptualLoss()
        
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)
        
        # Discriminator for GAN training
        if use_gan:
            self.discriminator = PatchDiscriminator(in_channels=3, base_channels=64, num_layers=3)
        
    def on_fit_start(self):
        """Initialize EMA when training starts."""
        if self.use_ema:
            self.ema = EMA(self.vae, decay=self.ema_decay, device=self.device)
            print(f"EMA initialized with decay={self.ema_decay}")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training step."""
        if self.use_ema and self.ema is not None:
            self.ema.update(self.vae)
    
    def on_validation_epoch_start(self):
        """Apply EMA weights for validation."""
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow(self.vae)
    
    def on_validation_epoch_end(self):
        """Restore original weights after validation."""
        if self.use_ema and self.ema is not None:
            self.ema.restore(self.vae)
    
    def on_save_checkpoint(self, checkpoint):
        """Save EMA state in checkpoint."""
        if self.use_ema and self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load EMA state from checkpoint."""
        if self.use_ema and 'ema_state_dict' in checkpoint:
            if self.ema is None:
                self.ema = EMA(self.vae, decay=self.ema_decay)
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            print("EMA state loaded from checkpoint")
    
    def get_ema_vae(self):
        """Get a copy of VAE with EMA weights applied (for inference/export)."""
        if not self.use_ema or self.ema is None:
            return self.vae
        
        # Create a copy and apply EMA weights
        ema_vae = copy.deepcopy(self.vae)
        for name, param in ema_vae.named_parameters():
            if name in self.ema.shadow:
                param.data = self.ema.shadow[name].clone()
        return ema_vae
        
    def get_current_kl_weight(self):
        # Linear warmup for KL weight
        if self.kl_anneal_epochs == 0:
            return self.kl_weight_max
        progress = min(self.current_epoch / self.kl_anneal_epochs, 1.0)
        return progress * self.kl_weight_max
    
    def get_current_gan_weight(self):
        # Only start GAN loss after disc_start_epoch
        if self.current_epoch < self.disc_start_epoch:
            return 0.0
        return self.gan_weight

    def compute_vae_loss(self, x, recon_x, mu, logvar):
        """Compute VAE losses."""
        # Reconstruction loss (L1 or MSE)
        recon_loss = self.recon_criterion(recon_x, x)
        
        # Perceptual loss
        perceptual = self.perceptual_loss(recon_x, x)
        
        # Combined reconstruction
        combined_recon = recon_loss + self.perceptual_weight * perceptual
        
        # SSIM loss
        ssim_loss = torch.tensor(0.0, device=x.device)
        if self.ssim_weight > 0:
            ssim_value = self.ssim_metric(recon_x, x)
            ssim_loss = 1 - ssim_value
            combined_recon = combined_recon + self.ssim_weight * ssim_loss
        
        # KL divergence
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        current_kl_weight = self.get_current_kl_weight()
        total_loss = combined_recon + current_kl_weight * kld_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'perceptual': perceptual,
            'ssim': ssim_loss,
            'kld': kld_loss,
            'kl_weight': current_kl_weight
        }

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        
        if self.use_gan:
            return self._training_step_gan(x, batch_idx)
        else:
            return self._training_step_vae(x, batch_idx)
    
    def _training_step_vae(self, x, batch_idx):
        """Standard VAE training step."""
        recon_x, mu, logvar = self.vae(x)
        losses = self.compute_vae_loss(x, recon_x, mu, logvar)
        
        self.log("train_loss", losses['total'], prog_bar=True)
        self.log("train_recon_loss", losses['recon'], prog_bar=False)
        self.log("train_perceptual_loss", losses['perceptual'], prog_bar=True)
        self.log("train_kld_loss", losses['kld'], prog_bar=True)
        self.log("kl_weight", losses['kl_weight'], prog_bar=False)
        if self.ssim_weight > 0:
            self.log("train_ssim_loss", losses['ssim'], prog_bar=False)
        
        return losses['total']
    
    def _training_step_gan(self, x, batch_idx):
        """GAN training step with alternating generator/discriminator updates."""
        opt_vae, opt_disc = self.optimizers()
        
        # Generate reconstruction
        recon_x, mu, logvar = self.vae(x)
        
        # ================== Train Discriminator ==================
        opt_disc.zero_grad()
        
        if self.current_epoch >= self.disc_start_epoch:
            # Real images
            real_pred = self.discriminator(x)
            # Fake images (detached)
            fake_pred = self.discriminator(recon_x.detach())
            
            d_loss = hinge_loss_d(real_pred, fake_pred)
            self.manual_backward(d_loss)
            opt_disc.step()
            self.log("train_d_loss", d_loss, prog_bar=True)
        else:
            d_loss = torch.tensor(0.0, device=x.device)
        
        # ================== Train VAE (Generator) ==================
        opt_vae.zero_grad()
        
        # VAE losses
        losses = self.compute_vae_loss(x, recon_x, mu, logvar)
        g_loss = losses['total']
        
        # Add adversarial loss
        current_gan_weight = self.get_current_gan_weight()
        if current_gan_weight > 0:
            fake_pred = self.discriminator(recon_x)
            adv_loss = hinge_loss_g(fake_pred)
            g_loss = g_loss + current_gan_weight * adv_loss
            self.log("train_adv_loss", adv_loss, prog_bar=True)
        
        self.manual_backward(g_loss)
        opt_vae.step()
        
        self.log("train_loss", g_loss, prog_bar=True)
        self.log("train_recon_loss", losses['recon'], prog_bar=False)
        self.log("train_perceptual_loss", losses['perceptual'], prog_bar=True)
        self.log("train_kld_loss", losses['kld'], prog_bar=True)
        self.log("kl_weight", losses['kl_weight'], prog_bar=False)
        self.log("gan_weight", current_gan_weight, prog_bar=False)
        
        return g_loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        recon_x, mu, logvar = self.vae(x)
        
        losses = self.compute_vae_loss(x, recon_x, mu, logvar)
        
        self.log("val_loss", losses['total'], prog_bar=True, sync_dist=True)
        self.log("val_recon_loss", losses['recon'], prog_bar=False, sync_dist=True)
        self.log("val_perceptual_loss", losses['perceptual'], prog_bar=True, sync_dist=True)
        self.log("val_kld_loss", losses['kld'], prog_bar=True, sync_dist=True)
        if self.ssim_weight > 0:
            self.log("val_ssim_loss", losses['ssim'], prog_bar=False, sync_dist=True)
        
        # Log images only for the first batch
        if batch_idx == 0:
            num_images = min(self.num_log_images, x.shape[0])
            original = x[:num_images]
            reconstructed = recon_x[:num_images]
            
            # Denormalize images from [-1, 1] to [0, 1]
            original = (original + 1) / 2
            reconstructed = (reconstructed + 1) / 2
            
            # Clamp to valid range
            original = torch.clamp(original, 0, 1)
            reconstructed = torch.clamp(reconstructed, 0, 1)
            
            # Create grid of original and reconstructed images
            comparison = torch.stack([original, reconstructed], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(comparison, nrow=num_images, normalize=False)
            
            # Log to tensorboard
            self.logger.experiment.add_image("val_reconstruction", grid, self.current_epoch)
        
        return losses['total']

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        recon_x, mu, logvar = self.vae(x)
        
        losses = self.compute_vae_loss(x, recon_x, mu, logvar)
        
        self.log("test_loss", losses['total'])
        self.log("test_recon_loss", losses['recon'])
        self.log("test_perceptual_loss", losses['perceptual'])
        self.log("test_kld_loss", losses['kld'])
        if self.ssim_weight > 0:
            self.log("test_ssim_loss", losses['ssim'])
        
        return losses['total']

    def configure_optimizers(self):
        # VAE optimizer
        vae_optimizer = torch.optim.AdamW(
            self.vae.parameters(), 
            lr=self.lr, 
            betas=(0.5, 0.9),  # Lower beta1 for better GAN training
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            vae_optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        if self.use_gan:
            # Discriminator optimizer
            disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.lr * 0.5,  # Slightly lower LR for discriminator
                betas=(0.5, 0.9),
                weight_decay=0.01
            )
            return [vae_optimizer, disc_optimizer], [vae_scheduler]
        else:
            return {
                "optimizer": vae_optimizer,
                "lr_scheduler": {
                    "scheduler": vae_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

if __name__ == "__main__":
    # Test the improved VAE at 64x64
    print("=" * 50)
    print("Testing VAE at 64x64 resolution")
    print("=" * 50)
    vae_64 = VAE(
        latent_dim=4,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_resolutions=(8,),
        dropout=0.0,
        input_resolution=64
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in vae_64.parameters())
    trainable_params = sum(p.numel() for p in vae_64.parameters() if p.requires_grad)
    
    imgs = torch.randn(4, 3, 64, 64)
    recon_imgs, mu, logvar = vae_64(imgs)
    print("Input images shape:", imgs.shape)
    print("Reconstructed images shape:", recon_imgs.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test encoding/decoding separately
    mu, logvar = vae_64.encode(imgs)
    z = vae_64.sample(mu, logvar)
    decoded = vae_64.decode(z / vae_64.scale_factor)
    print(f"Latent z shape: {z.shape}")
    print(f"Decoded shape: {decoded.shape}")
    
    # Test at 128x128
    print("\n" + "=" * 50)
    print("Testing VAE at 128x128 resolution")
    print("=" * 50)
    vae_128 = VAE(
        latent_dim=4,
        base_channels=128,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0,
        input_resolution=128
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in vae_128.parameters())
    trainable_params = sum(p.numel() for p in vae_128.parameters() if p.requires_grad)
    
    imgs = torch.randn(4, 3, 128, 128)
    recon_imgs, mu, logvar = vae_128(imgs)
    print("Input images shape:", imgs.shape)
    print("Reconstructed images shape:", recon_imgs.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test encoding/decoding separately
    mu, logvar = vae_128.encode(imgs)
    z = vae_128.sample(mu, logvar)
    decoded = vae_128.decode(z / vae_128.scale_factor)
    print(f"Latent z shape: {z.shape}")
    print(f"Decoded shape: {decoded.shape}")


