"""
Generate sample images from the trained Latent Diffusion Model.

This script generates a grid of images for demonstration purposes,
perfect for showcasing the model's capabilities in posts or presentations.

Usage:
    poetry run python generate_images.py --output showcase.png --grid 3x3
"""

import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from diffusers.vae.model import VAE
from diffusers.latent_diffusion.latent_diffusion import LatentDiffusionModel
from diffusers.basic_model.schedulers import offset_cosine_diffusion_scheduler

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ============================================================================
# Configuration
# ============================================================================

IMAGE_SIZE = 128
LATENT_CHANNELS = 4
NUM_CLASSES = 3
CLASS_NAMES = ["cat", "dog", "wild"]


# ============================================================================
# Load Models
# ============================================================================

def load_models(vae_path: str, ldm_path: str, device='cpu'):
    """Load pretrained VAE and Latent Diffusion Model from safetensors."""
    print("📦 Loading models...")
    vae = VAE.from_pretrained(vae_path)
    ldm = LatentDiffusionModel.from_pretrained(vae, ldm_path)
    print("✓ Models loaded successfully")
    return ldm.to(device).eval()


# ============================================================================
# Image Generation
# ============================================================================

@torch.no_grad()
def generate_images(model, class_id: int, num_images: int, num_steps: int, 
                   cfg_scale: float, device='cpu', save_intermediates=False):
    """Generate images using DDIM sampling.
    
    Args:
        model: The latent diffusion model
        class_id: Class label for generation
        num_images: Number of images to generate
        num_steps: Number of DDIM steps
        cfg_scale: Classifier-free guidance scale
        device: Device to run on
        save_intermediates: If True, return list of intermediate images for GIF creation
    
    Returns:
        images: Final generated images
        intermediate_images: List of images at each step (only if save_intermediates=True)
    """
    model.eval()
    intermediate_images = [] if save_intermediates else None
    
    # Start with pure noise
    latent_shape = (LATENT_CHANNELS, 16, 16)
    z = torch.randn(num_images, *latent_shape, device=device)
    
    # Prepare class labels for CFG
    class_labels = torch.full((num_images,), class_id, device=device, dtype=torch.long)
    null_labels = torch.full((num_images,), NUM_CLASSES, device=device, dtype=torch.long)
    
    # DDIM sampling
    num_training_steps = 1000
    step_ratio = num_training_steps / num_steps
    timesteps = [int(i * step_ratio) for i in range(num_steps)]
    timesteps.reverse()
    
    # Get all diffusion rates
    all_alphas = []
    all_betas = []
    for t in range(num_training_steps):
        alpha, beta = offset_cosine_diffusion_scheduler(t / num_training_steps)
        all_alphas.append(alpha)
        all_betas.append(beta)
    
    all_alphas = torch.tensor(all_alphas, device=device)
    all_betas = torch.tensor(all_betas, device=device)
    
    # Determine which steps to save for GIF (save every N steps to avoid too many frames)
    save_interval = max(1, len(timesteps) // 20) if save_intermediates else None
    
    for i in tqdm(range(len(timesteps)), desc=f"Generating {CLASS_NAMES[class_id]} images"):
        t = timesteps[i]
        t_norm = t / num_training_steps
        t_input = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        alpha_t = all_alphas[t].view(1, 1, 1, 1)
        beta_t = all_betas[t].view(1, 1, 1, 1)
        
        # CFG: predict with and without conditioning
        if cfg_scale > 1.0:
            # Concatenate for parallel inference
            z_in = torch.cat([z, z], dim=0)
            t_in = torch.cat([t_input, t_input], dim=0)
            labels_in = torch.cat([class_labels, null_labels], dim=0)
            
            noise_pred = model.unet(z_in, t_in / num_training_steps, labels_in)
            
            # Split and apply CFG
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model.unet(z, t_input / num_training_steps, class_labels)
        
        # DDIM: predict clean latent
        z0_pred = (z - alpha_t * noise_pred) / beta_t
        z0_pred = torch.clamp(z0_pred, -3, 3)
        
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_prev = all_alphas[t_prev].view(1, 1, 1, 1)
            beta_prev = all_betas[t_prev].view(1, 1, 1, 1)
            z = beta_prev * z0_pred + alpha_prev * noise_pred
        else:
            z = z0_pred
        
        # Save intermediate images for GIF
        if save_intermediates and (i % save_interval == 0 or i == len(timesteps) - 1):
            scale_factor = 0.38
            intermediate = model.vae.decode(z / scale_factor)
            intermediate = (intermediate + 1) / 2
            intermediate = torch.clamp(intermediate, 0, 1)
            intermediate_images.append(intermediate.cpu())
    
    # Decode to image space
    scale_factor = 0.38
    images = model.vae.decode(z / scale_factor)
    
    # Denormalize and clamp
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    if save_intermediates:
        return images, intermediate_images
    return images


# ============================================================================
# Visualization
# ============================================================================

def create_image_grid(images, grid_size, class_labels=None, save_path=None):
    """Create a grid of images with optional class labels."""
    rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            img = images[idx].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
            
            if class_labels is not None:
                class_name = CLASS_NAMES[class_labels[idx]]
                ax.set_title(class_name.capitalize(), fontsize=10, pad=5)
        else:
            ax.axis('off')
    
    plt.tight_layout(pad=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"✓ Image grid saved to: {save_path}")
    
    plt.close()


def create_class_showcase(images_per_class, save_path=None):
    """Create a showcase grid with samples from each class."""
    num_per_class = len(images_per_class[0])
    total_images = NUM_CLASSES * num_per_class
    
    fig, axes = plt.subplots(NUM_CLASSES, num_per_class, 
                            figsize=(num_per_class * 2, NUM_CLASSES * 2))
    
    if NUM_CLASSES == 1:
        axes = axes.reshape(1, -1)
    if num_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for class_idx in range(NUM_CLASSES):
        for img_idx in range(num_per_class):
            ax = axes[class_idx, img_idx]
            img = images_per_class[class_idx][img_idx].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
            
            # Add class label on the first image of each row
            if img_idx == 0:
                ax.set_ylabel(CLASS_NAMES[class_idx].capitalize(), 
                            fontsize=12, rotation=0, labelpad=40, va='center')
    
    plt.tight_layout(pad=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"✓ Class showcase saved to: {save_path}")
    
    plt.close()


def create_generation_gif(intermediate_images, grid_size, class_labels=None, 
                         save_path=None, duration=100, loop=0, final_frame_duration=2000):
    """Create a GIF showing the generation process from noise to final image.
    
    Args:
        intermediate_images: List of image tensors at different generation steps
        grid_size: Tuple of (rows, cols) for grid layout
        class_labels: Optional list of class labels for each image
        save_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
        final_frame_duration: Duration of the final frame in milliseconds (default 2000ms = 2s)
    """
    rows, cols = grid_size
    frames = []
    
    print(f"\n🎬 Creating GIF with {len(intermediate_images)} frames...")
    
    for step_idx, images in enumerate(tqdm(intermediate_images, desc="Processing frames")):
        # Create figure for this step
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
                
                if class_labels is not None:
                    class_name = CLASS_NAMES[class_labels[idx]]
                    ax.set_title(class_name.capitalize(), fontsize=10, pad=5)
            else:
                ax.axis('off')
        
        plt.tight_layout(pad=0.5)
        
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        frame = frame[:, :, :3]
        frames.append(Image.fromarray(frame))
        plt.close(fig)
    
    # Repeat the final frame to make it stay longer
    # Calculate how many times to repeat based on final_frame_duration
    num_final_repeats = max(1, final_frame_duration // duration)
    for _ in range(num_final_repeats - 1):
        frames.append(frames[-1].copy())
    
    # Save as GIF
    if save_path:
        # All frames now use the same duration
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=duration, loop=loop, optimize=False)
        print(f"✓ GIF saved to: {save_path}")
        print(f"  - Total frames: {len(frames)}")
        print(f"  - Duration per frame: {duration}ms")
        print(f"  - Final frame repeats: {num_final_repeats}x")
        print(f"  - Final frame total duration: {num_final_repeats * duration}ms")
        total_time = len(frames) * duration
        print(f"  - Total animation time: {total_time / 1000:.1f}s")


def create_class_showcase_gif(images_per_class_per_step, save_path=None, 
                              duration=100, loop=0, final_frame_duration=2000):
    """Create a GIF showing the generation process for a class showcase.
    
    Args:
        images_per_class_per_step: List of [images_per_class] for each step
        save_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
        final_frame_duration: Duration of the final frame in milliseconds (default 2000ms = 2s)
    """
    num_steps = len(images_per_class_per_step)
    num_per_class = len(images_per_class_per_step[0][0])
    frames = []
    
    print(f"\n🎬 Creating showcase GIF with {num_steps} frames...")
    
    for step_idx in tqdm(range(num_steps), desc="Processing frames"):
        images_per_class = images_per_class_per_step[step_idx]
        
        # Create figure for this step
        fig, axes = plt.subplots(NUM_CLASSES, num_per_class,
                                figsize=(num_per_class * 2, NUM_CLASSES * 2))
        
        if NUM_CLASSES == 1:
            axes = axes.reshape(1, -1)
        if num_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx in range(NUM_CLASSES):
            for img_idx in range(num_per_class):
                ax = axes[class_idx, img_idx]
                img = images_per_class[class_idx][img_idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
                
                # Add class label on the first image of each row
                if img_idx == 0:
                    ax.set_ylabel(CLASS_NAMES[class_idx].capitalize(),
                                fontsize=12, rotation=0, labelpad=40, va='center')
        
        plt.tight_layout(pad=0.3)
        
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        frame = frame[:, :, :3]
        frames.append(Image.fromarray(frame))
        plt.close(fig)
    
    # Repeat the final frame to make it stay longer
    # Calculate how many times to repeat based on final_frame_duration
    num_final_repeats = max(1, final_frame_duration // duration)
    for _ in range(num_final_repeats - 1):
        frames.append(frames[-1].copy())
    
    # Save as GIF
    if save_path:
        # All frames now use the same duration
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=duration, loop=loop, optimize=False)
        print(f"✓ GIF saved to: {save_path}")
        print(f"  - Total frames: {len(frames)}")
        print(f"  - Duration per frame: {duration}ms")
        print(f"  - Final frame repeats: {num_final_repeats}x")
        print(f"  - Final frame total duration: {num_final_repeats * duration}ms")
        total_time = len(frames) * duration
        print(f"  - Total animation time: {total_time / 1000:.1f}s")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate images from Latent Diffusion Model")
    parser.add_argument("--vae_checkpoint", type=str,
                        default="data/latent_diffusion_model/vae/vae-ema-final.safetensors",
                        help="Path to VAE safetensors checkpoint")
    parser.add_argument("--ldm_checkpoint", type=str,
                        default="data/latent_diffusion_model/unet/ldm-final.safetensors",
                        help="Path to LDM safetensors checkpoint")
    parser.add_argument("--output", type=str, default="generated_images.png",
                        help="Output image path")
    parser.add_argument("--grid", type=str, default="3x3",
                        help="Grid size (e.g., 3x3, 4x4)")
    parser.add_argument("--mode", type=str, default="mixed", 
                        choices=["mixed", "showcase", "single_class"],
                        help="Generation mode: mixed (random classes), showcase (organized by class), single_class")
    parser.add_argument("--class_id", type=int, default=0, choices=[0, 1, 2],
                        help="Class to generate (0=cat, 1=dog, 2=wild) for single_class mode")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of DDIM steps")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save generation process as GIF")
    parser.add_argument("--gif_duration", type=int, default=100,
                        help="Duration of each GIF frame in milliseconds")
    parser.add_argument("--final_frame_duration", type=int, default=2000,
                        help="Duration of the final GIF frame in milliseconds (default: 2000ms = 2s)")
    
    args = parser.parse_args()
    
    # Parse grid size
    if 'x' in args.grid:
        rows, cols = map(int, args.grid.split('x'))
    else:
        # Single number: for showcase mode, it's images per class (cols)
        # For other modes, treat it as a square grid
        num = int(args.grid)
        if args.mode == "showcase":
            rows = NUM_CLASSES
            cols = num
        else:
            rows = cols = num
    
    total_images = rows * cols
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print(f"{'='*80}")
    print(f"Generating {total_images} images ({args.grid} grid)")
    print(f"Mode: {args.mode}")
    print(f"DDIM steps: {args.num_steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Load model
    model = load_models(args.vae_checkpoint, args.ldm_checkpoint, args.device)
    
    if args.mode == "single_class":
        # Generate all images from one class
        print(f"\n🎨 Generating {total_images} {CLASS_NAMES[args.class_id]} images...")
        
        if args.save_gif:
            images, intermediate_images = generate_images(
                model, args.class_id, total_images, 
                args.num_steps, args.cfg_scale, args.device, save_intermediates=True
            )
            class_labels = [args.class_id] * total_images
            create_image_grid(images, (rows, cols), class_labels, args.output)
            
            # Create GIF
            gif_path = Path(args.output).with_suffix('.gif')
            create_generation_gif(intermediate_images, (rows, cols), class_labels, 
                                gif_path, duration=args.gif_duration,
                                final_frame_duration=args.final_frame_duration)
        else:
            images = generate_images(model, args.class_id, total_images, 
                                    args.num_steps, args.cfg_scale, args.device)
            class_labels = [args.class_id] * total_images
            create_image_grid(images, (rows, cols), class_labels, args.output)
        
    elif args.mode == "showcase":
        # Generate organized showcase by class
        images_per_row = cols
        print(f"\n🎨 Generating showcase with {images_per_row} images per class...")
        
        if args.save_gif:
            images_per_class = []
            all_intermediates = []  # List of intermediates for each class
            
            for class_id in range(NUM_CLASSES):
                print(f"\nGenerating {CLASS_NAMES[class_id]} samples...")
                images, intermediates = generate_images(
                    model, class_id, images_per_row,
                    args.num_steps, args.cfg_scale, args.device, save_intermediates=True
                )
                images_per_class.append(images)
                all_intermediates.append(intermediates)
            
            create_class_showcase(images_per_class, args.output)
            
            # Reorganize intermediates: from [class][step][images] to [step][class][images]
            num_steps = len(all_intermediates[0])
            images_per_class_per_step = []
            for step_idx in range(num_steps):
                step_images = [all_intermediates[class_id][step_idx] 
                             for class_id in range(NUM_CLASSES)]
                images_per_class_per_step.append(step_images)
            
            # Create GIF
            gif_path = Path(args.output).with_suffix('.gif')
            create_class_showcase_gif(images_per_class_per_step, gif_path, 
                                     duration=args.gif_duration,
                                     final_frame_duration=args.final_frame_duration)
        else:
            images_per_class = []
            for class_id in range(NUM_CLASSES):
                print(f"\nGenerating {CLASS_NAMES[class_id]} samples...")
                images = generate_images(model, class_id, images_per_row,
                                       args.num_steps, args.cfg_scale, args.device)
                images_per_class.append(images)
            
            create_class_showcase(images_per_class, args.output)
        
    else:  # mixed mode
        # Generate mixed images from all classes
        print(f"\n🎨 Generating {total_images} mixed images...")
        
        # Distribute classes evenly
        images_per_class = total_images // NUM_CLASSES
        remainder = total_images % NUM_CLASSES
        
        all_images = []
        all_labels = []
        all_intermediates_list = [] if args.save_gif else None
        
        for class_id in range(NUM_CLASSES):
            num_images = images_per_class + (1 if class_id < remainder else 0)
            if num_images > 0:
                print(f"\nGenerating {num_images} {CLASS_NAMES[class_id]} samples...")
                
                if args.save_gif:
                    images, intermediates = generate_images(
                        model, class_id, num_images,
                        args.num_steps, args.cfg_scale, args.device, save_intermediates=True
                    )
                    all_intermediates_list.append((intermediates, [class_id] * num_images))
                else:
                    images = generate_images(model, class_id, num_images,
                                           args.num_steps, args.cfg_scale, args.device)
                
                all_images.append(images)
                all_labels.extend([class_id] * num_images)
        
        all_images = torch.cat(all_images, dim=0)
        
        # Shuffle for mixed appearance
        indices = torch.randperm(len(all_images))
        all_images = all_images[indices]
        all_labels = [all_labels[i] for i in indices]
        
        create_image_grid(all_images, (rows, cols), all_labels, args.output)
        
        # Create GIF if requested
        if args.save_gif:
            # Combine intermediates from all classes and apply same shuffle
            num_steps = len(all_intermediates_list[0][0])
            combined_intermediates = []
            
            for step_idx in range(num_steps):
                step_images = []
                step_labels = []
                
                for intermediates, labels in all_intermediates_list:
                    step_images.append(intermediates[step_idx])
                    step_labels.extend(labels)
                
                step_images = torch.cat(step_images, dim=0)
                step_images = step_images[indices]
                step_labels = [step_labels[i] for i in indices]
                
                combined_intermediates.append(step_images)
            
            gif_path = Path(args.output).with_suffix('.gif')
            create_generation_gif(combined_intermediates, (rows, cols), all_labels,
                                gif_path, duration=args.gif_duration,
                                final_frame_duration=args.final_frame_duration)
    
    print(f"\n🎉 Generation complete!")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
