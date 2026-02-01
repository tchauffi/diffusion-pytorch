"""
Evaluate Latent Diffusion Models using FID (Fréchet Inception Distance).

Supports both architectures:
- Noise prediction (LatentDiffusionModel) with DDIM sampling
- Flow matching (FlowLatentDiffusionModel) with Euler/Heun ODE integration

This script:
1. Loads a trained latent diffusion model (either architecture)
2. Generates samples for each class
3. Compares generated images to real images using FID metric
4. Reports FID scores per class and overall

Usage:
    # For flow matching model (default):
    poetry run python evaluate_fid.py --model_type flow --sampler euler
    
    # For noise prediction model:
    poetry run python evaluate_fid.py --model_type noise --sampler ddim
"""

import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path

from diffusers.vae.model import VAE
from diffusers.latent_diffusion.latent_diffusion import LatentDiffusionModel
from diffusers.latent_diffusion.flow_latent_diff import FlowLatentDiffusionModel
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

# FID calculation settings
SAMPLES_PER_CLASS = 500  # Number of samples to generate per class
REAL_SAMPLES_PER_CLASS = 500  # Number of real samples to use per class
BATCH_SIZE = 32
NUM_INFERENCE_STEPS = 50  # Number of sampling steps
CFG_SCALE = 3.0  # Classifier-free guidance scale
MODEL_TYPE = "flow"  # "flow" or "noise"
SAMPLER = "euler"  # "euler", "heun" (for flow), "ddim" (for noise)


# ============================================================================
# Load Models
# ============================================================================

def load_models(vae_path: str, ldm_path: str, model_type: str = "flow", device='cpu'):
    """Load pretrained VAE and Latent Diffusion Model from safetensors.
    
    Args:
        vae_path: Path to VAE checkpoint
        ldm_path: Path to LDM/UNet checkpoint
        model_type: "flow" for FlowLatentDiffusionModel, "noise" for LatentDiffusionModel
        device: Device to load model on
    """
    # Load VAE using from_pretrained
    vae = VAE.from_pretrained(vae_path)
    
    # Load appropriate model type
    if model_type == "flow":
        ldm = FlowLatentDiffusionModel.from_pretrained(vae, ldm_path)
    elif model_type == "noise":
        ldm = LatentDiffusionModel.from_pretrained(vae, ldm_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'flow' or 'noise'.")
    
    return ldm.to(device).eval()


# ============================================================================
# Data Loading
# ============================================================================

def get_real_images(class_id: int, num_samples: int, device='cpu'):
    """Load real images from dataset for a specific class."""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    
    # Load dataset
    dataset = load_dataset(
        "huggan/AFHQ",
        split="train",
        streaming=False,
        trust_remote_code=True
    )
    
    # Filter by class (label is an integer: 0=cat, 1=dog, 2=wild)
    class_name = CLASS_NAMES[class_id]
    filtered_indices = [i for i, item in enumerate(dataset) if item['label'] == class_id]
    
    # Limit to num_samples
    if len(filtered_indices) > num_samples:
        filtered_indices = np.random.choice(filtered_indices, num_samples, replace=False).tolist()
    
    # Create subset
    subset = Subset(dataset, filtered_indices)
    
    # Load images
    images = []
    for idx in tqdm(range(len(subset)), desc=f"Loading real {class_name} images"):
        item = dataset[subset.indices[idx]]
        img = transform(item['image'])
        images.append(img)
    
    return torch.stack(images).to(device)


# ============================================================================
# Image Generation
# ============================================================================

@torch.no_grad()
def generate_samples(model, class_id: int, num_samples: int, 
                     num_steps: int, cfg_scale: float, batch_size: int, device='cpu',
                     sampler: str = "euler", model_type: str = "flow"):
    """Generate samples using the appropriate sampling method.
    
    Args:
        model: LatentDiffusionModel or FlowLatentDiffusionModel
        class_id: Class index for conditional generation
        num_samples: Number of samples to generate
        num_steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        batch_size: Batch size for generation
        device: Device to use
        sampler: "euler", "heun" (for flow) or "ddim" (for noise)
        model_type: "flow" or "noise"
    """
    model.eval()
    all_images = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Generating {CLASS_NAMES[class_id]} images"):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # Start with pure noise
        latent_shape = (LATENT_CHANNELS, 16, 16)
        z = torch.randn(current_batch_size, *latent_shape, device=device)
        
        # Prepare class labels for CFG
        class_labels = torch.full((current_batch_size,), class_id, device=device, dtype=torch.long)
        null_labels = torch.full((current_batch_size,), NUM_CLASSES, device=device, dtype=torch.long)
        
        if model_type == "flow":
            # Flow matching: Euler or Heun ODE integration
            z = _flow_matching_sample(model, z, num_steps, cfg_scale, 
                                       class_labels, null_labels, device, sampler)
        else:
            # Noise prediction: DDIM sampling
            z = _ddim_sample(model, z, num_steps, cfg_scale,
                            class_labels, null_labels, device)
        
        # Decode to image space
        scale_factor = 0.38
        images = model.vae.decode(z / scale_factor)
        
        # Denormalize and clamp
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        all_images.append(images)
    
    return torch.cat(all_images, dim=0)


def _flow_matching_sample(model, z, num_steps, cfg_scale, class_labels, null_labels, device, sampler):
    """Flow matching sampling with Euler or Heun integration."""
    current_batch_size = z.shape[0]
    use_cfg = cfg_scale > 1.0
    
    # Time steps from t=1 (noise) to t=0 (data)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    # Helper function to compute velocity with CFG (batched for efficiency)
    def get_velocity(z_t, t_scalar):
        t_batch = torch.full((current_batch_size,), t_scalar, device=device, dtype=torch.float32)
        
        if use_cfg:
            # Batch conditional and unconditional together for single forward pass
            z_in = torch.cat([z_t, z_t], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            labels_in = torch.cat([class_labels, null_labels], dim=0)
            
            v_both = model.unet(z_in, t_in, labels_in)
            v_cond, v_uncond = v_both.chunk(2, dim=0)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            return model.unet(z_t, t_batch, class_labels)
    
    if sampler == "euler":
        for i in range(len(timesteps) - 1):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t
            
            v = get_velocity(z, t)
            z = z + dt * v
            
    elif sampler == "heun":
        for i in range(len(timesteps) - 1):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t
            
            v1 = get_velocity(z, t)
            z_pred = z + dt * v1
            
            v2 = get_velocity(z_pred, t_next)
            z = z + dt * 0.5 * (v1 + v2)
    else:
        raise ValueError(f"Unknown flow sampler: {sampler}. Choose 'euler' or 'heun'.")
    
    return z


def _ddim_sample(model, z, num_steps, cfg_scale, class_labels, null_labels, device):
    """DDIM sampling for noise prediction models."""
    current_batch_size = z.shape[0]
    num_training_steps = 1000
    
    # Create timestep schedule
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
    
    for i, t in enumerate(timesteps):
        t_input = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
        
        alpha_t = all_alphas[t].view(1, 1, 1, 1)
        beta_t = all_betas[t].view(1, 1, 1, 1)
        
        # CFG: predict with and without conditioning
        if cfg_scale > 1.0:
            z_in = torch.cat([z, z], dim=0)
            t_in = torch.cat([t_input, t_input], dim=0)
            labels_in = torch.cat([class_labels, null_labels], dim=0)
            
            noise_pred = model.unet(z_in, t_in / num_training_steps, labels_in)
            
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
    
    return z


# ============================================================================
# FID Calculation
# ============================================================================

def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor, device='cpu'):
    """Compute FID between real and generated images."""
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Process in batches to avoid OOM
    batch_size = 32
    
    # Add real images
    for i in tqdm(range(0, len(real_images), batch_size), desc="Processing real images"):
        batch = real_images[i:i+batch_size]
        # Convert to uint8 format expected by FID (0-255 range)
        batch_uint8 = (batch * 255).to(torch.uint8)
        fid.update(batch_uint8, real=True)
    
    # Add fake images
    for i in tqdm(range(0, len(fake_images), batch_size), desc="Processing fake images"):
        batch = fake_images[i:i+batch_size]
        # Convert to uint8 format expected by FID (0-255 range)
        batch_uint8 = (batch * 255).to(torch.uint8)
        fid.update(batch_uint8, real=False)
    
    return fid.compute().item()


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Latent Diffusion Model using FID")
    parser.add_argument("--vae_checkpoint", type=str, 
                        default="data/latent_diffusion_model/vae/vae-ema-final.safetensors",
                        help="Path to VAE safetensors checkpoint")
    parser.add_argument("--ldm_checkpoint", type=str,
                        default="data/flow_matching_model/unet/ldm-final.safetensors",
                        help="Path to LDM safetensors checkpoint")
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE, choices=["flow", "noise"],
                        help="Model type: 'flow' (velocity prediction) or 'noise' (noise prediction)")
    parser.add_argument("--samples_per_class", type=int, default=SAMPLES_PER_CLASS,
                        help="Number of samples to generate per class")
    parser.add_argument("--real_samples", type=int, default=REAL_SAMPLES_PER_CLASS,
                        help="Number of real samples to use per class")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for generation")
    parser.add_argument("--num_steps", type=int, default=NUM_INFERENCE_STEPS,
                        help="Number of sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=CFG_SCALE,
                        help="Classifier-free guidance scale")
    parser.add_argument("--sampler", type=str, default=SAMPLER, choices=["euler", "heun", "ddim"],
                        help="Sampler: 'euler'/'heun' (for flow) or 'ddim' (for noise)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of evaluation runs for computing std (default: 1, no std)")
    parser.add_argument("--output", type=str, default="fid_results.txt",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Validate sampler for model type
    if args.model_type == "flow" and args.sampler == "ddim":
        print("Warning: DDIM sampler is for noise prediction models. Switching to 'euler'.")
        args.sampler = "euler"
    elif args.model_type == "noise" and args.sampler in ["euler", "heun"]:
        print(f"Warning: {args.sampler} sampler is for flow models. Switching to 'ddim'.")
        args.sampler = "ddim"
    
    print(f"Using device: {args.device}")
    print(f"Model type: {args.model_type}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"LDM checkpoint: {args.ldm_checkpoint}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Real samples per class: {args.real_samples}")
    print(f"Sampling steps: {args.num_steps}")
    print(f"Sampler: {args.sampler}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Evaluation runs: {args.num_runs}")
    print("-" * 80)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_models(args.vae_checkpoint, args.ldm_checkpoint, args.model_type, args.device)
    print("✓ Model loaded successfully")
    
    # Run evaluation multiple times for std computation
    all_run_results = {class_name: [] for class_name in CLASS_NAMES}
    all_run_results['overall'] = []
    
    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n{'='*80}")
            print(f"RUN {run_idx + 1}/{args.num_runs}")
            print(f"{'='*80}")
            torch.manual_seed(42 + run_idx)
            torch.cuda.manual_seed_all(42 + run_idx)
        
        # Evaluate each class
        run_results = {}
        all_real = []
        all_fake = []
        
        for class_id in range(NUM_CLASSES):
            class_name = CLASS_NAMES[class_id]
            print(f"\n{'='*80}")
            print(f"Evaluating class: {class_name.upper()} (ID: {class_id})")
            print(f"{'='*80}")
            
            # Load real images
            print(f"\n📥 Loading {args.real_samples} real {class_name} images...")
            real_images = get_real_images(class_id, args.real_samples, args.device)
            print(f"✓ Loaded {len(real_images)} real images")
            
            # Generate fake images
            print(f"\n🎨 Generating {args.samples_per_class} fake {class_name} images...")
            fake_images = generate_samples(
                model, class_id, args.samples_per_class,
                args.num_steps, args.cfg_scale, args.batch_size, args.device,
                args.sampler, args.model_type
            )
            print(f"✓ Generated {len(fake_images)} fake images")
            
            # Compute FID for this class
            print(f"\n📊 Computing FID for {class_name}...")
            fid_score = compute_fid(real_images, fake_images, args.device)
            run_results[class_name] = fid_score
            all_run_results[class_name].append(fid_score)
            
            print(f"✓ FID ({class_name}): {fid_score:.2f}")
            
            # Store for overall FID
            all_real.append(real_images)
            all_fake.append(fake_images)
        
        # Compute overall FID for this run
        print(f"\n{'='*80}")
        print("Computing OVERALL FID across all classes...")
        print(f"{'='*80}")
        
        all_real_combined = torch.cat(all_real, dim=0)
        all_fake_combined = torch.cat(all_fake, dim=0)
        
        overall_fid = compute_fid(all_real_combined, all_fake_combined, args.device)
        all_run_results['overall'].append(overall_fid)
    
    # Compute statistics across runs
    results = {}
    results_std = {}
    for key in all_run_results:
        scores = np.array(all_run_results[key])
        results[key] = scores.mean()
        results_std[key] = scores.std() if len(scores) > 1 else 0.0
    
    # Print results
    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS")
    print(f"{'='*80}")
    
    for class_name in CLASS_NAMES:
        fid_mean = results[class_name]
        fid_std = results_std[class_name]
        if args.num_runs > 1:
            print(f"  {class_name.capitalize():10s}: FID = {fid_mean:8.2f} ± {fid_std:6.2f}")
        else:
            print(f"  {class_name.capitalize():10s}: FID = {fid_mean:8.2f}")
    
    overall_mean = results['overall']
    overall_std = results_std['overall']
    if args.num_runs > 1:
        print(f"\n  {'Overall':10s}: FID = {overall_mean:8.2f} ± {overall_std:6.2f}")
    else:
        print(f"\n  {'Overall':10s}: FID = {overall_mean:8.2f}")
    print(f"{'='*80}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        model_name = "Flow Matching" if args.model_type == "flow" else "Noise Prediction"
        f.write(f"{model_name} Latent Diffusion Model - FID Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"VAE Checkpoint: {args.vae_checkpoint}\n")
        f.write(f"LDM Checkpoint: {args.ldm_checkpoint}\n")
        f.write(f"Samples per class: {args.samples_per_class}\n")
        f.write(f"Real samples per class: {args.real_samples}\n")
        f.write(f"Sampling steps: {args.num_steps}\n")
        f.write(f"Sampler: {args.sampler}\n")
        f.write(f"CFG scale: {args.cfg_scale}\n")
        f.write(f"Evaluation runs: {args.num_runs}\n\n")
        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        for class_name in CLASS_NAMES:
            fid_mean = results[class_name]
            fid_std = results_std[class_name]
            if args.num_runs > 1:
                f.write(f"  {class_name.capitalize():10s}: FID = {fid_mean:8.2f} ± {fid_std:6.2f}\n")
            else:
                f.write(f"  {class_name.capitalize():10s}: FID = {fid_mean:8.2f}\n")
        overall_mean = results['overall']
        overall_std = results_std['overall']
        if args.num_runs > 1:
            f.write(f"\n  {'Overall':10s}: FID = {overall_mean:8.2f} ± {overall_std:6.2f}\n")
        else:
            f.write(f"\n  {'Overall':10s}: FID = {overall_mean:8.2f}\n")
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
