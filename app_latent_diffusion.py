"""
Gradio App for Latent Diffusion Model

A web interface to generate images using a trained Latent Diffusion Model.
"""

import torch
import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global model cache
_model_cache = {}


def load_latent_diffusion_model(vae_checkpoint: str, ldm_checkpoint: str, device: str = "cpu", num_classes: int = None):
    """Load the Latent Diffusion Model with pretrained VAE and UNet."""
    from diffusers.vae.model import VAE
    from diffusers.latent_diffusion.latent_diffusion import LatentDiffusionModel, LatentUNet
    
    cache_key = f"{vae_checkpoint}_{ldm_checkpoint}_{num_classes}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # VAE configuration (should match your trained VAE)
    vae = VAE(
        latent_dim=4,
        base_channels=64,
        channel_multipliers=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0,
        input_resolution=128
    )
    
    # Load VAE weights
    if Path(vae_checkpoint).exists():
        checkpoint = torch.load(vae_checkpoint, map_location=device, weights_only=False)
        if 'vae_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['vae_state_dict'])
            print(f"Loaded VAE from EMA checkpoint: {vae_checkpoint}")
        elif 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = {k.replace('vae.', ''): v for k, v in checkpoint['state_dict'].items() 
                          if k.startswith('vae.')}
            if state_dict:
                vae.load_state_dict(state_dict)
            print(f"Loaded VAE from Lightning checkpoint: {vae_checkpoint}")
        else:
            vae.load_state_dict(checkpoint)
            print(f"Loaded VAE state dict: {vae_checkpoint}")
    else:
        print(f"WARNING: VAE checkpoint not found: {vae_checkpoint}")
    
    # Load LDM checkpoint to get config
    ldm_config = {
        'latent_channels': 4,
        'base_channels': 256,
        'channel_multipliers': (1, 2, 2, 4),
        'num_steps': 1000,
        'num_classes': None
    }
    
    ldm_checkpoint_data = None
    if Path(ldm_checkpoint).exists():
        ldm_checkpoint_data = torch.load(ldm_checkpoint, map_location=device, weights_only=False)
        if 'config' in ldm_checkpoint_data:
            ldm_config.update(ldm_checkpoint_data['config'])
            print(f"Loaded config from checkpoint: {ldm_config}")
    
    # Override num_classes if provided
    if num_classes is not None and num_classes > 0:
        ldm_config['num_classes'] = num_classes
    
    # Create LDM with config from checkpoint
    model = LatentDiffusionModel(
        vae=vae,
        latent_channels=ldm_config['latent_channels'],
        base_channels=ldm_config['base_channels'],
        channel_multipliers=ldm_config['channel_multipliers'],
        num_steps=ldm_config['num_steps'],
        use_ema=False,  # Will load EMA weights directly
        num_classes=ldm_config['num_classes']
    )
    
    # Load LDM/UNet weights
    if ldm_checkpoint_data is not None:
        if 'ema_state_dict' in ldm_checkpoint_data and ldm_checkpoint_data['ema_state_dict'] is not None:
            model.unet.load_state_dict(ldm_checkpoint_data['ema_state_dict'])
            print(f"Loaded UNet from EMA weights: {ldm_checkpoint}")
        elif 'unet_state_dict' in ldm_checkpoint_data:
            model.unet.load_state_dict(ldm_checkpoint_data['unet_state_dict'])
            print(f"Loaded UNet weights: {ldm_checkpoint}")
        elif 'state_dict' in ldm_checkpoint_data:
            # Lightning checkpoint format
            state_dict = {k.replace('unet.', ''): v for k, v in ldm_checkpoint_data['state_dict'].items() 
                          if k.startswith('unet.')}
            if state_dict:
                model.unet.load_state_dict(state_dict)
            print(f"Loaded UNet from Lightning checkpoint: {ldm_checkpoint}")
    else:
        print(f"WARNING: LDM checkpoint not found: {ldm_checkpoint}")
    
    model.eval()
    model = model.to(device)
    
    _model_cache[cache_key] = model
    return model


@torch.no_grad()
def generate_images(
    num_images: int,
    num_steps: int,
    seed: int = -1,
    class_name: str = "random",
    cfg_scale: float = 1.0,
    progress=gr.Progress()
):
    """Generate images using the Latent Diffusion Model."""
    # Hardcoded paths
    vae_checkpoint = "data/cat_model/vae-ema-final.pt"
    ldm_checkpoint = "data/cat_model/ldm-final.pt"
    
    # Map class names to labels (3 = null/unconditional class)
    class_map = {"🐱 Cat": 0, "🐕 Dog": 1, "🦁 Wild": 2, "🎲 Random": 3}
    class_label = class_map.get(class_name, 3)
    
    # Random uses unconditional mode (null class)
    is_unconditional = (class_label == 3)
    
    try:
        progress(0, desc="Loading model...")
        model = load_latent_diffusion_model(vae_checkpoint, ldm_checkpoint, device, num_classes=3)
        
        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        
        progress(0.1, desc="Generating latents...")
        
        # Get latent shape (4 channels, 16x16 for 128x128 images with 8x downsampling)
        latent_shape = (4, 16, 16)
        num_training_steps = 1000  # Must match training
        
        # Custom DDIM sampling with progress
        from diffusers.basic_model.schedulers import offset_cosine_diffusion_scheduler
        
        # Start with pure noise
        z = torch.randn(num_images, *latent_shape, device=device)
        
        # Use model's UNet
        unet = model.unet
        unet.eval()
        
        # Prepare class labels for conditional generation (always use classes)
        nc = 3
        use_cfg = cfg_scale > 1.0 and not is_unconditional
        class_labels = torch.full((num_images,), class_label, device=device, dtype=torch.long)
        null_labels = torch.full((num_images,), nc, device=device, dtype=torch.long)  # null class
        
        # Create timestep subsequence for DDIM (allows fewer steps than training)
        step_ratio = num_training_steps / num_steps
        timesteps = (torch.arange(num_steps, device=device) * step_ratio).long()
        timesteps = torch.flip(timesteps, [0])  # High noise to low
        
        # Get schedule for all training steps
        all_steps = torch.arange(num_training_steps, device=device)
        all_alphas, all_betas = offset_cosine_diffusion_scheduler(all_steps / num_training_steps)
        all_alphas = all_alphas.to(device)  # noise rate
        all_betas = all_betas.to(device)    # signal rate
        
        # DDIM sampling loop (deterministic)
        for i, t in enumerate(timesteps):
            progress(i / len(timesteps) * 0.8 + 0.1, 
                    desc=f"DDIM step {i+1}/{len(timesteps)}")
            
            t_input = torch.full((num_images,), t.item(), device=device, dtype=torch.float32)
            
            alpha_t = all_alphas[t].view(1, 1, 1, 1)
            beta_t = all_betas[t].view(1, 1, 1, 1)
            
            # Predict noise with optional classifier-free guidance
            if use_cfg:
                # Conditional prediction
                noise_pred_cond = unet(z, t_input / num_training_steps, class_labels)
                # Unconditional prediction
                noise_pred_uncond = unet(z, t_input / num_training_steps, null_labels)
                # CFG: combine predictions
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No CFG
                noise_pred = unet(z, t_input / num_training_steps, class_labels)
            
            # DDIM: predict clean latent
            z0_pred = (z - alpha_t * noise_pred) / beta_t
            z0_pred = torch.clamp(z0_pred, -3, 3)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = all_alphas[t_prev].view(1, 1, 1, 1)
                beta_prev = all_betas[t_prev].view(1, 1, 1, 1)
                
                # DDIM deterministic update
                z = beta_prev * z0_pred + alpha_prev * noise_pred
            else:
                z = z0_pred
        
        progress(0.9, desc="Decoding images...")
        
        # Decode to image space
        scale_factor = 0.38  # Match training scale factor
        with torch.no_grad():
            model.vae.eval()
            images = model.vae.decode(z / scale_factor)
        
        # Convert to PIL images
        images = (images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0, 1)
        images = images.cpu().numpy()
        images = (images * 255).astype(np.uint8)
        
        pil_images = []
        for i in range(num_images):
            img_array = images[i].transpose(1, 2, 0)  # CHW -> HWC
            pil_images.append((Image.fromarray(img_array), f"Image {i+1}"))
        
        progress(1.0, desc="Done!")
        return pil_images
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_img = Image.new('RGB', (128, 128), color='red')
        return [(error_img, f"Error: {str(e)}")]


def clear_cache():
    """Clear the model cache."""
    global _model_cache
    _model_cache = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Cache cleared!"


# Create Gradio interface
with gr.Blocks(title="Latent Diffusion Model Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 This cat does not exist
    
    Generate images using a trained Latent Diffusion Model. The model operates in a compressed 
    latent space (16×16) using a VAE, making generation faster and more memory efficient than 
    pixel-space diffusion.
                
    The model have been trained on a dataset of cat, dog, and wild animal images using the [AFHQ dataset](https://huggingface.co/datasets/huggan/AFHQ).
    
    **Architecture:** VAE (128×128 → 16×16 latent) + UNet denoiser
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            
            num_images = gr.Slider(
                minimum=1,
                maximum=16,
                value=4,
                step=1,
                label="Number of Images",
                info="How many images to generate"
            )
            
            num_steps = gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=5,
                label="Diffusion Steps",
                info="More steps = higher quality but slower (50-100 recommended)"
            )
            
            seed = gr.Number(
                value=-1,
                label="Seed",
                info="Set to -1 for random, or use a specific seed for reproducibility",
                precision=0
            )
            
            gr.Markdown("### 🏷️ Class Selection")
            
            class_name = gr.Radio(
                choices=["🐱 Cat", "🐕 Dog", "🦁 Wild", "🎲 Random"],
                value="🐱 Cat",
                label="Image Class",
                info="Select which type of animal to generate"
            )
            
            cfg_scale = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                value=3.0,
                step=0.5,
                label="CFG Scale",
                info="Classifier-free guidance strength (1.0=no guidance, 3-7=typical)"
            )
            
            with gr.Row():
                generate_btn = gr.Button("🎨 Generate Images", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear Cache", variant="secondary", scale=1)
            
            status = gr.Textbox(label="Status", interactive=False, visible=False)
        
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Generated Images")
            output_gallery = gr.Gallery(
                label="Generated Images",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                show_label=False
            )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - **First generation** may be slow as the model loads into memory
    - **50-100 steps** usually gives good results, more steps = finer details
    - Use a **specific seed** to reproduce the same images
    - **CFG scale 3-7** typically works well for better image quality
    - Lower **number of images** if you run out of GPU memory
    
    ### 📊 Model Info
    - **Classes:** Cat, Dog, Wild animals
    - **Input Resolution:** 128×128 pixels
    - **Latent Resolution:** 16×16 (4 channels)
    - **Compression:** 8× spatial, 512× total
    """)
    
    # Event handlers
    generate_btn.click(
        fn=generate_images,
        inputs=[num_images, num_steps, seed, class_name, cfg_scale],
        outputs=[output_gallery]
    )
    
    clear_btn.click(
        fn=clear_cache,
        outputs=[status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[status]
    )


if __name__ == "__main__":
    demo.queue()  # Enable queueing for progress updates
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )
