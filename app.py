import torch
import gradio as gr
from diffusers.basic_model.models.diffusion_model import DiffusionModel
import numpy as np
from PIL import Image

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


# Load the trained model
def load_model(checkpoint_path="data/V1-ema-epoch=99-step=12800.ckpt" , device = "cpu"):
    """Load the trained diffusion model from checkpoint"""
    model = DiffusionModel.load_from_checkpoint(
        checkpoint_path,
        in_channels=3,
        out_channels=3, 
        num_filters=32,
        num_steps=1000,
        lr=1e-3,
        map_location=torch.device(device)
    )
    model.eval()
    model = model.to(device)
    return model


def generate_images(num_images, diffusion_steps, checkpoint_path, device = "cpu" if not torch.cuda.is_available() else "cuda"):
    """Generate images using the trained diffusion model"""
    try:
        # Load model
        model = load_model(checkpoint_path, device=device)
        
        # Generate random noise
        noise = torch.randn(num_images, 3, 64, 64).to(device)
        
        # Run reverse diffusion
        with torch.no_grad():
            generated_images = model.reverse_diffusion(noise, diffusion_steps)
        
        # Convert to images
        generated_images = (generated_images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        generated_images = generated_images.clamp(0, 1)
        generated_images = generated_images.cpu().numpy()
        generated_images = (generated_images * 255).astype(np.uint8)
        
        # Convert to PIL images - return as list of tuples (image, caption)
        pil_images = []
        for i in range(num_images):
            img_array = generated_images[i].transpose(1, 2, 0)  # CHW -> HWC
            pil_images.append((Image.fromarray(img_array), f"Image {i+1}"))
        
        return pil_images
    
    except Exception as e:
        return [(Image.new('RGB', (64, 64), color='red'), f"Error: {str(e)}")]


# Create Gradio interface
with gr.Blocks(title="Diffusion Model Demo") as demo:
    gr.Markdown("""
    # 🎨 Diffusion Model Image Generator
    
    Generate flower images using a trained diffusion model. The model was trained on the Flowers-102 dataset
    and generates 64x64 images through a reverse diffusion process.
    """)
    
    with gr.Row():
        with gr.Column():
            num_images = gr.Slider(
                minimum=1,
                maximum=16,
                value=4,
                step=1,
                label="Number of Images",
                info="How many images to generate"
            )
            
            diffusion_steps = gr.Slider(
                minimum=10,
                maximum=1000,
                value=50,
                step=10,
                label="Diffusion Steps",
                info="More steps = higher quality but slower"
            )
            
            checkpoint_path = gr.Textbox(
                value="data/V1-ema-epoch=99-step=12800.ckpt",
                label="Checkpoint Path",
                info="Path to the trained model checkpoint"
            )
            
            generate_btn = gr.Button("🎨 Generate Images", variant="primary")
        
        with gr.Column():
            output_gallery = gr.Gallery(
                label="Generated Images",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain"
            )
    
    gr.Markdown("""
    ### Tips:
    - Fewer diffusion steps (50-100) are faster but lower quality
    - More steps (500-1000) give better results but take longer
    - The model generates 64x64 flower images
    """)
    
    generate_btn.click(
        fn=generate_images,
        inputs=[num_images, diffusion_steps, checkpoint_path],
        outputs=output_gallery
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
