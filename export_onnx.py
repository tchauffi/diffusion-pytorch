"""Export VAE and UNet models to ONNX format for web inference."""

import torch
import torch.onnx
from pathlib import Path
import onnx

# Import the model classes
from diffusers.vae import VAE
from diffusers.latent_diffusion import LatentDiffusionModel


def convert_to_single_file(onnx_path: str):
    """Convert ONNX model with external data to a single file."""
    print(f"Converting {onnx_path} to single file...")
    model = onnx.load(onnx_path)
    # Convert external data to embedded
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=False,
    )
    # Remove the .data file if it exists
    data_path = Path(onnx_path + ".data")
    if data_path.exists():
        data_path.unlink()
        print(f"Removed external data file: {data_path}")


def export_vae_decoder(vae: VAE, output_path: str):
    """Export only the VAE decoder to ONNX (we only need decoding for inference)."""
    print(f"Exporting VAE decoder to {output_path}...")
    vae.eval()
    
    # Create a wrapper that only exposes the decoder
    class VAEDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.decoder = vae.decoder
        
        def forward(self, z):
            return self.decoder(z)
    
    decoder = VAEDecoder(vae)
    decoder.eval()
    
    # Dummy input: latent tensor (batch, 4, 16, 16) for 128x128 images
    # VAE with 4 resolution levels (channel_multipliers [1,2,2,4]) downsamples by 2^3 = 8
    # So 128 / 8 = 16
    dummy_latent = torch.randn(1, 4, 16, 16)
    
    print(f"Exporting VAE decoder to {output_path}...")
    torch.onnx.export(
        decoder,
        dummy_latent,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['latent'],
        output_names=['image'],
        dynamic_axes={
            'latent': {0: 'batch_size'},
            'image': {0: 'batch_size'}
        }
    )
    print(f"VAE decoder exported successfully!")


def export_unet(ldm: LatentDiffusionModel, output_path: str):
    """Export the UNet to ONNX."""
    print(f"Exporting UNet to {output_path}...")
    
    # Get the UNet from the LDM
    unet = ldm.unet
    unet.eval()
    
    # Wrapper for ONNX export with proper inputs
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
        
        def forward(self, x, timestep, class_label):
            return self.unet(x, timestep, class_label)
    
    wrapper = UNetWrapper(unet)
    wrapper.eval()
    
    # Dummy inputs
    batch_size = 1
    dummy_x = torch.randn(batch_size, 4, 16, 16)  # Latent at 16x16 for 128x128 images
    dummy_timestep = torch.tensor([0.5])  # Normalized timestep in [0, 1]
    dummy_class = torch.tensor([0], dtype=torch.long)  # Class label
    
    print(f"Exporting UNet to {output_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_timestep, dummy_class),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['latent', 'timestep', 'class_label'],
        output_names=['noise_pred'],
        dynamic_axes={
            'latent': {0: 'batch_size'},
            'timestep': {0: 'batch_size'},
            'class_label': {0: 'batch_size'},
            'noise_pred': {0: 'batch_size'}
        }
    )
    print(f"UNet exported successfully!")


if __name__ == "__main__":
    # Paths - using the same structure as the notebook
    vae_path = "data/latent_diffusion_model/vae/vae-ema-final.safetensors"
    unet_path = "data/latent_diffusion_model/unet/ldm-final.safetensors"
    output_dir = Path("webapp/frontend/public/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models using from_pretrained (like the notebook)
    print("Loading VAE...")
    vae = VAE.from_pretrained(vae_path)
    
    print("Loading LDM...")
    ldm = LatentDiffusionModel.from_pretrained(vae, unet_path)
    
    # Export models
    vae_onnx_path = str(output_dir / "vae-decoder.onnx")
    unet_onnx_path = str(output_dir / "unet.onnx")
    
    export_vae_decoder(vae, vae_onnx_path)
    export_unet(ldm, unet_onnx_path)
    
    # Convert to single files (embed external data)
    print("\nConverting models to single files...")
    convert_to_single_file(vae_onnx_path)
    convert_to_single_file(unet_onnx_path)
    
    print("\nDone! Models exported to:")
    print(f"  - {output_dir / 'vae-decoder.onnx'}")
    print(f"  - {output_dir / 'unet.onnx'}")
