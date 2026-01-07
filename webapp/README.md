# Latent Diffusion Web App

A web application for generating images using a Latent Diffusion Model, running entirely in the browser via WebAssembly.

## Architecture

```
webapp/
├── diffusion-wasm/     # Rust WASM inference engine
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs      # Main WASM bindings
│       ├── unet.rs     # UNet model
│       ├── vae.rs      # VAE decoder
│       └── scheduler.rs # DDIM scheduler
│
└── frontend/           # Next.js web interface
    ├── app/
    │   ├── page.tsx    # Main UI
    │   └── layout.tsx
    └── public/
        ├── wasm/       # Compiled WASM (generated)
        └── models/     # SafeTensors model files
```

## Tech Stack

- **Inference**: [Candle](https://github.com/huggingface/candle) - Rust ML framework
- **WASM**: [wasm-pack](https://rustwasm.github.io/wasm-pack/) - Rust to WASM compiler
- **Frontend**: [Next.js](https://nextjs.org/) 14 with TypeScript
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Hosting**: GitHub Pages (static)

## Prerequisites

- Rust (stable)
- wasm-pack
- Node.js 18+

## Development

### 1. Build WASM Module

```bash
cd webapp/diffusion-wasm
wasm-pack build --target web --dev
```

### 2. Copy WASM to Frontend

```bash
mkdir -p webapp/frontend/public/wasm
cp -r webapp/diffusion-wasm/pkg/* webapp/frontend/public/wasm/
```

### 3. Copy Model Files

```bash
mkdir -p webapp/frontend/public/models
cp data/latent_diffusion_model/vae/vae-ema-final.safetensors webapp/frontend/public/models/
cp data/latent_diffusion_model/unet/ldm-final.safetensors webapp/frontend/public/models/
```

### 4. Run Development Server

```bash
cd webapp/frontend
npm install
npm run dev
```

Open http://localhost:3000

## Production Build

```bash
# Build WASM (release)
cd webapp/diffusion-wasm
wasm-pack build --target web --release

# Build Next.js (static export)
cd ../frontend
npm run build
```

The static site will be in `webapp/frontend/out/`.

## Deployment

The project uses GitHub Actions to automatically deploy to GitHub Pages.

1. Enable GitHub Pages in repository settings (Source: GitHub Actions)
2. Push to `main` or `webapp` branch
3. The workflow will:
   - Build the Rust WASM module
   - Build the Next.js static site
   - Deploy to GitHub Pages

## Model Weight Keys

The Rust implementation expects SafeTensors with these weight key patterns:

### VAE Decoder
- `decoder.conv_in.weight/bias`
- `decoder.mid_block1.norm1.weight/bias`, etc.
- `decoder.norm_out.weight/bias`
- `decoder.conv_out.weight/bias`

### UNet
- `time_embed.0.weight/bias`, `time_embed.2.weight/bias`
- `class_embed.weight`
- `conv_in.weight/bias`
- `down_blocks.{i}.res1.norm1.weight/bias`, etc.
- `mid_block.res1.norm1.weight/bias`, etc.
- `up_blocks.{i}.res1.norm1.weight/bias`, etc.
- `norm_out.weight/bias`
- `conv_out.weight/bias`

## Limitations

- **Model Size**: The UNet is 758MB, which takes time to download
- **CPU Only**: WASM runs on CPU, so generation is slower than GPU
- **Memory**: Large models may cause OOM on devices with <4GB RAM
- **Browser Support**: Requires modern browser with WASM support

## Tips for Optimization

1. **Quantization**: Convert model to FP16 or INT8 to reduce size
2. **Model Distillation**: Train a smaller model
3. **Progressive Loading**: Show intermediate results during generation
4. **Web Workers**: Run inference in a worker to avoid UI blocking
