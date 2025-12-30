# diffusion-pytorch
This repos contains some experimentation around diffusion using pytorch. 

## Setup
Make sure you have [Poetry](https://python-poetry.org/docs/) installed. Then, clone the repository and navigate into it:
```bash
git clone https://github.com/tchauffi/diffusion-pytorch.git
cd diffusion-pytorch
```

Use poetry to install dependencies:
```bash
poetry install
```

## Training
To train a diffusion model, run:
```bash
poetry run python train.py
```

## Generating Images using Gradio App

![Gradio](doc/gradio-app.png)

To generate images using a trained diffusion model, run the Gradio app:
```bash
poetry run python app.py
```

The app will be accessible at `http://localhost:7860`. You can specify the number of images to generate, the number of diffusion steps, and the path to the trained model checkpoint.

