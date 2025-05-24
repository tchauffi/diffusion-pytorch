import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import pytorch_lightning as pl
from PIL import Image
from datasets import load_dataset
from diffusers.basic_model.models.diffusion_model import DiffusionModel
from diffusers.basic_model.callbacks import DiffusionImageLogger

from torchvision.transforms import functional as F
import torchvision


class CenterCrop(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            # Size must match the ratio while cropping to the edge of the image
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w >= h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return torchvision.transforms.functional.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


model = DiffusionModel(
    in_channels=3, out_channels=3, num_filters=32, num_steps=1000, lr=1e-3
)
model = model.to("mps")

img_callback = DiffusionImageLogger(num_images=4, every_n_epochs=1)

trainer = pl.Trainer(max_epochs=100, callbacks=[img_callback])

dataset = load_dataset("huggan/flowers-102-categories")

train_dataset = dataset["train"]

transfrom = transforms.Compose(
    [
        CenterCrop(),
        transforms.Resize((64, 64), Image.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def transforms_fn(examples):
    examples["image"] = [transfrom(image.convert("RGB")) for image in examples["image"]]
    return examples


train_dataset.set_transform(transforms_fn)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

trainer.fit(model, train_loader)
