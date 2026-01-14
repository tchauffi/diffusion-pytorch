from typing import Callable, Optional

import torch
from torch import nn

from diffusers.basic_model.models.unet import timestep_embedding


class PatchEmbedder(nn.Module):
    def __init__(self, img_size:int=224, patch_size:int=16, in_chan:int=3, embed_dim:int=768, flatten:bool=True, bias:bool=True, norm_layer: Optional[Callable]= None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.flatten = flatten
        self.norm = norm_layer if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x)  # B, embed_dim, grid_size, grid_size
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        x = self.norm(x)
        return x

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes:int, embed_dim:int, dropout:float=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes + 1 , embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        x = self.dropout(class_labels + 1 ) # shift by 1 to account for unconditional class
        x = self.embedding(x) # B, embed_dim
        return x
    
class TimeEmbedder(nn.Module):
    def __init__(self, embed_dim:int, max_period:int=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        x = timestep_embedding(timesteps, self.embed_dim, self.max_period)  # B, embed_dim
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
