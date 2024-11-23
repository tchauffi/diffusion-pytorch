import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()

        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(width)
        self.residual_compress = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        input_width = x.shape[1]
        if input_width != self.conv1.in_channels:
            residual = self.residual_compress(x)
        else:
            residual = x

        x = self.batch_norm(x)
        x = self.conv1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = x + residual

        return x


class DownsampleBlock(nn.Module):
    def __init__(self, width, block_depth):
        super().__init__()

        self.layers = nn.ModuleList([ResidualBlock(width) for _ in range(block_depth)])
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x, skips = x
        for layer in self.layers:
            x = layer(x)
        skips.append(x)

        x = self.pool(x)

        return x, skips


class UpsampleBlock(nn.Module):
    def __init__(self, width, block_depth):
        super().__init__()

        self.layers = nn.ModuleList([ResidualBlock(width) for _ in range(block_depth)])
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, x):
        x, skips = x
        x = self.upsample(x)

        skip = skips.pop()
        # concatenate the skip connection
        x = torch.cat([x, skip], dim=1)

        for layer in self.layers:
            x = layer(x)

        return x


def sinusoidal_embedding(x):
    frequencies = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(1.0)), torch.log(torch.tensor(10000.0)), 16
        )
    )

    angular_speeds = 2.0 * torch.pi * frequencies

    embedings = torch.cat(
        [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], dim=-1
    )

    return embedings
