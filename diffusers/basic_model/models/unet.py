import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SinusoidalEmbeddings(nn.Module):
    """
    This module generates sinusoidal embeddings for the input tensor.
    """

    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        """
        Forward pass of the sinusoidal embeddings module.
        """
        with torch.no_grad():
            embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]


class Attention(nn.Module):
    """
    This module performs multi-head scaled dot-product attention.
    """

    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        """
        Forward pass of the attention module.
        """
        h, w = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj1(x)
        x = rearrange(x, "b L (C H K) -> K b H L C", K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout_prob
        )
        x = rearrange(x, "b H (h w) C -> b h w (C H)", h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, "b h w C -> b C h w")


class ResidualConvBlock(nn.Module):
    """
    A residual convolutional block that consists of two convolutional layers with batch normalization
    and SiLU activation function. The input is added to the output of the second convolutional layer
    to form a residual connection.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolutional layer.
        silu (function): SiLU activation function.
    Methods:
        forward(x):
            Forward pass of the residual convolutional block.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after applying the residual connection, convolutions,
                              batch normalization, and SiLU activation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        attention=False,
    ):
        super(ResidualConvBlock, self).__init__()
        self.pre_conv = in_channels != out_channels
        self.attention = attention
        if self.pre_conv:
            self.conv0 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = F.silu

        if attention:
            self.attention = Attention(out_channels, num_heads=8, dropout_prob=0.1)

    def forward(self, x):
        """Forward pass of the ResNet block.
        This function performs the forward pass through a ResNet block with two convolutional layers,
        batch normalization, and skip connection (residual connection).
        Args:
            x (torch.Tensor): Input tensor. Shape: [batch_size, channels, height, width]
        Returns:
            torch.Tensor: Output tensor after passing through the ResNet block.
                         Has the same shape as the input tensor.
        Note:
            The residual connection allows the network to learn residual functions with reference
            to the input layer, helping with the training of very deep networks.
        """
        if self.pre_conv:
            x = self.conv0(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        if self.attention:
            out = self.attention(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.silu(out)
        return out


class DownBlock(nn.Module):
    """Down block in U-Net architecture.

    This block performs convolution operations followed by downsampling. It consists of
    a residual convolution block followed by max pooling. During forward pass, it returns
    both the downsampled feature map and a skip connection for later use in the corresponding
    up block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Downsampled feature map
            - torch.Tensor: Skip connection for corresponding up block
    """

    def __init__(self, in_channels, out_channels, attention=False):
        super(DownBlock, self).__init__()
        self.conv = ResidualConvBlock(in_channels, out_channels, attention=attention)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.downsample(skip)
        return x, skip


class UpBlock(nn.Module):
    """A module for upsampling feature maps in a U-Net architecture.

    This block performs bilinear upsampling of the input tensor followed by concatenation
    with a skip connection and convolution through a residual block. It's typically used
    in the decoder part of U-Net architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        upsample (nn.Upsample): Bilinear upsampling layer with scale factor of 2.
        conv (ResidualConvBlock): Residual convolution block for feature processing.

    Returns:
        torch.Tensor: Processed tensor after upsampling, concatenation and convolution.
    """

    def __init__(self, in_channels, out_channels, attention=False):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = ResidualConvBlock(in_channels, out_channels, attention=attention)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """U-Net architecture for image segmentation and generation tasks.
    This implementation follows the original U-Net architecture with downsampling and
    upsampling paths connected by skip connections. The network consists of a contracting
    path (downsampling), a bottleneck, and an expansive path (upsampling).
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_filters (int, optional): Number of base filters, which doubles at each downsampling step.
                                   Defaults to 32.
    Attributes:
        down1-4 (DownBlock): Downsampling blocks that reduce spatial dimensions and increase features
        center (ResidualConvBlock): Bottleneck block with residual connections
        up1-4 (UpBlock): Upsampling blocks that increase spatial dimensions and decrease features
        out (nn.Conv2d): Final 1x1 convolution to map to output channels
        sigmoid (nn.Sigmoid): Sigmoid activation for final output
    Returns:
        torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width)
    Note:
        The spatial dimensions of the input must be divisible by 16 due to the four
        downsampling operations in the network.
    """

    def __init__(self, in_channels, out_channels, num_filters=32):
        super(UNet, self).__init__()
        self.entry_conv = nn.Conv2d(in_channels, num_filters, 3, 1, 1)
        self.down1 = DownBlock(num_filters, num_filters)
        self.down2 = DownBlock(num_filters, num_filters * 2, attention=True)
        self.down3 = DownBlock(num_filters * 2, num_filters * 4)
        self.bottleneck1 = ResidualConvBlock(num_filters * 4, num_filters * 8)
        self.bottleneck2 = ResidualConvBlock(num_filters * 8, num_filters * 8)
        self.up3 = UpBlock(num_filters * 8 + num_filters * 4, num_filters * 4)
        self.up2 = UpBlock(
            num_filters * 4 + num_filters * 2, num_filters * 2, attention=True
        )
        self.up1 = UpBlock(num_filters * 2 + num_filters, num_filters)
        self.out = nn.Conv2d(num_filters, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

        self.time_embedder = SinusoidalEmbeddings(1000, 32)

    def forward(self, x, t):
        x = self.entry_conv(x)
        embeds = self.time_embedder(x, t)
        x = x + embeds
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
