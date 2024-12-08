import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps, dim, max_period=1000.0):
    """
    Generate sinusoidal embeddings for the given timesteps.
    """
    half_dim = dim // 2
    frequencies = torch.exp(
        torch.log(torch.tensor(max_period, dtype=torch.float32))
        * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        / half_dim
    ).to(timesteps.device)
    args = timesteps[:, None].float() * frequencies[None]
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embeddings = torch.cat(
            [embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1
        )
    return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


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
    ):
        super(ResidualConvBlock, self).__init__()
        self.pre_conv = in_channels != out_channels
        if self.pre_conv:
            self.conv0 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, embeddings):
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

        x = x + embeddings[:, : x.shape[1], :, :]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.silu(out)
        out += x
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
        self.conv1 = ResidualConvBlock(in_channels, out_channels)
        self.conv2 = ResidualConvBlock(out_channels, out_channels)
        self.downsample = nn.MaxPool2d(2)
        if attention:
            self.attention = AttentionBlock(out_channels, num_heads=4)

    def forward(self, x, embeddings):
        x = self.conv1(x, embeddings)
        if hasattr(self, "attention"):
            x = self.attention(x)
        skip = self.conv2(x, embeddings)
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
        self.conv1 = ResidualConvBlock(in_channels, out_channels)
        self.conv2 = ResidualConvBlock(out_channels, out_channels)
        self.out_channels = out_channels
        if attention:
            self.attention = AttentionBlock(out_channels, num_heads=4)

    def forward(self, x, skip, embeddings):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x, embeddings)
        if hasattr(self, "attention"):
            x = self.attention(x)
        x = self.conv2(x, embeddings)
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

    # TODO : make the unet more flexible to parameter
    """

    def __init__(self, in_channels, out_channels, num_filters=32):
        super(UNet, self).__init__()

        self.max_filters_nbr = num_filters * 8 + num_filters * 4
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
        self.end_norm = nn.GroupNorm(num_filters, num_filters)
        self.out = nn.Conv2d(num_filters, out_channels, 1)
        self.time_embedder = timestep_embedding

    def forward(self, x, t):
        x = self.entry_conv(x)
        embeds = (
            self.time_embedder(t, self.max_filters_nbr)
            .unsqueeze(2)
            .unsqueeze(3)
            .to(x.device)
        )
        x, skip1 = self.down1(x, embeds)
        x, skip2 = self.down2(x, embeds)
        x, skip3 = self.down3(x, embeds)
        x = self.bottleneck1(x, embeds)
        x = self.bottleneck2(x, embeds)
        x = self.up3(x, skip3, embeds)
        x = self.up2(x, skip2, embeds)
        x = self.up1(x, skip1, embeds)
        x = self.end_norm(x)
        x = F.silu(x)
        x = self.out(x)
        return x
