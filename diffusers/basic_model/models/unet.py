import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
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
        self.silu = F.silu

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
        print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.silu(out)
        return out
