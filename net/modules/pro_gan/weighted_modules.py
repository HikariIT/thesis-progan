import torch.nn as nn
import numpy as np
import torch


class WeightedConv2d(nn.Module):
    # Conv2d layer with equalized learning rate

    conv: nn.Conv2d
    bias: nn.Parameter
    scale: float

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, gain: float = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # This part is related to equalized learning rate. It scale the weights of a layer based on how many weights it has.
        bias = self.conv.bias
        self.conv.bias = None

        if bias is None:
            bias = torch.zeros(out_channels)
            raise ValueError('Bias is None in WeightedConv2d. This is unexpected. Please check the code (weighted_modules.py)')

        scale_factor = np.prod(list(self.conv.weight.shape)[1:])
        self.scale = gain * np.sqrt(2 / (scale_factor)) # self.scale = gain/np.sqrt(scale_factor), with gain = np.sqrt(2), pending for testing...
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1, 1))

        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias


class WeightedLinear(nn.Module):
    # Linear layer with equalized learning rate

    linear: nn.Linear
    bias: nn.Parameter
    scale: float

    def __init__(self, in_features: int, out_features: int, gain: float = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = self.linear.bias
        self.linear.bias = None # type: ignore - TODO: Check if this is needed or not

        scale_factor = in_features
        self.scale = gain * np.sqrt(2 / scale_factor)

        nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias
