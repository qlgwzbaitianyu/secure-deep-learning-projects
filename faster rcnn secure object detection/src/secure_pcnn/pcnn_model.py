from __future__ import annotations

from itertools import product
from typing import Sequence

import torch
from torch import compile as torch_compile
from torch import nn

from .smc import SecureSMC


class SecurePCNN(nn.Module):
    """Configurable PCNN feature extractor used in the secure detector experiments.

    The original research implementation hard-coded the convolutional channel
    sizes and pooling parameters. This version takes those values from the
    external configuration and makes the secure-share inputs explicit.
    """

    def __init__(
        self,
        input_channels: int,
        feature_channels: Sequence[int],
        conv_kernel_size: int,
        conv_padding: int,
        pool_kernel_size: int,
        pool_stride: int,
        smc: SecureSMC,
    ) -> None:
        super().__init__()
        if not feature_channels:
            raise ValueError("feature_channels must not be empty")

        self.smc = smc
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in feature_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                )
            )
            in_channels = out_channels
        self.features = nn.ModuleList(layers)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.relu = nn.ReLU(inplace=True)

    def convolution1(self, x: torch.Tensor) -> torch.Tensor:
        return self.features[0](x)

    def secure_relu(self, bottom: torch.Tensor, peer_share: torch.Tensor) -> torch.Tensor:
        if bottom.shape != peer_share.shape:
            raise ValueError("bottom and peer_share must have identical shapes")

        output = bottom.detach().clone()
        for index in product(*(range(size) for size in output.shape)):
            value_1 = float(bottom[index].item())
            value_2 = float(peer_share[index].item())
            _, _, msb_1, msb_2 = self.smc.SecBitExtra(value_1, value_2)
            if msb_1 ^ msb_2:
                output[index] = 0
        return output

    def secure_max_pool(
        self,
        bottom: torch.Tensor,
        peer_share: torch.Tensor,
        return_first_share: bool,
    ) -> torch.Tensor:
        if bottom.shape != peer_share.shape:
            raise ValueError("bottom and peer_share must have identical shapes")
        if bottom.ndim != 4:
            raise ValueError("expected [batch, channels, height, width]")

        batch, channels, height, width = bottom.shape
        kernel = self.pool.kernel_size
        stride = self.pool.stride
        if isinstance(kernel, int):
            kernel_height = kernel_width = kernel
        else:
            kernel_height, kernel_width = kernel
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        out_height = 1 + (height - kernel_height) // stride_h if height >= kernel_height else 0
        out_width = 1 + (width - kernel_width) // stride_w if width >= kernel_width else 0
        output = torch.zeros(
            (batch, channels, out_height, out_width),
            dtype=bottom.dtype,
            device=bottom.device,
        )

        for n in range(batch):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        start_h = h * stride_h
                        start_w = w * stride_w
                        x_slice = bottom[n, c, start_h:start_h + kernel_height, start_w:start_w + kernel_width]
                        y_slice = peer_share[n, c, start_h:start_h + kernel_height, start_w:start_w + kernel_width]
                        x, y = self.smc.SecMaxIndex(x_slice, y_slice)
                        selected = bottom if return_first_share else peer_share
                        output[n, c, h, w] = selected[n, c, start_h + x, start_w + y]
        return output
