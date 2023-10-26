from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer

class NewResBlock(ResBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
        dropout_prob: float = 0.5,
    ):
        super(NewResBlock, self).__init__(spatial_dims, in_channels, norm, kernel_size, act)

        # Introducing a new dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.conv1(x)

        x = self.act(x) 
        x = self.dropout(x)
        x = self.conv2(x)

        x += identity

        return x

class ResNet3d(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (3, 3, 9, 3),
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[NewResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        return x
    
    
class ResNet3dClassification(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        num_classes: int = 2,  # number of classes for classification
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (3, 3, 9, 3),
    ):
        super().__init__()

        self.encoder = ResNet3d(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            norm_name=norm_name,
            num_groups=num_groups,
            use_conv_final=use_conv_final,
            blocks_down=blocks_down
        )

        # Define the classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(init_filters * (2 ** (len(blocks_down) - 1)), num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)
        return x
