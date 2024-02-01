# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..layers import ResLayer
from .resnet import BasicBlock

class FireBlock(nn.Module):
    def __init__(self, 
                 inp_dim, 
                 out_dim, 
                 sr=2, 
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(FireBlock, self).__init__(init_cfg)
        
        self.conv1    = nn.Conv2d(inp_dim, out_dim // sr, kernel_size=1, stride=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_dim // sr)
        self.conv_1x1 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=1, stride=stride, bias=False)
        self.conv_3x3 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=3, padding=1, 
                                  stride=stride, groups=out_dim // sr, bias=False)
        self.bn2      = nn.BatchNorm2d(out_dim)
        self.skip     = (stride == 1 and inp_dim == out_dim)
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        conv2 = torch.cat((self.conv_1x1(bn1), self.conv_3x3(bn1)), 1)
        bn2   = self.bn2(conv2)
        if self.skip:
            return self.relu(bn2 + x)
        else:
            return self.relu(bn2)
        
class FireLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Defaults to 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Defaults to None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Defaults to dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Defaults to True
    """

    def __init__(self,
                 block: BaseModule,
                 inplanes: int,
                 planes: int,
                 num_blocks: int,
                 stride: int = 1,
                 avg_down: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 downsample_first: bool = True,
                 **kwargs) -> None:
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []

        # downsample_first=False is for HourglassModule
        for _ in range(num_blocks - 1):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=inplanes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        super().__init__(*layers)

class HourglassSqModule(BaseModule):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in current and
            follow-up HourglassModule.
        norm_cfg (ConfigType): Dictionary to construct and config norm layer.
            Defaults to `dict(type='BN', requires_grad=True)`
        upsample_cfg (ConfigType): Config dict for interpolate layer.
            Defaults to `dict(mode='nearest')`
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization.
    """

    def __init__(self,
                 depth: int,
                 stage_channels: List[int],
                 stage_blocks: List[int],
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = FireLayer(
            FireBlock, cur_channel, cur_channel, cur_block, norm_cfg=norm_cfg)

        self.low1 = FireLayer(
            FireBlock, cur_channel, next_channel, cur_block, stride=2, norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassSqModule(depth - 1, stage_channels[1:], stage_blocks[1:])
        else:
            self.low2 = FireLayer(
                FireBlock, next_channel, next_channel, next_block, norm_cfg=norm_cfg)

        self.low3 = FireLayer(
            FireBlock, next_channel, cur_channel, cur_block, norm_cfg=norm_cfg, downsample_first=False)

        self.up2 = nn.ConvTranspose2d(cur_channel, cur_channel, 4, stride=2, padding=1)
        self.upsample_cfg = upsample_cfg

    def forward(self, x: torch.Tensor) -> nn.Module:
        """Forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        # Fixing `scale factor` (e.g. 2) is common for upsampling, but
        # in some cases the spatial size is mismatched and error will arise.
        up2 = self.up2(low3)
        
        return up1 + up2


@MODELS.register_module()
class HourglassSqNet(BaseModule):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`_ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (Sequence[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (Sequence[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (norm_cfg): Dictionary to construct and config norm layer.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization.

    Example:
        >>> from mmdet.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times: int = 5,
                 num_stacks: int = 2,
                 stage_channels: Sequence = (256, 256, 384, 384, 384, 512),
                 stage_blocks: Sequence = (2, 2, 2, 2, 2, 4),
                 feat_channel: int = 256,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 init_cfg: OptMultiConfig = None) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(
                3, cur_channel // 4, 7, padding=3, stride=2,
                norm_cfg=norm_cfg),
            ResLayer(
                BasicBlock,
                cur_channel // 4,
                cur_channel // 2,
                1,
                stride=2,
                norm_cfg=norm_cfg),
            ResLayer(
                BasicBlock,
                cur_channel // 2,
                cur_channel,
                1,
                stride=2,
                norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassSqModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            cur_channel,
            cur_channel,
            num_stacks - 1,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.ModuleList([
            ConvModule(
                cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.ModuleList([
            ConvModule(
                feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self) -> None:
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super().init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward function."""
        inter_feat = self.stem(x)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                inter_feat = self.conv1x1s[ind](
                    inter_feat) + self.remap_convs[ind](
                        out_feat)
                inter_feat = self.inters[ind](self.relu(inter_feat))

        return out_feats[-1:]
