#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

# from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .swin_transformer import build_swint
from .convnext import convnext_tiny, convnext_base, convnext_large
from .resnet import resnet50
import torch.utils.checkpoint as checkpoint
class YOLOPAFPNNEW(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    Unicorn supports more types of backbones like Swin, ConvNext, ResNet
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        backbone_name="swin_tiny_patch4_window7_224",
        build_fpn=True,
        use_checkpoint=False,
        checkpoint_whole_backbone=False,
    ):
        super().__init__()
        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.checkpoint_whole_backbone = checkpoint_whole_backbone
        if "swin" in backbone_name:
            self.backbone = build_swint(backbone_name)
        elif "convnext" in backbone_name:
            if backbone_name == "convnext_base":
                self.backbone = convnext_base(use_checkpoint=use_checkpoint)
            elif backbone_name == "convnext_large":
                self.backbone = convnext_large(use_checkpoint=use_checkpoint)
            else:
                self.backbone = convnext_tiny(use_checkpoint=use_checkpoint)
        elif backbone_name == "resnet50":
            self.backbone = resnet50(pretrained=True)
        else:
            raise ValueError()

        self.in_features = in_features
        self.in_channels = in_channels
        self.width = width
        if build_fpn:
            Conv = DWConv if depthwise else BaseConv

            """add BaseConv to adjust channel number"""
            if width != 1:
                self.adjust0 = BaseConv(in_channels[2], int(in_channels[2] * width), 1, 1, act=act)
                self.adjust1 = BaseConv(in_channels[1], int(in_channels[1] * width), 1, 1, act=act)
                self.adjust2 = BaseConv(in_channels[0], int(in_channels[0] * width), 1, 1, act=act)
            
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.lateral_conv0 = BaseConv(
                int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
            )
            self.C3_p4 = CSPLayer(
                int(2 * in_channels[1] * width),
                int(in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )  # cat

            self.reduce_conv1 = BaseConv(
                int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
            )
            self.C3_p3 = CSPLayer(
                int(2 * in_channels[0] * width),
                int(in_channels[0] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )

            # bottom-up conv
            self.bu_conv2 = Conv(
                int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
            )
            self.C3_n3 = CSPLayer(
                int(2 * in_channels[0] * width),
                int(in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )

            # bottom-up conv
            self.bu_conv1 = Conv(
                int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
            )
            self.C3_n4 = CSPLayer(
                int(2 * in_channels[1] * width),
                int(in_channels[2] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )

    def forward(self, input, return_base_feat=False, run_fpn=True):
        """
        Args:
            inputs: input images.
            return_base_feat: whether to return the original backbone feature

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        # out_features = self.backbone(input)
        # features = [out_features[f] for f in self.in_features]
        # [x2, x1, x0] = features
        if self.checkpoint_whole_backbone:
            (x2, x1, x0) = checkpoint.checkpoint(self.backbone, input)
        else:
            [x2, x1, x0] = self.backbone(input)
        if run_fpn:
            if self.width != 1:
                x2_adj, x1_adj, x0_adj = self.adjust2(x2), self.adjust1(x1), self.adjust0(x0)
            else:
                x2_adj, x1_adj, x0_adj = x2, x1, x0

            fpn_out0 = self.lateral_conv0(x0_adj)  # 1024->512/32
            f_out0 = self.upsample(fpn_out0)  # 512/16
            f_out0 = torch.cat([f_out0, x1_adj], 1)  # 512->1024/16
            f_out0 = self.C3_p4(f_out0)  # 1024->512/16

            fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
            f_out1 = self.upsample(fpn_out1)  # 256/8
            f_out1 = torch.cat([f_out1, x2_adj], 1)  # 256->512/8
            pan_out2 = self.C3_p3(f_out1)  # 512->256/8

            p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
            p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
            pan_out1 = self.C3_n3(p_out1)  # 512->512/16

            p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
            p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
            pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

            outputs = (pan_out2, pan_out1, pan_out0)
            if return_base_feat:
                return outputs, (x2, x1, x0)
            else:
                return outputs
        else:
            return (x2, x1, x0)