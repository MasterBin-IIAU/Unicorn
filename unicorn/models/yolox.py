#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# ------------------------------------------------------------------------
# Unicorn
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# ----------------------------------------------------------------------

import torch.nn as nn

# import time

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        # s = time.time()
        fpn_outs = self.backbone(x)
        # e_b = time.time()
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)
        # e_d = time.time()
        # print("backbone time: %.4f, head time: %.4f " % (e_b-s, e_d-e_b))
        return outputs

class YOLOXMask(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, masks=None):
        # fpn output content features of [dark3, dark4, dark5]
        # s = time.time()
        fpn_outs = self.backbone(x)
        # e_b = time.time()
        if self.training:
            assert targets is not None
            assert masks is not None
            loss_dict = self.head(fpn_outs, targets, x, masks)
            return loss_dict
        else:
            outputs = self.head(fpn_outs)
            # e_d = time.time()
            # print("backbone time: %.4f, head time: %.4f " % (e_b-s, e_d-e_b))
            return outputs
