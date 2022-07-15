#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

# Object Detection
from .losses import IOUloss
from .backbone.darknet import CSPDarknet, Darknet
from .backbone.yolo_pafpn_new import YOLOPAFPNNEW
from .yolo_head_det import YOLOXHeadDet
from .yolox import YOLOX
# Instance Segmentation
from .yolo_head_det_mask import YOLOXHeadDetMask
from .yolox import YOLOXMask
# Unified Tracking and Segmentation
from .unicorn_head import UnicornHead
from .unicorn_head_mask import UnicornHeadMask
from .unicorn import Unicorn, UnicornActor
