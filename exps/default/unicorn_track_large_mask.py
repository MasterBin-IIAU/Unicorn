#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2022 ByteDance. All Rights Reserved.
import os
from unicorn.exp import ExpTrackMask
"""
The main setting used in the Unicorn paper (ConvNext-Large Backbone)
We load weights pretrained on COCO with input resolution of 800x1280
"""
class Exp(ExpTrackMask):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.backbone_name = "convnext_large"
        self.in_channels = [384, 768, 1536]
        self.pretrain_name = "unicorn_track_large"
