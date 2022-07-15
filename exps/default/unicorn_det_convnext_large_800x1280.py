#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2022 ByteDance. All Rights Reserved.
import os
from unicorn.exp import ExpDet

class Exp(ExpDet):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.backbone_name = "convnext_large"
        self.pretrained_name = "convnext_large_22k_224.pth"
        self.in_channels = [384, 768, 1536]
        self.use_checkpoint = True
        self.input_size = (800, 1280)
        self.test_size = (800, 1280)