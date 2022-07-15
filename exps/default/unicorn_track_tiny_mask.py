#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2022 ByteDance. All Rights Reserved.
import os
from unicorn.exp import ExpTrackMask
"""
The baseline setting used in the ablation
We load weights pretrained on COCO with input resolution of 800x1280
"""
class Exp(ExpTrackMask):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.pretrain_name = "unicorn_track_tiny"
        

