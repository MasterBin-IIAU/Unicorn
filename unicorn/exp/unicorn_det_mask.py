#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------
# Unicorn
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# ----------------------------------------------------------------------


import os
import torch
import torch.nn as nn
import torch.distributed as dist
from .unicorn_det import ExpDet, convert_bn_model_to_gn
from unicorn.data import get_unicorn_datadir
"""
Instance segmentation (baseline setting)
"""
class ExpDetMask(ExpDet):
    def __init__(self):
        super().__init__()
        self.task = "inst"
        # ---------------- model config ---------------- #
        self.ctrl_loc = "reg" # reg or cls
        self.use_raft = True
        self.d_rate = 2
        self.pretrain_name = "unicorn_det_convnext_tiny_800x1280"
        # ---------------- dataloader config ---------------- #
        self.input_size = (800, 1280)
        # --------------- transform config ----------------- #
        self.mosaic_prob = -1.0 # for instance segmentation, we disable mosaic augmentation
        # --------------  training config --------------------- #
        self.max_epoch = 30
        self.no_aug_epochs = 3
        self.ema = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.sem_loss_on = False
        self.train_mask_only = True
        # -----------------  testing config ------------------ #
        self.test_size = (800, 1280)


    def get_model(self, load_pretrain=True):
        from unicorn.models import YOLOXMask, YOLOPAFPNNEW, YOLOXHeadDetMask

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            backbone = YOLOPAFPNNEW(self.depth, self.width, in_channels=self.in_channels, act=self.act, backbone_name=self.backbone_name)
            head = YOLOXHeadDetMask(self.num_classes, self.width, in_channels=self.in_channels, act=self.act, ctrl_loc=self.ctrl_loc, sot_mode=True, \
                sem_loss_on=self.sem_loss_on, use_attention=self.use_attention, n_layer_att=self.n_layer_att, use_raft=self.use_raft, up_rate=8//self.d_rate)
            self.model = YOLOXMask(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        if self.use_gn:
            self.model = convert_bn_model_to_gn(self.model, num_groups=16)
        """only train the mask branch and the controller (freeze all other parameters)"""
        self.model.requires_grad_(False)
        self.model.head.controllers.requires_grad_(True)
        self.model.head.mask_branch.requires_grad_(True)
        if load_pretrain:
            """load backbone pretrained model"""
            filename = "Unicorn_outputs/%s/best_ckpt.pth" % self.pretrain_name
            ckpt_path = os.path.join(get_unicorn_datadir(), "..", filename)
            print("Loading pretrained weights from %s" % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(ckpt["model"], strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            del ckpt
            torch.cuda.empty_cache()
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from unicorn.data import (
            TrainTransform_Ins,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from unicorn.data.datasets import COCOInsDataset, MosaicDetectionIns
        from unicorn.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCOInsDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.train_name,
                img_size=self.input_size,
                preproc=None,
                cache=cache_img,
            )

        dataset = MosaicDetectionIns(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform_Ins(
                max_labels=100,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                legacy=self.normalize,
                d_rate=1/self.d_rate),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def preprocess(self, inputs, targets, tsize, masks=None):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
        # deal with masks
        if masks is not None:
            masks = nn.functional.interpolate(
            masks, size=(tsize[0]//self.d_rate, tsize[1]//self.d_rate), mode="bilinear", align_corners=False
        )
            return inputs, targets, masks
        else:
            return inputs, targets

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from unicorn.evaluators import COCOInstEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOInstEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            max_ins=self.max_ins,
            mask_thres=self.mask_thres,
            d_rate=self.d_rate,
            use_raft=self.use_raft
        )
        return evaluator