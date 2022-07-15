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
import random
import torch
import torch.nn as nn
from .unicorn_track import ExpTrack, convert_bn_model_to_gn
from unicorn.models.deformable_transformer import build_deforamble_transformer
from unicorn.models.transformer_encoder import build_transformer_encoder
from unicorn.models.deformable_transformer import build_conv_interact
from unicorn.models.position_encoding import build_position_encoding
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from unicorn.data import get_unicorn_datadir
"""
The baseline setting of the ablation.
By default, we use BDD100K as the MOT training set.
Based on box-level pretrained weights, train VOS & MOTS.
"""
class ExpTrackMask(ExpTrack):
    def __init__(self):
        super().__init__()
        self.task = "uni"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # for segmentation
        self.ema = False
        self.ctrl_loc = "reg" # reg or cls
        self.sem_loss_on = False
        self.train_mask = True
        self.train_mask_only = True
        self.max_inst_coco_vos = 5 # max number of instances used in COCO-VOS dataset
        self.mhs = False # we don't use MOTS data to train VOS because the large domain gap
        self.use_raft = True # use the upsampling module in RAFT
        self.d_rate = 2 # downsample the original mask size by 2
        self.test_data_dir = os.path.join(get_unicorn_datadir(), "MOTS")


    def get_model(self, load_pretrain=True):
        from unicorn.models import Unicorn, YOLOPAFPNNEW, UnicornHeadMask

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            backbone = YOLOPAFPNNEW(self.depth, self.width, in_channels=self.in_channels, act=self.act, backbone_name=self.backbone_name, \
                use_checkpoint=True)
            head = UnicornHeadMask(self.num_classes, self.width, in_channels=self.in_channels, act=self.act, use_l1=self.always_l1, \
                use_attention=self.use_attention, n_layer_att=self.n_layer_att, unshared_obj=self.unshared_obj, unshared_reg=self.unshared_reg, \
                mot_weight=self.mot_weight, scale_all_mot=self.scale_all_mot, fuse_method=self.fuse_method, learnable_fuse=self.learnable_fuse,
                ctrl_loc=self.ctrl_loc, sem_loss_on=self.sem_loss_on, use_raft=self.use_raft, up_rate=8//self.d_rate)
            """additional modules"""
            if self.interact_mode == "deform":
                transformer = build_deforamble_transformer()
                pos_embed = build_position_encoding()
            elif self.interact_mode == "full":
                transformer = build_transformer_encoder()
                pos_embed = build_position_encoding()
            elif self.interact_mode == "conv":
                transformer = build_conv_interact()
                pos_embed = None
            else:
                raise ValueError("unsupported interact mode")
            self.model = Unicorn(backbone, head, pos_embed, transformer, bidirect=self.bidirect, grid_sample=self.grid_sample, mhs=self.mhs, \
                embed_dim=self.embed_dim, scale_all_mot=self.scale_all_mot, mot_weight=self.mot_weight, interact_mode=self.interact_mode)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        if self.backbone_name == "resnet50":
            import copy
            # for CNN backbones with BN, we keep using BN for simplicity
            backbone_ori = copy.deepcopy(self.model.backbone.backbone)
            if self.use_gn:
                self.model = convert_bn_model_to_gn(self.model, num_groups=16)
            self.model.backbone.backbone = backbone_ori
        else:
            if self.use_gn:
                self.model = convert_bn_model_to_gn(self.model, num_groups=16)
        """only train the mask branch and the controller (freeze all other parameters)"""
        self.model.requires_grad_(False)
        self.model.head.controllers.requires_grad_(True)
        self.model.head.mask_branch.requires_grad_(True)
        if load_pretrain:
            """Load pretrained model"""
            filename = "Unicorn_outputs/%s/latest_ckpt.pth"%self.pretrain_name
            ckpt_path = os.path.join(get_unicorn_datadir(), "..", filename)
            print("Loading pretrained weights from %s" % ckpt_path)
            state_dict = torch.load(ckpt_path, map_location='cpu')["model"]
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            del state_dict
            torch.cuda.empty_cache()
        return self.model


    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from unicorn.data import TrainTransform_Ins
        from unicorn.data.datasets import (MosaicDetectionUni, OmniDatasetPlus)
        from unicorn.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            if self.sot_only and (not self.mot_only):
                print("Training VOS only...")
                omni_dataset_vos = self.get_sot_dataset()
                omni_dataset_mots = None
                fix, fix_id = True, 0
            elif self.mot_only and (not self.sot_only):
                print("Training MOTS only...")
                omni_dataset_vos = None
                omni_dataset_mots = self.get_mot_dataset()
                fix, fix_id = True, 1
            elif (not self.sot_only) and (not self.mot_only):
                print("Training both VOS and MOTS...")
                omni_dataset_vos = self.get_sot_dataset()
                omni_dataset_mots = self.get_mot_dataset()
                fix, fix_id = False, None
            else:
                raise ValueError("self.sot_only and self.mot_only can not be simultaneously set to True")
            omni_dataset = OmniDatasetPlus(self.input_size, [omni_dataset_vos, omni_dataset_mots], \
                [1, 1], self.samples_per_epoch, mode=self.train_mode, fix=fix, fix_id=fix_id)

        dataset = MosaicDetectionUni(
            omni_dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform_Ins(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob, d_rate=1/self.d_rate),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            has_mask=True
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        # sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        # batch_sampler = YoloBatchSampler(
        #     sampler=sampler,
        #     batch_size=batch_size,
        #     drop_last=False,
        #     mosaic=not no_aug,
        # )

        # dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        # dataloader_kwargs["batch_sampler"] = batch_sampler

        # # Make sure each process has different random seed, especially for 'fork' method.
        # # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        # dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        # train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        train_sampler = DistributedSampler(dataset) if is_distributed else None
        shuffle = False if is_distributed else True
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, 
        num_workers=self.data_num_workers, drop_last=True, sampler=train_sampler)
        return train_loader

    def get_sot_dataset(self):
        from unicorn.data.datasets import COCOMOTSDataset, SaliencyDataset, DAVISDataset, YoutubeVOSDataset, OmniDataset
        coco_dataset = COCOMOTSDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            cat_names_in_coco=None, cat_names_full=None, max_inst=self.max_inst_coco_vos)
        saliency_dataset = SaliencyDataset(img_size=self.input_size)
        davis_dataset = DAVISDataset(img_size=self.input_size)
        youtube_dataset = YoutubeVOSDataset(img_size=self.input_size)
        omni_dataset_vos = OmniDataset(self.input_size, [coco_dataset, saliency_dataset, davis_dataset, youtube_dataset], \
            [1, 1, 1, 1], self.samples_per_epoch)
        return omni_dataset_vos
    
    def get_mot_dataset(self):
        from unicorn.data import TrainTransform_Ins
        from unicorn.data.datasets import (OmniDataset, BDDOmniMOTSDataset, COCOMOTSDataset, MOTSMOTDataset)
        if self.mot_test_name == "bdd100k":
            bdd_dataset = BDDOmniMOTSDataset(
                split="train",
                img_size=self.input_size,
                preproc=None,
            )
            omni_dataset_mots = OmniDataset(self.input_size, [bdd_dataset], \
                [1, ], self.samples_per_epoch)
        elif self.mot_test_name == "motchallenge":
            """COCO Instance Segmentation (person) and MOTS"""
            coco_dataset_person = COCOMOTSDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.train_name,
                img_size=self.input_size,
                preproc=TrainTransform_Ins(max_labels=100, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                cat_names_in_coco=["person"], cat_names_full=["person"],
                )
            mots_dataset = MOTSMOTDataset(img_size=self.input_size, preproc=None)
            omni_dataset_mots = OmniDataset(self.input_size, [coco_dataset_person, mots_dataset], \
                [1, 1], self.samples_per_epoch)
        else:
            raise ValueError("Unsupported mot_test_name")
        return omni_dataset_mots

    def preprocess(self, inputs, targets, tsize, masks=None):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            if inputs.ndim == 4:
                inputs = nn.functional.interpolate(
                    inputs, size=tsize, mode="bilinear", align_corners=False
                )
                if masks is not None:
                    masks = nn.functional.interpolate(
                        masks, size=(tsize[0]//self.d_rate, tsize[1]//self.d_rate), mode="bilinear", align_corners=False)
            elif inputs.ndim == 5:
                output_list = []
                for i in range(inputs.size(1)):
                    output_list.append(nn.functional.interpolate(inputs[:, i], size=tsize, mode="bilinear", align_corners=False))
                inputs = torch.stack(output_list, dim=1)
                output_mask_list = []
                for i in range(inputs.size(1)):
                    output_mask_list.append(nn.functional.interpolate(masks[:, i], size=(tsize[0]//self.d_rate, tsize[1]//self.d_rate), mode="bilinear", align_corners=False))
                masks = torch.stack(output_mask_list, dim=1)
            targets[..., 1:5:2] = targets[..., 1:5:2] * scale_x
            targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
        if masks is not None:
            return inputs, targets, masks
        else:
            return inputs, targets

