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
from .base_exp import BaseExp
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
"""
class ExpTrack(BaseExp):
    def __init__(self):
        super().__init__()
        self.task = "uni"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # ---------------- model config ---------------- #
        self.num_classes = 8
        self.depth = 1.0
        self.width = 1.0
        self.act = 'silu'
        self.use_gn = True
        # backbone
        self.backbone_name = "convnext"
        self.in_channels = [192, 384, 768]
        # embedding
        self.embed_dim = 128
        self.interact_mode = "deform"
        # head
        self.use_attention = True
        self.n_layer_att = 3
        self.unshared_obj = True
        self.unshared_reg = True
        self.fuse_method = "sum"
        self.learnable_fuse = True

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 0 # num_workers has to be 0 for alternative training
        self.input_size = (800, 1280)
        self.multiscale_range = 2
        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.train_name = "train2017"
        self.val_ann = "instances_val2017.json"
        self.val_name = "val2017"
        # --------------- transform config ----------------- #
        self.mosaic_prob = -1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        self.normalize = False # Depending on whether the COCO pretrained model uses normalization
        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = 15
        self.warmup_lr = 0
        self.basic_lr_per_img = 5e-4 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 3
        self.min_lr_ratio = 0.1
        self.ema = True
        self.mhs = True
        self.weight_decay = 5e-4
        self.print_interval = 15
        self.eval_interval = 10
        self.debug_only = False
        self.samples_per_epoch = 200000 # 10w for SOT, 10w for MOT
        self.sync_bn = False
        self.always_l1 = True
        self.use_grad_acc = True
        self.grad_acc_step = 2
        self.grid_sample = True
        self.bidirect = True
        self.train_mode = "alter"
        self.alter_step = 1
        self.mot_weight = 3
        self.scale_all_mot = True # scale all MOT loss term by mot_weight
        self.pretrain_name = "unicorn_det_convnext_tiny_800x1280"
        # -----------------  testing config ------------------ #
        self.test_size = (800, 1280)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.test_ann = "test.json" # evaluate on MOT Challenge 17 train/test
        self.test_name = "test"
        self.test_data_dir = os.path.join(get_unicorn_datadir(), "mot")
        # -----------------  other config ------------------ #
        self.sot_only = False
        self.mot_only = False
        self.mot_test_name = "bdd100k" # test dataset for mot. bdd100k or motchallenge

    def get_model(self, load_pretrain=True):
        from unicorn.models import Unicorn, YOLOPAFPNNEW, UnicornHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            backbone = YOLOPAFPNNEW(self.depth, self.width, in_channels=self.in_channels, act=self.act, backbone_name=self.backbone_name, \
                use_checkpoint=True)
            head = UnicornHead(self.num_classes, self.width, in_channels=self.in_channels, act=self.act, use_l1=self.always_l1, \
                use_attention=self.use_attention, n_layer_att=self.n_layer_att, unshared_obj=self.unshared_obj, unshared_reg=self.unshared_reg, \
                mot_weight=self.mot_weight, scale_all_mot=self.scale_all_mot, fuse_method=self.fuse_method, learnable_fuse=self.learnable_fuse)
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
        if load_pretrain:
            """Load COCO pretrained model"""
            filename = "Unicorn_outputs/%s/best_ckpt.pth" % self.pretrain_name
            ckpt_path = os.path.join(get_unicorn_datadir(), "..", filename)
            print("Loading COCO pretrained weights from %s" % ckpt_path)
            state_dict = torch.load(ckpt_path, map_location='cpu')["model"]
            # Deal with SOT and MOT head pretrained parameters
            new_state_dict = {}

            for k, v in state_dict.items():
                if not k.startswith("head."):
                    new_state_dict[k] = v
                else:
                    if k in ["head.cls_preds.0.weight", "head.cls_preds.0.bias", "head.cls_preds.1.weight", "head.cls_preds.1.bias",
                    "head.cls_preds.2.weight", "head.cls_preds.2.bias"]:
                        if self.num_classes == 8:
                            new_state_dict[k] = v[[0,0,2,7,5,6,3,1]] # [80] or [80, 256, 1, 1]
                        elif self.num_classes == 1:
                            new_state_dict[k] = v[0:1]
                        else:
                            raise ValueError("Invalid num_classes")
                    elif self.unshared_obj and k.startswith("head.obj_preds."):
                        k_new = k.replace("head.obj_preds.", "head.obj_preds_sot.")
                        new_state_dict[k] = v
                        new_state_dict[k_new] = v
                    elif self.unshared_reg and k.startswith("head.reg_preds."):
                        k_new = k.replace("head.reg_preds.", "head.reg_preds_sot.")
                        new_state_dict[k] = v
                        new_state_dict[k_new] = v
                    else:
                        new_state_dict[k] = v
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            del state_dict
            torch.cuda.empty_cache()
        return self.model
    
    def get_actor(self, model):
        from unicorn.models import UnicornActor
        return UnicornActor(model)

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from unicorn.data import TrainTransform_omni
        from unicorn.data.datasets import (MosaicDetectionUni, OmniDatasetPlus)
        from unicorn.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            if self.sot_only and (not self.mot_only):
                print("Training SOT only...")
                omni_dataset_sot = self.get_sot_dataset()
                omni_dataset_mot = None
                fix, fix_id = True, 0
            elif self.mot_only and (not self.sot_only):
                print("Training MOT only...")
                omni_dataset_sot = None
                omni_dataset_mot = self.get_mot_dataset()
                fix, fix_id = True, 1
            elif (not self.sot_only) and (not self.mot_only):
                print("Training both SOT and MOT...")
                omni_dataset_sot = self.get_sot_dataset()
                omni_dataset_mot = self.get_mot_dataset()
                fix, fix_id = False, None
            else:
                raise ValueError("self.sot_only and self.mot_only can not be simultaneously set to True")
            omni_dataset = OmniDatasetPlus(self.input_size, [omni_dataset_sot, omni_dataset_mot], \
                [1, 1], self.samples_per_epoch, mode=self.train_mode, fix=fix, fix_id=fix_id)
        # for SOT, the maximum number of targets is far less than 120
        dataset = MosaicDetectionUni(
            omni_dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform_omni(
                max_labels=100,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                legacy=self.normalize),
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

    def get_sot_dataset(self, cache_img=False):
        from unicorn.data import TrainTransform
        from unicorn.data.datasets import (COCOSOTDataset, Lasot, Got10k, TrackingNet, OmniDataset)
        coco_dataset = COCOSOTDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache_img,
        )
        lasot_dataset = Lasot(img_size=self.input_size)
        got10k_dataset = Got10k(img_size=self.input_size)
        trackingnet_dataset = TrackingNet(img_size=self.input_size)
        omni_dataset_sot = OmniDataset(self.input_size, [coco_dataset, lasot_dataset, got10k_dataset, trackingnet_dataset], \
            [1, 1, 1, 1], self.samples_per_epoch)
        return omni_dataset_sot
    
    def get_mot_dataset(self):
        from unicorn.data import TrainTransform
        from unicorn.data.datasets import (OmniDataset, BDDOmniDataset, MOTOmniDataset)
        if self.mot_test_name == "bdd100k":
            bdd_dataset = BDDOmniDataset(
                split="train",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=100,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
            )
            omni_dataset_mot = OmniDataset(self.input_size, [bdd_dataset], \
                [1, ], self.samples_per_epoch)
        elif self.mot_test_name == "motchallenge":
            """MOT17"""
            mot_dataset = MOTOmniDataset(data_dir="datasets/mot", json_file="train_omni.json",
                name="train", img_size=self.input_size, preproc=None)
            """CrowdHuman"""
            crowd_dataset = MOTOmniDataset(data_dir="datasets/crowdhuman", json_file="train.json",
                name="CrowdHuman_train", img_size=self.input_size, preproc=None)
            """CityPerson"""
            cityperson_dataset = MOTOmniDataset(data_dir="datasets", json_file="Cityscapes/annotations/train.json",
                name=None, img_size=self.input_size, preproc=None)
            """ETH"""
            eth_dataset = MOTOmniDataset(data_dir="datasets", json_file="ETHZ/annotations/train.json",
                name=None, img_size=self.input_size, preproc=None)
            omni_dataset_mot = OmniDataset(self.input_size, [mot_dataset, crowd_dataset, cityperson_dataset, eth_dataset], \
                [2, 6, 1, 1], self.samples_per_epoch)
        else:
            raise ValueError("Unsupported mot_test_name")
        return omni_dataset_mot

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            if inputs.ndim == 4:
                inputs = nn.functional.interpolate(
                    inputs, size=tsize, mode="bilinear", align_corners=False
                )
            elif inputs.ndim == 5:
                output_list = []
                for i in range(inputs.size(1)):
                    output_list.append(nn.functional.interpolate(inputs[:, i], size=tsize, mode="bilinear", align_corners=False))
                inputs = torch.stack(output_list, dim=1)
            targets[..., 1:5:2] = targets[..., 1:5:2] * scale_x
            targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            param_dicts = [{"params": [p for n, p in self.model.named_parameters()]}]
            optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=self.weight_decay)

            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from unicorn.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from unicorn.data.datasets import MOTDataset
        from unicorn.data import ValTransform

        valdataset = MOTDataset(
            data_dir=self.test_data_dir,
            json_file=self.test_ann,
            img_size=self.test_size,
            name=self.test_name,
            preproc=ValTransform(),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from unicorn.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)


def convert_bn_model_to_gn(module, num_groups=16):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`torch.nn.GroupNorm`.
    Args:
        module: your network module
        num_groups: num_groups of GN
    """
    mod = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        mod = nn.GroupNorm(num_groups, module.num_features,
                        eps=module.eps, affine=module.affine)
        # mod = nn.modules.linear.Identity()
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_bn_model_to_gn(
            child, num_groups=num_groups))
    del module
    return mod