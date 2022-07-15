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
import torch.distributed as dist
import torch.nn as nn
from .base_exp import BaseExp
from unicorn.data import get_unicorn_datadir
"""basic settings for object detector training"""
class ExpDet(BaseExp):
    def __init__(self):
        super().__init__()
        self.task = "det"
        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.0
        self.width = 1.0
        self.act = 'silu'
        self.backbone_name = "convnext"
        self.pretrained_name = "convnext_tiny_1k_224_ema.pth"
        self.in_channels = [192, 384, 768]
        self.use_gn = True
        self.use_attention = True
        self.n_layer_att = 3

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.train_name = "train2017"
        self.val_ann = "instances_val2017.json"
        self.val_name = "val2017"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
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
        self.normalize = False # whether to divide 255, substract mean, and divide std 
        self.data_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.data_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = 100
        self.warmup_lr = 0
        self.basic_lr_per_img = 1e-3 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.025
        self.ema = True
        self.always_l1 = False
        self.weight_decay = 5e-2
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.debug_only = False
        self.use_grad_acc = False # whether to use gradient accumulation
        self.grad_acc_step = 1
        self.use_checkpoint = False
        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.max_ins = None
        self.mask_thres = 0.3

    def get_model(self, load_pretrain=True):
        from unicorn.models import YOLOX, YOLOXHeadDet, YOLOPAFPNNEW

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            backbone = YOLOPAFPNNEW(self.depth, self.width, in_channels=self.in_channels, act=self.act, 
            backbone_name=self.backbone_name, use_checkpoint=self.use_checkpoint)
            head = YOLOXHeadDet(self.num_classes, self.width, in_channels=self.in_channels, act=self.act, 
            use_attention=self.use_attention, n_layer_att=self.n_layer_att)
            self.model = YOLOX(backbone, head)

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
                """load backbone pretrained model"""
                filename = self.pretrained_name
                ckpt_path = os.path.join(get_unicorn_datadir(), "..", filename)
                print("Loading pretrained backbone weights from %s" % ckpt_path)
                ckpt = torch.load(ckpt_path, map_location='cpu')
                missing_keys, unexpected_keys = self.model.backbone.backbone.load_state_dict(ckpt["model"], strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                del ckpt
                torch.cuda.empty_cache()
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from unicorn.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from unicorn.data.datasets import COCODataset, MosaicDetection
        from unicorn.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
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

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
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
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            param_dicts = [{"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}]
            print("trained parameters:", [n for n, p in self.model.named_parameters() if p.requires_grad])
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

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from unicorn.data import ValTransform
        from unicorn.data.datasets import COCODataset
        legacy = self.normalize
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            name=self.val_name if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
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