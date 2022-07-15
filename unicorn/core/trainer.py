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


import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from unicorn.data import DataPrefetcher, DataPrefetcherIns
from unicorn.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    dist,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
import random
import numpy as np
from copy import deepcopy

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        if self.exp.task == "uni":
            init_seeds(self.rank + 42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        if self.exp.debug_only:
            self.debug_data()
        self.train_in_epoch()
        self.after_train()
    
    def debug_data(self):
        import sys
        import numpy as np
        import cv2
        import torch.distributed as dist
        from PIL import Image
        import torch.nn.functional as F
        # data_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # data_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        save_dir = "debug"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx, (inps, targets, _, _, mask) in enumerate(self.train_loader):
            if idx >= 2:
                break
            # inps (bs, 3, H, W), targets (bs, M, 5), mask: (bs, M, H, W)
            mask = F.interpolate(mask, scale_factor=4, mode="bilinear", align_corners=False)
            bs = targets.size(0)
            inps_np = inps.cpu().numpy()
            mask_np = mask.cpu().numpy()
            for b in range(bs):
                # for i in range(2):
                cur_img = np.ascontiguousarray(inps_np[b].transpose((1, 2, 0)))
                v_mask = targets[b].sum(dim=-1)>0
                nt = v_mask.sum()
                v_targets = targets[b][v_mask].int()
                save_name = os.path.join(save_dir, "idx_%d_batch%d_rank_%d.jpg" %(idx, b, dist.get_rank()))
                save_img = cur_img.copy()
                for n in range(nt):
                    cx, cy, w, h = v_targets[n][1:5].tolist()
                    x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
                    cv2.rectangle(save_img, (x1, y1), (x2, y2), color=(0,0,255), thickness=2)
                cv2.imwrite(save_name, save_img)
                # mask
                if nt > 0:
                    cur_mask = mask_np[b].transpose((1, 2, 0))[:, :, :nt] # (H, W, M)
                    mask_vis = np.concatenate([np.zeros((cur_mask.shape[0], cur_mask.shape[1], 1)), cur_mask], axis=-1) # (H, W, C+1)
                    mask_indice = np.argmax(mask_vis, axis=-1).astype(np.uint8)
                    prj_root = "/opt/tiger/omnitrack"
                    palette = Image.open(os.path.join(prj_root, "data/DAVIS/Annotations/480p/blackswan/00000.png")).getpalette()
                    img_E = Image.fromarray(mask_indice)
                    img_E.putpalette(palette)
                    img_E.save(os.path.join(save_dir, "idx_%d_batch%d_rank_%d_mask.png" %(idx, b, dist.get_rank())))

        sys.exit(0)

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            if self.exp.task == "uni":
                self.train_in_iter_uni()
            else:
                self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_in_iter_uni(self):
        for self.iter, data in enumerate(self.train_loader):
            if len(data) == 4:
                inps, targets, task_ids, masks = data[0], data[1], data[2], data[3]
            elif len(data) == 3:
                inps, targets, task_ids, masks = data[0], data[1], data[2], None
            else:
                raise ValueError("len(data) should be 3 or 4")
            if self.iter < self.max_iter:
                self.before_iter()
                self.train_one_iter_uni(inps, targets, task_ids, masks)
                self.after_iter()

    def train_one_iter(self):
        torch.cuda.synchronize()
        iter_start_time = time.time()
        if self.exp.task == "det":
            inps, targets = self.prefetcher.next()
        elif self.exp.task == "inst":
            inps, targets, masks = self.prefetcher.next()
            masks = masks.to(self.data_type)
            masks.requires_grad = False
        else:
            raise ValueError()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        if self.exp.task == "inst":
            inps, targets, masks = self.exp.preprocess(inps, targets, self.input_size, masks)
        else:
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        torch.cuda.synchronize()
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            if self.exp.task == "det":
                outputs = self.model(inps, targets)
            elif self.exp.task == "inst":
                outputs = self.model(inps, targets, masks)
            else:
                outputs = self.actor(inps, targets)
        loss = outputs["total_loss"]
        if self.exp.use_grad_acc:
            loss /= self.exp.grad_acc_step
        if not self.exp.use_grad_acc:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.scaler.scale(loss).backward()
            if (self.iter + 1) % self.exp.grad_acc_step == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        # for k, v in self.model.named_parameters():
        #     if "controller" in k:
        #         print(k, v.grad)
        #     if "mask_branch" in k:
        #         print(k, v.grad)
        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            if "factor" in param_group:
                param_group["lr"] = lr * param_group["factor"]
            else:
                param_group["lr"] = lr
        torch.cuda.synchronize()
        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def train_one_iter_uni(self, inps, targets, task_ids, masks=None):
        torch.cuda.synchronize()
        iter_start_time = time.time()
        inps = inps.to(self.data_type).cuda()
        targets = targets.to(self.data_type).cuda()
        targets.requires_grad = False
        task_ids = task_ids.int().cuda()
        task_ids.requires_grad = False
        if masks is not None:
            masks = masks.to(self.data_type).cuda()
            masks.requires_grad = False
            inps_pro, targets_pro, masks_pro = self.exp.preprocess(inps, targets, self.input_size, masks=masks)
        else:
            inps_pro, targets_pro = self.exp.preprocess(inps, targets, self.input_size)
        torch.cuda.synchronize()
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            if masks is not None:
                outputs = self.actor(inps_pro, targets_pro, task_ids, masks=masks_pro)
            else:
                outputs = self.actor(inps_pro, targets_pro, task_ids)
        loss = outputs["total_loss"]

        if self.exp.use_grad_acc:
            loss = loss / self.exp.grad_acc_step
        if not self.exp.use_grad_acc:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.scaler.scale(loss).backward()
            if (self.iter + 1) % self.exp.grad_acc_step == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            if "factor" in param_group:
                param_group["lr"] = lr * param_group["factor"]
            else:
                param_group["lr"] = lr
        torch.cuda.synchronize()
        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        # )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        if self.exp.task == "det":
            logger.info("init prefetcher...")
            self.prefetcher = DataPrefetcher(self.train_loader)
        elif self.exp.task == "inst":
            logger.info("init prefetcher for instance segmentation...")
            self.prefetcher = DataPrefetcherIns(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)
        """Sync-BN"""
        if self.exp.task not in ["det", "inst"]:
            if self.exp.sync_bn:
                print("Using Sync-BN")
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
        if getattr(self.exp, "train_mask_only", False):
            model.backbone.eval() # fix backbone BN
            self.set_train_mask_only(model.head) # freeze head BN

        model_wo_ddp = deepcopy(model)
        if self.is_distributed:
            find_unused_parameters = getattr(self.exp, "find_unused_parameters", False)
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=find_unused_parameters)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model_wo_ddp, 0.9998, is_distributed=self.is_distributed)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        if getattr(self.exp, "train_mask_only", False):
            print("DON'T Switch the whole model to TRAIN mode")
        else:
            self.model.train()
        if self.exp.task == "det":
            self.evaluator = self.exp.get_evaluator(
                batch_size=self.args.batch_size, is_distributed=self.is_distributed, legacy=self.exp.normalize
            )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

        if self.exp.task == "uni":
            self.actor = self.exp.get_actor(self.model)
            logger.info("Actor is created.")

    def after_train(self):
        if self.exp.task == "det":
            logger.info(
                "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
            )
        logger.info("training is done.")

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            if self.exp.task == "det":
                self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")

            if self.is_distributed:
                if hasattr(self.model.module.head, "use_l1"):
                    self.model.module.head.use_l1 = True
            else:
                if hasattr(self.model.head, "use_l1"):
                    self.model.head.use_l1 = True
            if not getattr(self.exp, "disable_eval", False):
                self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        if self.exp.task == "det":
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                use_gn = getattr(self.exp, "use_gn", False)
                if not use_gn:
                    all_reduce_norm(self.model)
                if self.exp.task == "det":
                    self.evaluate_and_save_model()


    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )
        # alternate tasks
        step = getattr(self.exp, "alter_step", 10)
        if (self.progress_in_iter + 1) % step == 0:
            if hasattr(self.exp, "train_mode"):
                if self.exp.train_mode == "alter":
                    self.train_loader.dataset.dataset.alter_task()
                    # print("cur_task_id:", self.train_loader.dataset.dataset.cur_task_id)

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt
            if os.path.exists(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=self.device)
                # resume the model/optimizer state dict
                model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                # resume the training states variables
                start_epoch = (
                    self.args.start_epoch - 1
                    if self.args.start_epoch is not None
                    else ckpt["start_epoch"]
                )
                self.start_epoch = start_epoch
                self.best_ap = ckpt["best_ap"]
            else:
                self.start_epoch = 0
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()
        use_ap50 = getattr(self.exp, "use_ap50", False)
        if use_ap50:
            self.save_ckpt("last_epoch", ap50 > self.best_ap)
            self.best_ap = max(self.best_ap, ap50)
        else:
            self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
            self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model.module
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def set_train_mask_only(self, head):
        import torch.nn as nn
        for m in head.modules(): # fix head BN
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()
        # open BN for mask prediction
        head.controllers.train()
        head.mask_branch.train()
        head.mask_head.train()