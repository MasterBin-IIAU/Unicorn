#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np

from unicorn.utils import adjust_box_anns, get_local_rank

from ..data_augment import box_candidates, random_perspective
from .datasets_wrapper import Dataset
import copy
import random
from .mosaicdetection import get_mosaic_coordinate, MosaicDetection

"""2021.10.13 modified mosaic dataset for SOT"""

"""2022.xx.xx MosaicDetectionUni class can be used by SOT-MOT or VOS-MOTS, but NOT for SOT-MOT-VOS=MOTS"""
class MosaicDetectionUni(MosaicDetection):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, perspective=0.0,
        enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0, flip_prob=0.5, 
        has_mask=False, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(dataset, img_size, mosaic=mosaic, preproc=preproc,
        degrees=degrees, translate=translate, mosaic_scale=mosaic_scale,
        mixup_scale=mixup_scale, shear=shear, perspective=perspective,
        enable_mixup=enable_mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob)

        self.flip_prob = flip_prob
        self.has_mask = has_mask

    def __len__(self):
        return len(self.dataset)

    def mosaic_idx2coord(self, idx, input_h, input_w, cy ,cx):
        # return x1-y1-x2-y2
        if idx==0:
            return (0, 0, cx, cy)
        elif idx==1:
            return (cx, 0, input_w, cy)
        elif idx==2:
            return (0, cy, cx, input_h)
        elif idx==3:
            return (cx, cy, input_w, input_h)
        else:
            raise ValueError("mosaic idx should be 0, 1, 2, 3")
    
    def fill_grid_with_patch(self, target_lbs, grid_xyxy, target_img, mosaic_img):
        img_h, img_w, _ = mosaic_img.shape
        # compute coordinates
        patch_h, patch_w = grid_xyxy[3]-grid_xyxy[1], grid_xyxy[2]-grid_xyxy[0]
        tx1, ty1, tx2, ty2, cls_ = target_lbs
        # we first try putting the target at the center of the grid
        tcx, tcy = (tx1+tx2)/2, (ty1+ty2)/2
        px1 = round(max(0, tcx - patch_w/2))
        py1 = round(max(0, tcy - patch_h/2))
        px2 = px1 + patch_w
        py2 = py1 + patch_h
        if px1>0 and py1>0 and px2<img_w and py2<img_h:
            # if the original patch are totally contained in the original image, we add a random displacement
            delta_x = int(random.uniform(tx2-px2, tx1-px1))
            delta_y = int(random.uniform(ty2-py2, ty1-py1))
            px1 = round(max(0, px1+delta_x))
            py1 = round(max(0, py1+delta_y))
            px2 = px1 + patch_w
            py2 = py1 + patch_h
        patch_img = target_img[py1:py2, px1:px2]
        patch_h, patch_w = patch_img.shape[:2]
        # fill
        gx1, gy1 = grid_xyxy[:2]
        mosaic_img[gy1: gy1+patch_h, gx1: gx1+patch_w] = patch_img
        # return the coordinates on the mosaic image
        # here we clip the boxes inside the grids
        tx1_new = min(max(grid_xyxy[0], tx1 - px1 + grid_xyxy[0]), grid_xyxy[2])
        ty1_new = min(max(grid_xyxy[1], ty1 - py1 + grid_xyxy[1]), grid_xyxy[3])
        tx2_new = min(max(grid_xyxy[0], tx2 - px1 + grid_xyxy[0]), grid_xyxy[2])
        ty2_new = min(max(grid_xyxy[1], ty2 - py1 + grid_xyxy[1]), grid_xyxy[3])
        return np.array([tx1_new, ty1_new, tx2_new, ty2_new, cls_])
    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            raise ValueError("For SOT-MOT joint training, we don't allow mosaic augmentation")
        else:
            self.dataset._input_dim = self.input_dim
            ori_data, task_id, _, _ = self.dataset.pull_item(idx)
            if self.has_mask:
                img_list, label_list, mask_list = [], [], []
                flip = False
                if random.random() < self.flip_prob:
                    flip = True
                for idx, (img, label, mask) in enumerate(ori_data):
                    img = img.astype(np.uint8)
                    img, label, mask = self.preproc(img, label, mask, self.input_dim, joint=True, flip=flip)
                    img_list.append(img)
                    label_list.append(label)
                    mask_list.append(mask)
                # (2, 3, H, W), (2, M, 5), (1, ), (2, M, H/4, W/4)
                return np.stack(img_list, axis=0), np.stack(label_list, axis=0), task_id, np.stack(mask_list, axis=0)
            else:
                img_list, label_list = [], []
                flip = False
                if random.random() < self.flip_prob:
                    flip = True
                for idx, (img, label) in enumerate(ori_data):
                    img = img.astype(np.uint8)
                    img, label = self.preproc(img, label, self.input_dim, joint=True, flip=flip)
                    img_list.append(img)
                    label_list.append(label)
                # (2, 3, H, W), (2, M, 5), (1, )
                return np.stack(img_list, axis=0), np.stack(label_list, axis=0), task_id

"""2022.02.07 MosaicDetectionUni4tasks is designed for SOT-MOT-VOS-MOTS"""
class MosaicDetectionUni4tasks(MosaicDetection):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, perspective=0.0,
        enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0, flip_prob=0.5, 
        *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(dataset, img_size, mosaic=mosaic, preproc=preproc,
        degrees=degrees, translate=translate, mosaic_scale=mosaic_scale,
        mixup_scale=mixup_scale, shear=shear, perspective=perspective,
        enable_mixup=enable_mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob)

        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.dataset)
    
    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            raise ValueError("For SOT-MOT-VOS-MOTS joint training, we don't allow mosaic augmentation")
        else:
            self.dataset._input_dim = self.input_dim
            ori_data, task_id, _, _ = self.dataset.pull_item(idx)
            img_list, label_list, mask_list = [], [], []
            flip = False
            if random.random() < self.flip_prob:
                flip = True
            for idx, data in enumerate(ori_data):
                if len(data) == 3:
                    img, label, mask = data[0], data[1], data[2]
                elif len(data) == 2:
                    img, label, mask = data[0], data[1], None
                else:
                    raise ValueError("len(data) has to be 2 or 3")
                img = img.astype(np.uint8)
                img, label, mask = self.preproc(img, label, mask, self.input_dim, joint=True, flip=flip)
                img_list.append(img)
                label_list.append(label)
                if mask is not None:
                    mask_list.append(mask)
            if len(mask_list) != 0:
                # (2, 3, H, W), (2, M, 5), (1, ), (2, M, H/4, W/4)
                return np.stack(img_list, axis=0), np.stack(label_list, axis=0), task_id, np.stack(mask_list, axis=0)
            else:
                # (2, 3, H, W), (2, M, 5), (1, )
                return np.stack(img_list, axis=0), np.stack(label_list, axis=0), task_id
