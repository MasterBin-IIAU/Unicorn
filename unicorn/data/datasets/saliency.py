#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from .datasets_wrapper import Dataset
import random
from unicorn.data import get_unicorn_datadir

class SaliencyDataset(Dataset):

    def __init__(
        self,
        img_size=(416, 416),
        preproc=None,
        min_sz=0,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        self.data_dir = os.path.join(get_unicorn_datadir(), "saliency")
        self.min_sz = min_sz
        self.img_dir = os.path.join(self.data_dir, "image")
        self.mask_dir = os.path.join(self.data_dir, "mask")
        num_img = len(os.listdir(self.img_dir))
        num_mask = len(os.listdir(self.mask_dir))
        assert num_img == num_mask
        self.num_img = num_img
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return self.num_img

    def load_resized_img(self, index):
        img = self.load_image(index) # BGR Image
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        img_file = os.path.join(self.img_dir, "%08d.jpg" % (index+1))
        img = cv2.imread(img_file)
        assert img is not None
        return img

    def load_resized_mask(self, index):
        mask, res = self.load_mask(index)
        r = min(self.img_size[0] / mask.shape[0], self.img_size[1] / mask.shape[1])
        resized_mask = cv2.resize(
            mask,
            (int(mask.shape[1] * r), int(mask.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        # resize would transform (H, W, 1) to (H, W). So we need to manually add an axis
        if len(resized_mask.shape) == 2:
            resized_mask = resized_mask[:, :, None] # to (H, W, 1)
        # rescale the box
        res[:, :4] *= r
        return resized_mask, res

    def load_mask(self, index):
        mask_path = os.path.join(self.mask_dir, "%08d.png" %(index+1))
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
        mask_arr = mask[:, :, None].astype(np.float32) / 255 # (H, W, N)
        # generate box based on mask
        ret, binary = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        contour = contours[np.argmax(cnt_area)]  # use max area polygon
        polygon = contour.reshape(-1, 2)
        x1, y1, w, h = cv2.boundingRect(polygon)  # Min Max Rectangle
        num_objs = 1
        res = np.zeros((num_objs, 6))
        res[0, 0:4] = [x1, y1, x1+w, y1+h]
        res[0, 4] = 0 # cls
        res[0, 5] = 1 # track id
        return mask_arr, res

    def pull_item(self, idx, num_frames=2):
        """idx is invalid"""
        index = random.randint(0, self.__len__() - 1)
        img = self.load_resized_img(index) # resized image (without padding)
        mask_rsz, res = self.load_resized_mask(index) # resized mask (without padding) (H, W, N), N is the number of instances
        assert res.shape[0] == mask_rsz.shape[-1]
        return [(img, res, mask_rsz)] * 2, None, None
    

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
