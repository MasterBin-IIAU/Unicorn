#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from .datasets_wrapper import Dataset
import random
import glob
from PIL import Image
from unicorn.data import get_unicorn_datadir

class DAVISDataset(Dataset):

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
        self.data_dir = os.path.join(get_unicorn_datadir(), "DAVIS")
        self.min_sz = min_sz
        resolution = "480p"
        imset='2017/train.txt'
        self.mask_dir = os.path.join(self.data_dir, 'Annotations', resolution)
        self.mask480_dir = os.path.join(self.data_dir, 'Annotations', '480p')
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages', resolution)
        _imset_dir = os.path.join(self.data_dir, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))

        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.videos)

    def load_resized_img(self, img_file):
        img = self.load_image(img_file) # BGR Image
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, img_file):
        img = cv2.cvtColor(np.array(Image.open(img_file).convert('RGB')), cv2.COLOR_RGB2BGR) 
        assert img is not None
        return img

    def load_resized_mask(self, mask_file):
        mask, res = self.load_mask(mask_file)
        if len(res) == 0:
            return mask, res
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

    def load_mask(self, mask_file):
        mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        num_objs = np.max(mask)
        res = np.zeros((num_objs, 6))
        K = num_objs + 1
        M = np.zeros((mask.shape[0], mask.shape[1], K), dtype=np.uint8) # (H, W, num_obj+1)
        for k in range(K):
            M[:, :, k] = (mask == k).astype(np.float32)
            if k > 0:
                y_t, x_t = np.nonzero(M[:, :, k])
                if len(x_t) > 1:
                    x1, x2, y1, y2 = np.min(x_t), np.max(x_t), np.min(y_t), np.max(y_t)
                    res[k-1, 0:4] = [x1, y1, x2, y2]
                    res[k-1, 4] = 0 # cls
                    res[k-1, 5] = k # track id
        return M[:, :, 1:], res

    def pull_item(self, idx, num_frames=2):
        index = random.randint(0, self.__len__() - 1)
        video = self.videos[index]
        valid = False
        while not valid:
            frame_list = sorted(random.sample(range(self.num_frames[video]), num_frames))
            result_list = []
            for f in range(num_frames):
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame_list[f]))
                mask_rsz, res = self.load_resized_mask(mask_file)
                assert res.shape[0] == mask_rsz.shape[-1]
                if res.shape[0] == 0:
                    valid = False
                    break
                else:
                    valid = True
                img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame_list[f]))
                img = self.load_resized_img(img_file)
                result_list.append((img, res, mask_rsz))
        return result_list, None, None
    

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
