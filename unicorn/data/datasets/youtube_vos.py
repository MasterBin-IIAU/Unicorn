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

class YoutubeVOSDataset(Dataset):

    def __init__(
        self,
        img_size=(416, 416),
        preproc=None,
        min_sz=0,
        box_only=False
    ):
        super().__init__(img_size)
        self.data_dir = os.path.join(get_unicorn_datadir(), "ytbvos18/train")
        self.min_sz = min_sz
        self.mask_dir = os.path.join(self.data_dir, 'Annotations')
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')

        self.videos = [i.split('/')[-1] for i in glob.glob(os.path.join(self.image_dir, '*'))]
        self.num_frames = {}
        self.img_files = {}
        self.mask_files = {}
        for _video in self.videos:
            tmp_imgs = glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))
            tmp_masks = glob.glob(os.path.join(self.mask_dir, _video, '*.png'))
            tmp_imgs.sort()
            tmp_masks.sort()
            self.img_files[_video] = tmp_imgs
            self.mask_files[_video] = tmp_masks
            self.num_frames[_video] = len(tmp_imgs)

        self.img_size = img_size
        self.preproc = preproc
        self.box_only = box_only

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
            idx_list = []
            for f in range(num_frames):
                mask_file = self.mask_files[video][frame_list[f]]
                mask_rsz, res = self.load_resized_mask(mask_file)
                assert res.shape[0] == mask_rsz.shape[-1]
                if res.shape[0] == 0:
                    valid = False
                    break
                else:
                    valid = True
                img_file = self.img_files[video][frame_list[f]]
                img = self.load_resized_img(img_file)
                result_list.append((img, res, mask_rsz))
            # double check whether the track ids are matched
            if valid:
                matched = False
                track_ids_r = result_list[0][1][:, 5] # (M, )
                track_ids_c = result_list[1][1][:, 5] # (N, )
                for i, tid_r in enumerate(track_ids_r):
                    if tid_r == 0:
                        continue
                    for j, tid_c in enumerate(track_ids_c):
                        if tid_c == 0:
                            continue
                        if tid_r == tid_c:
                            matched = True
                            idx_list.append((i, j))
                            break
                    if matched:
                        break
                if not matched:
                    valid = False
        if self.box_only:
            track_ids_r = result_list[0][1][:, 5] # (M, )
            track_ids_c = result_list[1][1][:, 5] # (N, )
            (i, j) = random.choice(idx_list)
            result_list_box = [[None, None], [None, None]]
            result_list_box[0][0] = result_list[0][0]
            result_list_box[1][0] = result_list[1][0]
            result_list_box[0][1] = result_list[0][1][i:i+1]
            result_list_box[1][1] = result_list[1][1][j:j+1]
            result_list_box[0] = tuple(result_list_box[0])
            result_list_box[1] = tuple(result_list_box[1])
            return result_list_box, None, None
        else:
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
