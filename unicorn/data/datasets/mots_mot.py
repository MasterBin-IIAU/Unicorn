#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import cv2
import numpy as np

from ..dataloading import get_unicorn_datadir
from .datasets_wrapper import Dataset


import random
"""22.01.17 MOT Challenge MOTS dataset (for finetuning MOTS)"""
import pycocotools.mask as rletools
from copy import deepcopy


class SegmentedObject:
    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


def load_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")
            try:
                frame = int(fields[0])
            except:
                raise Exception("<exc>Error in {} in line: {}<!exc>".format(path.split("/")[-1], line))
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                raise Exception("<exc>Multiple objects with track id " + fields[1] + " in frame " + fields[0] + "<!exc>")
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                raise Exception( "<exc>Unknown object class " + fields[2] + "<!exc>")
            if class_id == 10:
                continue
            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
                raise Exception( "<exc>Objects with overlapping masks in frame " + fields[0] + "<!exc>")
            else:
                combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


class MOTSMOTDataset(Dataset):
    """
    MOTS dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        img_size=(416, 416),
        preproc=None,
        min_sz=0,
        max_interval=30
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
        if data_dir is None:
            data_dir = os.path.join(get_unicorn_datadir(), "MOTS")
        self.data_dir = os.path.join(data_dir, "train")
        self.min_sz = min_sz
        self.max_interval = max_interval
        self.imgs = None
        self.img_size = img_size
        self.preproc = preproc
        # load image paths and rle masks
        self.data_dict = {}
        seq_list = os.listdir(self.data_dir)
        for seq_id, seq_name in enumerate(seq_list):
            self.data_dict[seq_id] = {}
            gt_path = os.path.join(self.data_dir, seq_name, "gt/gt.txt")
            self.data_dict[seq_id]["seq_name"] = seq_name
            self.data_dict[seq_id]["label"] = load_txt(gt_path)

    def __len__(self):
        return len(self.data_dict.keys())

    def __del__(self):
        del self.imgs

    def pull_item(self, idx, num_frames=2):
        valid = False
        while not valid:
            seq_id = random.sample(self.data_dict.keys(), k=1)[0]
            cur_seq_data = self.data_dict[seq_id]["label"]
            seq_name = self.data_dict[seq_id]["seq_name"]
            frame_ids = cur_seq_data.keys()
            if self.max_interval is not None:
                ref_frame_id = int(random.sample(frame_ids, k=1)[0])
                interval = random.randint(1, self.max_interval)
                cur_frame_id = ref_frame_id + interval
                if cur_frame_id not in frame_ids:
                    cur_frame_id = ref_frame_id - interval
            else:
                ref_frame_id, cur_frame_id = random.sample(frame_ids, k=2)
                ref_frame_id = int(ref_frame_id)
                cur_frame_id = int(cur_frame_id)
            result_ref = self.load_img_res_mask(seq_name, ref_frame_id, cur_seq_data)
            result_cur = self.load_img_res_mask(seq_name, cur_frame_id, cur_seq_data)
            valid = True
        return [result_ref, result_cur], None, None
    
    def resize_img_mask(self, img, res, mask):
        r = min(self.img_size[0] / mask.shape[0], self.img_size[1] / mask.shape[1])
        """mask"""
        resized_mask = cv2.resize(
            mask,
            (int(mask.shape[1] * r), int(mask.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        # resize would transform (H, W, 1) to (H, W). So we need to manually add an axis
        if len(resized_mask.shape) == 2:
            resized_mask = resized_mask[:, :, None] # to (H, W, 1)
        """img"""
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        """res"""
        res_rsz = deepcopy(res)
        res_rsz[:, :4] *= r
        return resized_img, res_rsz, resized_mask
    
    def load_img_res_mask(self, seq_name, frame_id, cur_seq_data):
        file_name_r = os.path.join(self.data_dir, seq_name, "img1/%06d.jpg" %frame_id)
        img = cv2.imread(file_name_r)
        ref_frame_data = cur_seq_data[frame_id]
        num_obj_r = len(ref_frame_data)
        res = np.zeros((num_obj_r, 6))
        mask_list = []
        for obj_idx, obj in enumerate(ref_frame_data):
            x, y, w, h = rletools.toBbox(obj.mask)
            res[obj_idx, :4] = [x, y, x+w, y+h]
            res[obj_idx, 4] = 0
            res[obj_idx, 5] = obj.track_id % 1000
            mask_list.append(rletools.decode(obj.mask))
        masks = np.stack(mask_list, axis=-1).astype(np.float32) # (H, W, N)
        return self.resize_img_mask(img, res, masks)