import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import cv2
from collections import OrderedDict
from .datasets_wrapper import Dataset
import copy
from unicorn.data import get_unicorn_datadir
class Got10k(Dataset):

    def __init__(self, root=None, split="vottrain", seq_ids=None, data_fraction=None,
    img_size=(416, 416)):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = os.path.join(get_unicorn_datadir(), "GOT10K")
        super().__init__(img_size)
        self.img_size = img_size

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            if split == 'train':
                file_path = os.path.join(cur_dir, 'data_specs', 'got10k_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(cur_dir, 'data_specs', 'got10k_val_split.txt')
            elif split == 'train_full':
                file_path = os.path.join(cur_dir, 'data_specs', 'got10k_train_full_split.txt')
            elif split == 'vottrain':
                file_path = os.path.join(cur_dir, 'data_specs', 'got10k_vot_train_split.txt')
            elif split == 'votval':
                file_path = os.path.join(cur_dir, 'data_specs', 'got10k_vot_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'got10k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True
    
    def get_num_sequences(self):
        return len(self.sequence_list)
    
    def _load_meta_info(self):
        def _read_meta(meta_info):

            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1],
                                       'motion_class': meta_info[6].split(': ')[-1],
                                       'major_class': meta_info[7].split(': ')[-1],
                                       'root_class': meta_info[8].split(': ')[-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1]})

            return object_meta
        sequence_meta_info = {}
        for s in self.sequence_list:
            try:
                with open(os.path.join(self.root, "train/%s/meta_info.ini")) as f:
                    meta_info = f.readlines()
                    sequence_meta_info[s] = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1],
                                            'motion_class': meta_info[6].split(': ')[-1],
                                            'major_class': meta_info[7].split(': ')[-1],
                                            'root_class': meta_info[8].split(': ')[-1],
                                            'motion_adverb': meta_info[9].split(': ')[-1]})
            except:
                sequence_meta_info[s] = OrderedDict({'object_class_name': None,
                                                     'motion_class': None,
                                                     'major_class': None,
                                                     'root_class': None,
                                                     'motion_adverb': None})
        return sequence_meta_info

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'train/list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(self.root, seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return gt

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(self.root, seq_path, "absence.label")
        cover_file = os.path.join(self.root, seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join("train", self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = torch.tensor(self._read_bb_anno(seq_path))
        height, width = self._get_frame(seq_path, 0).shape[:2]
        """xywh -> xyxy"""
        bbox_xyxy = copy.deepcopy(bbox)
        bbox_xyxy[:, 2] += bbox_xyxy[:, 0]
        bbox_xyxy[:, 3] += bbox_xyxy[:, 1]
        """clip within image"""
        bbox_xyxy[:, 0] = np.clip(bbox_xyxy[:, 0], 0, width)
        bbox_xyxy[:, 1] = np.clip(bbox_xyxy[:, 1], 0, height)
        bbox_xyxy[:, 2] = np.clip(bbox_xyxy[:, 2], 0, width)
        bbox_xyxy[:, 3] = np.clip(bbox_xyxy[:, 3], 0, height)
        """xyxy -> xywh"""
        bbox_new = copy.deepcopy(bbox_xyxy)
        bbox_new[:, 2] -= bbox_new[:, 0]
        bbox_new[:, 3] -= bbox_new[:, 1]
        """"""
        valid = (bbox_new[:, 2] > 32) & (bbox_new[:, 3] > 32)
        visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox_new, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return cv2.imread(os.path.join(self.root, self._get_frame_path(seq_path, frame_id)))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta

    def pull_item(self, idx, num_frames=2, allow_invisible=False):
        """idx is invalid"""
        valid = False
        while not valid:
            # Sample a sequence
            seq_id = random.randint(0, self.get_num_sequences() - 1)
            # Sample frames
            seq_info_dict = self.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            if sum(visible) > num_frames:
                valid = True
                min_id, max_id = 0, len(visible)
                if allow_invisible:
                    valid_ids = [i for i in range(min_id, max_id)]
                else:
                    valid_ids = [i for i in range(min_id, max_id) if visible[i]]
                frame_ids = random.sample(valid_ids, num_frames) # without replacement
                # get final results
                seq_path = self._get_sequence_path(seq_id)
                bboxes = self._read_bb_anno(seq_path)
                result = []
                for f_id in frame_ids:
                    ori_frame = self._get_frame(seq_path, f_id)
                    height, width = ori_frame.shape[:2]
                    ori_box = bboxes[f_id]
                    ori_box[2:] += ori_box[:2]
                    res = np.zeros((1, 5))
                    res[0, 0:4] = ori_box
                    res[0, 4] = 0
                    r = min(self.img_size[0] / height, self.img_size[1] / width)
                    res[:, :4] *= r # coordinates on the resized image
                    resized_img = cv2.resize(ori_frame, (int(width * r), int(height * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    result.append((resized_img, res))
                return tuple(result), seq_id, 0

    def pull_item_id(self, seq_id, obj_id, num_frames, allow_invisible=False):
        seq_info_dict = self.get_sequence_info(seq_id)
        visible = seq_info_dict['visible']
        min_id, max_id = 0, len(visible)
        if allow_invisible:
            valid_ids = [i for i in range(min_id, max_id)]
        else:
            valid_ids = [i for i in range(min_id, max_id) if visible[i]]
        try:
            frame_ids = random.sample(valid_ids, num_frames) # without replacement
        except:
            frame_ids = random.choices(valid_ids, k=num_frames) # with replacement
        # get final results
        seq_path = self._get_sequence_path(seq_id)
        bboxes = self._read_bb_anno(seq_path)
        result = []
        for f_id in frame_ids:
            ori_frame = self._get_frame(seq_path, f_id)
            height, width = ori_frame.shape[:2]
            ori_box = bboxes[f_id]
            ori_box[2:] += ori_box[:2]
            res = np.zeros((1, 5))
            res[0, 0:4] = ori_box
            res[0, 4] = 0
            r = min(self.img_size[0] / height, self.img_size[1] / width)
            res[:, :4] *= r # coordinates on the resized image
            resized_img = cv2.resize(ori_frame, (int(width * r), int(height * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            result.append((resized_img, res))
        return tuple(result)

    def __len__(self):
        return self.get_num_sequences()
