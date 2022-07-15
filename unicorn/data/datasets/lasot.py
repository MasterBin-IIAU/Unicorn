import os
import os.path
import copy
import torch
import numpy as np
import pandas
import csv
import random
import cv2
from collections import OrderedDict
from .datasets_wrapper import Dataset
from unicorn.data import get_unicorn_datadir
class Lasot(Dataset):

    def __init__(self, root=None, vid_ids=None, split="train", data_fraction=None,
    img_size=(416, 416)):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = os.path.join(get_unicorn_datadir(), "LaSOT")
        super().__init__(img_size)
        self.img_size = img_size
        self.sequence_list = self._build_sequence_list(vid_ids, split)
        class_list = [seq_name.split('-')[0] for seq_name in self.sequence_list]
        self.class_list = []
        for ele in class_list:
            if ele not in self.class_list:
                self.class_list.append(ele)
        # Keep a list of all classes
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            if split == 'train':
                file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_specs", 'lasot_train_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'lasot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(self.root, seq_path, "groundtruth.txt")
        gt_arr = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return gt_arr

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(self.root, seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(self.root, seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(class_name, class_name + '-' + vid_id)

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
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox_new, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return cv2.imread(os.path.join(self.root, self._get_frame_path(seq_path, frame_id)))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

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
        # Sample frames
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
