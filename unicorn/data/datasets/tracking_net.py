import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict
from unicorn.data import get_unicorn_datadir
from .datasets_wrapper import Dataset
import cv2
import copy

def list_sequences(root, set_ids):
    """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)
    args:
        root: Root directory to TrackingNet
        set_ids: Sets (0-11) which are to be used
    returns:
        list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
    """
    sequence_list = []

    for s in set_ids:
        anno_dir = os.path.join(root, "TRAIN_" + str(s), "anno")

        sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]
        sequence_list += sequences_cur_set

    return sequence_list


class TrackingNet(Dataset):
    """ TrackingNet dataset.
    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf
    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """
    def __init__(self, set_ids=None, data_fraction=None, img_size=(416, 416)):
        """
        args:
            root        - The path to the TrackingNet folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            set_ids (None) - List containing the ids of the TrackingNet sets to be used for training. If None, all the
                            sets (0 - 11) will be used.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = os.path.join(get_unicorn_datadir(), "TrackingNet")
        super().__init__(img_size)
        self.img_size = img_size

        if set_ids is None:
            set_ids = [i for i in range(4)] # Here we only 4 of the 11

        self.set_ids = set_ids

        # Keep a list of all videos. Sequence list is a list of tuples (set_id, video_name) containing the set_id and
        # video_name for each sequence
        self.sequence_list = list_sequences(self.root, self.set_ids)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_to_class_map, self.seq_per_class = self._load_class_info()

        # we do not have the class_lists for the tracking net
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def _load_class_info(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        class_map_path = os.path.join(cur_dir, 'data_specs', 'trackingnet_classmap.txt')

        with open(class_map_path, 'r') as f:
            seq_to_class_map = {seq_class.split('\t')[0]: seq_class.rstrip().split('\t')[1] for seq_class in f}

        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = seq_to_class_map.get(seq[1], 'Unknown')
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_to_class_map, seq_per_class

    def get_name(self):
        return 'trackingnet'

    def has_class_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = seq_path.replace("frames", "anno") + ".txt"
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

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
        visible = valid.clone().byte()
        return {'bbox': bbox_new, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        frame_path = os.path.join(seq_path, str(frame_id) + ".jpg")
        return cv2.imread(frame_path)
    
    def _get_sequence_path(self, seq_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        return os.path.join(self.root, "TRAIN_" + str(set_id), "frames", vid_name)

    def _get_class(self, seq_id):
        seq_name = self.sequence_list[seq_id][1]
        return self.seq_to_class_map[seq_name]

    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class

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
    def get_num_sequences(self):
        return len(self.sequence_list)

    def __len__(self):
        return self.get_num_sequences()