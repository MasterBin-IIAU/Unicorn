import numpy as np
from .datasets_wrapper import Dataset
import os
from mmcv import Config
from mmdet.datasets import build_dataset
import sys

qdtrack_root = "external/qdtrack"
sys.path.append(qdtrack_root)
from qdtrack import __version__
import numpy as np
from qdtrack.datasets import BDDVideoDataset # keep this line
import random
"""22.01.18 BDD100K MOTS (instance segmentation + MOTS)"""
class BDDOmniMOTSDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        split,
        img_size=(608, 1088),
        preproc=None,
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
        config_file = os.path.join(qdtrack_root, "configs/bdd100k/segtrack-frcnn_r50_fpn_12e_bdd10k.py")
        cfg = Config.fromfile(config_file)
        assert split in ["train", "val", "test"]
        self.split = split
        if split == "train":
            self.datasets = build_dataset(cfg.data.train)
        elif split == "val":
            self.datasets = build_dataset(cfg.data.val)
        elif split == "test":
            cfg.data.test.test_mode = True
            self.datasets = build_dataset(cfg.data.test)
        self.img_size = img_size
        self.preproc = preproc
        self.class_ids = list(range(1, 9)) # 1,2,..,8
        self.len_dataset = len(self.datasets)

    def __len__(self):
        return len(self.datasets)
    
    def load_anno(self, index):
        data = self.datasets[index]
        ref_gt_bboxes = data["ref_gt_bboxes"].data.numpy() # bounding boxes (N, 4)
        ref_gt_labels = data["ref_gt_labels"].data.unsqueeze(-1).numpy() # class labels (N, 1)
        res = np.concatenate([ref_gt_bboxes, ref_gt_labels], axis=-1) # (N, 5)
        return res
    
    def pull_item(self, idx, num_frames=2):
        index = random.randint(0, self.len_dataset-1)
        data = self.datasets[index]
        ref_norm = data["ref_img_metas"].data["img_norm_cfg"]
        mean, std = ref_norm["mean"].reshape(3, 1, 1), ref_norm["std"].reshape(3, 1, 1)
        # ref frame
        ref_img = (data["ref_img"].data.numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
        ref_img = np.ascontiguousarray(ref_img)
        ref_gt_bboxes = data["ref_gt_bboxes"].data.numpy() # bounding boxes (N, 4)
        ref_gt_labels = data["ref_gt_labels"].data.unsqueeze(-1).numpy() # class labels (N, 1)
        ref_gt_masks = data["ref_gt_masks"].data.to_ndarray().transpose((1, 2, 0)).astype(np.float32) # (N, H, W)
        # cur frame
        cur_img = (data["img"].data.numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
        cur_img = np.ascontiguousarray(cur_img)
        cur_gt_bboxes = data["gt_bboxes"].data.numpy() # bounding boxes (N, 4)
        cur_gt_labels = data["gt_labels"].data.unsqueeze(-1).numpy() # class labels (N, 1)
        cur_gt_masks = data["gt_masks"].data.to_ndarray().transpose((1, 2, 0)).astype(np.float32) # (H, W, N)
        # deal with trackids
        ref_gt_match_indices = data["ref_gt_match_indices"].data.numpy()
        cur_gt_match_indices = data["gt_match_indices"].data.numpy()
        n_ref, n_cur = len(ref_gt_bboxes), len(cur_gt_bboxes)
        ref_ids, cur_ids = np.arange(1, n_ref+1), np.zeros((n_cur, ))
        for i in range(n_ref):
            if ref_gt_match_indices[i] != -1:
                cur_ids[ref_gt_match_indices[i]] = i+1
        tmp_id = n_ref
        for j in range(n_cur):
            if cur_gt_match_indices[j] == -1:
                assert cur_ids[j] == 0
                tmp_id += 1
                cur_ids[j] = tmp_id
        ref_ids = ref_ids[..., np.newaxis] # (n_ref, 1)
        cur_ids = cur_ids[..., np.newaxis] # (n_cur, 1)
        # merge results
        ref_res = np.concatenate([ref_gt_bboxes, ref_gt_labels, ref_ids], axis=-1) # (N, 6)
        cur_res = np.concatenate([cur_gt_bboxes, cur_gt_labels, cur_ids], axis=-1) # (N, 6)
        return [(ref_img, ref_res, ref_gt_masks), (cur_img, cur_res, cur_gt_masks)], None, None
    
    def pull_item_test(self, index):
        data = self.datasets[index]
        norm = data["img_metas"][0].data["img_norm_cfg"]
        mean, std = norm["mean"].reshape(3, 1, 1), norm["std"].reshape(3, 1, 1)
        # current frame
        img = (data["img"][0].numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
        img = np.ascontiguousarray(img)
        # get the original size and the original image (without padding)
        ori_H, ori_W, _ = data["img_metas"][0].data["ori_shape"]
        img_ori = img[:ori_H, :ori_W, :]
        # get img_info and id
        img_info = (ori_H, ori_W)
        return img_ori, None, img_info, np.array([index])

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
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        if self.split == "test":
            img, target, img_info, img_id = self.pull_item_test(index)
        else:
            img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
