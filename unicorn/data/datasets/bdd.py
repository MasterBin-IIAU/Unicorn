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

class BDDDataset(Dataset):
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
        config_file = os.path.join(qdtrack_root, "configs/bdd100k/unicorn.py")
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

    def __len__(self):
        return len(self.datasets)
    
    def load_anno(self, index):
        data = self.datasets[index]
        ref_gt_bboxes = data["ref_gt_bboxes"].data.numpy() # bounding boxes (N, 4)
        ref_gt_labels = data["ref_gt_labels"].data.unsqueeze(-1).numpy() # class labels (N, 1)
        res = np.concatenate([ref_gt_bboxes, ref_gt_labels], axis=-1) # (N, 5)
        return res
    
    def pull_item(self, index):
        data = self.datasets[index]
        ref_norm = data["ref_img_metas"].data["img_norm_cfg"]
        mean, std = ref_norm["mean"].reshape(3, 1, 1), ref_norm["std"].reshape(3, 1, 1)
        # ref frame
        ref_img = (data["ref_img"].data.numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
        ref_img = np.ascontiguousarray(ref_img)
        ref_gt_bboxes = data["ref_gt_bboxes"].data.numpy() # bounding boxes (N, 4)
        ref_gt_labels = data["ref_gt_labels"].data.unsqueeze(-1).numpy() # class labels (N, 1)
        res = np.concatenate([ref_gt_bboxes, ref_gt_labels], axis=-1) # (N, 5)
        # get img_info and id
        height, width = ref_img.shape[:-1]
        img_info = (height, width)
        return ref_img, res.copy(), img_info, np.array([index])
    
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
