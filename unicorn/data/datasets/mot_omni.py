import cv2
import numpy as np
from pycocotools.coco import COCO

import os

from ..dataloading import get_unicorn_datadir
from .datasets_wrapper import Dataset
import json
import random
from copy import deepcopy

class MOTOmniDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None,
        max_interval=30,
        id_bias=None
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
            data_dir = os.path.join(get_unicorn_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file
        self.id_bias = id_bias
        if "annotations" in self.json_file:
            json_path = os.path.join(self.data_dir, self.json_file)
        else:
            json_path = os.path.join(self.data_dir, "annotations", self.json_file)
        mot_json = json.load(open(json_path, "r"))
        if ("images" in mot_json) and ("annotations" in mot_json):
            self.is_video = False
            self.coco = COCO(json_path)
            self.ids = self.coco.getImgIds()
            self.class_ids = sorted(self.coco.getCatIds())
            cats = self.coco.loadCats(self.coco.getCatIds())
            self._classes = tuple([c["name"] for c in cats])
            self.annotations = self._load_coco_annotations()
            self.video_list = list(range(len(self.annotations)))
        else:
            self.annotations = mot_json
            self.video_list = self.annotations.keys()
            self.is_video = True
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.max_interval = max_interval

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if not self.is_video:
                res[ix, 5] = ix + 1
            else:
                assert (obj["track_id"] != -1)
            # if self.id_bias is not None:
            #     res[ix, 5] = obj["track_id"] + self.id_bias
            # else:
            #     res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        if ("frame_id" in im_ann) and ("video_id" in im_ann):
            frame_id = im_ann["frame_id"]
            video_id = im_ann["video_id"]
            img_info = (height, width, frame_id, video_id, file_name)
        else:
            img_info = (height, width, file_name)
        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, file_name, res):
        if self.name is not None:
            img_file = os.path.join(self.data_dir, self.name, file_name)
        else:
            img_file = os.path.join(self.data_dir, file_name)
        img = cv2.imread(img_file)
        assert img is not None
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        # coordinates on the resized image
        res_rsz = deepcopy(res)
        res_rsz[:, :4] *= r
        return (resized_img, res_rsz)

    def pull_item(self, index, num_frames=2):
        valid = False
        while not valid:
            try:
                video_id = random.sample(self.video_list, k=1)[0]
                if self.is_video:
                    cur_seq_anno = self.annotations[video_id]
                    frame_ids = cur_seq_anno.keys()
                    if self.max_interval is not None:
                        ref_frame_id = int(random.sample(frame_ids, k=1)[0])
                        interval = random.randint(1, self.max_interval)
                        cur_frame_id = ref_frame_id + interval
                        if str(cur_frame_id) not in frame_ids:
                            cur_frame_id = ref_frame_id - interval
                    else:
                        ref_frame_id, cur_frame_id = random.sample(frame_ids, k=2)
                        ref_frame_id = int(ref_frame_id)
                        cur_frame_id = int(cur_frame_id)
                    res_r, file_name_r = cur_seq_anno[str(ref_frame_id)]["res"], cur_seq_anno[str(ref_frame_id)]["file_name"]
                    res_c, file_name_c = cur_seq_anno[str(cur_frame_id)]["res"], cur_seq_anno[str(cur_frame_id)]["file_name"]
                    result = []
                    result.append(self.load_resized_img(file_name_r, np.array(res_r)))
                    result.append(self.load_resized_img(file_name_c, np.array(res_c)))
                else:
                    res, _, file_name = self.annotations[video_id]
                    # load image and preprocess
                    result = [self.load_resized_img(file_name, res)] * 2
                valid = True
            except:
                pass
        return result, None, None

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
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
