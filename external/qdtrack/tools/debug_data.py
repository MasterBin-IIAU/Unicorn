import argparse
import copy
import os
import torch
from mmcv import Config
from mmdet.datasets import build_dataset

from qdtrack import __version__
import cv2
import numpy as np
from qdtrack.datasets import BDDVideoDataset # keep this line

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

"""2021.12.03 debug training data"""
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    datasets = [build_dataset(cfg.data.val)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    save_dir = "debug"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, data in enumerate(datasets[0]):
        if idx == 1:
            break
        print(data)
        ref_norm = data["ref_img_metas"].data["img_norm_cfg"]
        mean, std = ref_norm["mean"].reshape(3, 1, 1), ref_norm["std"].reshape(3, 1, 1)
        ref_gt_match_indices = data["ref_gt_match_indices"].data.numpy()
        cur_gt_match_indices = data["gt_match_indices"].data.numpy()
        # ref frame
        ref_img = (data["ref_img"].data.numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
        ref_img = np.ascontiguousarray(ref_img)
        ref_gt_bboxes = data["ref_gt_bboxes"].data.numpy()
        # current frame
        cur_img = (data["img"].data.numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
        cur_img = np.ascontiguousarray(cur_img)
        cur_gt_bboxes = data["gt_bboxes"].data.numpy()

        assert len(ref_gt_bboxes) == len(ref_gt_match_indices)
        assert len(cur_gt_bboxes) == len(cur_gt_match_indices)
        for i in range(len(ref_gt_bboxes)):
            x1, y1, x2, y2 = list(ref_gt_bboxes[i])
            ref_img_copy = copy.deepcopy(ref_img)
            cv2.rectangle(ref_img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
            cur_img_copy = copy.deepcopy(cur_img)
            j = ref_gt_match_indices[i]
            if j != -1:
                x1, y1, x2, y2 = list(cur_gt_bboxes[j])
                cv2.rectangle(cur_img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
            ref_save_img = np.concatenate([ref_img_copy, cur_img_copy], axis=1) # (H, W1+W2, 3)
            cv2.imwrite(os.path.join(save_dir, "ref_img_%02d.jpg"%i), ref_save_img)
            print(i, j)

        for i in range(len(cur_gt_bboxes)):
            x1, y1, x2, y2 = list(cur_gt_bboxes[i])
            cur_img_copy = copy.deepcopy(cur_img)
            cv2.rectangle(cur_img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
            j = cur_gt_match_indices[i]
            ref_img_copy = copy.deepcopy(ref_img)
            if j != -1:
                x1, y1, x2, y2 = list(ref_gt_bboxes[j])
                cv2.rectangle(ref_img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
            cur_save_img = np.concatenate([cur_img_copy, ref_img_copy], axis=1) # (H, W1+W2, 3)
            cv2.imwrite(os.path.join(save_dir, "cur_img_%02d.jpg"%i), cur_save_img)
            print(i, j)




if __name__ == '__main__':
    main()
