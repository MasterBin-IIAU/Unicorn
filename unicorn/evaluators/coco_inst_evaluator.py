#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------
# Unicorn
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# ----------------------------------------------------------------------


import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm

import torch
import torchvision

from unicorn.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from unicorn.utils.boxes import postprocess_inst
import torch.nn.functional as F
import cv2
import numpy as np
"""22.01.09 COCO Instance Segmentation Evaluator"""
class COCOInstEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False, max_ins=None, mask_thres=0.5, 
        d_rate=4, use_raft=False):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        print("Using COCO Instance Segmentation Evaluator...")
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.max_ins = max_ins
        self.mask_thres = mask_thres
        self.d_rate = d_rate
        self.use_raft = use_raft

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        # warmup (optional. If cudnn.benchmark is True, please use warmup)
        if distributed:
            mask_head = model.module.head.mask_head
        else:
            mask_head = model.head.mask_head
        # mask_head.warmup()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                if self.use_raft:
                    outputs, locations, dynamic_params, fpn_levels, mask_feats, up_masks = model(imgs)
                else:
                    outputs, locations, dynamic_params, fpn_levels, mask_feats = model(imgs)

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                # outputs = postprocess(
                #     outputs, self.num_classes, self.confthre, self.nmsthre
                # )
                if self.use_raft:
                    outputs, outputs_mask = postprocess_inst(
                        outputs, locations, dynamic_params, fpn_levels, mask_feats, mask_head,
                        self.num_classes, self.confthre, self.nmsthre, d_rate=self.d_rate, up_masks=up_masks)
                else:
                    outputs, outputs_mask = postprocess_inst(
                        outputs, locations, dynamic_params, fpn_levels, mask_feats, mask_head,
                        self.num_classes, self.confthre, self.nmsthre, d_rate=self.d_rate)
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, outputs_mask, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, outputs_mask, info_imgs, ids):
        data_list = []
        for (output, output_mask, img_h, img_w, img_id) in zip(
            outputs, outputs_mask, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            
            # instance segmentation
            ori_masks = F.interpolate(output_mask, scale_factor=1/scale, mode="bilinear", align_corners=False)\
                [:, 0, :img_h, :img_w] > self.mask_thres # (N, height, width)
            ori_masks_arr = ori_masks.cpu().numpy()

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                # instance segmentation
                mask = ori_masks_arr[ind]
                contours, _ = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = contour.flatten().tolist()
                    # segmentation.append(contour)
                    if len(contour) > 4:
                        pred_data["segmentation"].append(contour)
                # from pycocotools import mask as maskUtils
                # # polygon to compressed RLE
                # rles = maskUtils.frPyObjects(pred_data["segmentation"], img_h, img_w)
                # rle = maskUtils.merge(rles)
                # # compressed RLE to binary mask
                # m = maskUtils.decode(rle)
                if pred_data["segmentation"] == []:
                    continue
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from unicorn.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[0])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
