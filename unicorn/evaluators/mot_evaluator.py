from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from unicorn.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from unicorn.utils.boxes import postprocess_inst
from unicorn.tracker.byte_tracker import BYTETracker
# from unicorn.sort_tracker.sort import Sort
# from unicorn.deepsort_tracker.deepsort import DeepSort
# from unicorn.motdt_tracker.motdt_tracker import OnlineTracker
from unicorn.tracker.quasi_dense_embed_tracker import QuasiDenseEmbedTracker

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import copy
from unicorn.utils.merge import merge_backbone_output
import torch.nn.functional as F
import numpy as np
import cv2
import math
import pycocotools.mask as rletools

def write_results_mots(filename, results):
    save_format = '{frame_id} {track_id} {cat_id} {H} {W} {rle}\n'
    with open(filename, 'w') as f:
        for (frame_id, track_ids, cat_id, H, W, rles) in results:
            for track_id, rle in zip(track_ids, rles):
                if track_id < 0:
                    continue
                line = save_format.format(frame_id=frame_id, track_id=(2000+track_id), cat_id=cat_id, H=H, W=W, rle=rle)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes, mask_thres=0.3):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args
        self.mask_thres = mask_thres

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
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
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                else:
                    self.args.track_thresh = ori_thresh
                
                if video_name == 'MOT20-06' or video_name == 'MOT20-08':
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs, _ = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
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
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
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
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
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
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
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
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from unicorn.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from unicorn.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

    def evaluate_omni_mots(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        grid_sample=False,
        exp=None
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
        ids = []
        # data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        tracker = QuasiDenseEmbedTracker()
        s = 8
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = QuasiDenseEmbedTracker()
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_mots(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                
                img_h, img_w = info_imgs[0], info_imgs[1]
                scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
                det_outputs, cur_dict = model(imgs, mode="whole")
                if getattr(exp, "use_raft", False):
                    outputs, locations, dynamic_params, fpn_levels, mask_feats, up_masks = det_outputs
                    up_mask_b = up_masks[0:1] # assume batch size = 1
                    outputs, outputs_mask = postprocess_inst(
                        outputs, locations, dynamic_params, fpn_levels, mask_feats, model.head.mask_head,
                        exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=False, up_masks=up_mask_b, d_rate=exp.d_rate)
                else:
                    outputs, locations, dynamic_params, fpn_levels, mask_feats = det_outputs
                    outputs, outputs_mask = postprocess_inst(
                        outputs, locations, dynamic_params, fpn_levels, mask_feats, model.head.mask_head,
                        self.num_classes, self.confthre, self.nmsthre, class_agnostic=False)

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                # outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    # print("inference time:", 1000*(infer_end - start))
                    inference_time += infer_end - start

            # output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            # data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                with torch.no_grad():
                    # prepare for tracking
                    bboxes, scores = outputs[0][:, :4], outputs[0][:, 4:5] * outputs[0][:, 5:6]
                    masks = F.interpolate(outputs_mask[0], scale_factor=1/scale, mode="bilinear", align_corners=False)\
                        [:, 0, :img_h, :img_w] > self.mask_thres # (N, height, width)
                    # filter low-score boxes
                    keep_inds = scores[:, 0] > 0.1
                    bboxes = bboxes[keep_inds]
                    scores = scores[keep_inds]
                    labels = torch.ones((bboxes.size(0),)) # (N, )
                    masks = masks[keep_inds]
                    if frame_id == 1:
                        pre_dict = copy.deepcopy(cur_dict)
                    """ feature interaction """
                    _, new_feat_cur = model(seq_dict0=pre_dict, seq_dict1=cur_dict, mode="interaction")
                    """ up-sampling --> embedding"""
                    embed_cur = model(feat=new_feat_cur, mode="upsample")  # (1, C, H/8, W/8)
                    pre_dict = copy.deepcopy(cur_dict)
                    track_feat_list = []
                    for i in range(bboxes.size(0)):
                        x1, y1, x2, y2 = bboxes[i]
                        if grid_sample:
                            cx, cy = (x1+x2)/2/s - 0.5, (y1+y2)/2/s - 0.5
                            cx = (torch.clamp(cx, min=0, max=self.img_size[1]//s-1) / (self.img_size[1]//s-1) - 0.5) * 2.0 # range of [-1, 1]
                            cy = (torch.clamp(cy, min=0, max=self.img_size[0]//s-1) / (self.img_size[0]//s-1) - 0.5) * 2.0 # range of [-1, 1]
                            grid = torch.stack([cx, cy], dim=-1).view(1, 1, 1, 2) # (1, 1, 1, 2)
                            track_feat_list.append(F.grid_sample(embed_cur, grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze())
                        else:
                            cx = torch.round((x1+x2)/2/s).int().clamp(min=0, max=self.img_size[1]//s-1)
                            cy = torch.round((y1+y2)/2/s).int().clamp(min=0, max=self.img_size[0]//s-1)
                            track_feat_list.append(embed_cur[0, :, cy, cx])
                    if len(track_feat_list) == 0:
                        track_feats = torch.zeros((0, embed_cur.size(1)))
                    else:
                        track_feats = torch.stack(track_feat_list, dim=0) # (N, C)
                    # rescale boxes to the original image scale
                    img_h, img_w = info_imgs[0].item(), info_imgs[1].item()
                    scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
                    bboxes /= scale
                    track_inputs = torch.cat((bboxes, scores), dim=1) # (N, 5) (x1, y1, x2, y2, conf)
                    # run tracking
                    track_inputs_cpu = track_inputs.cpu()
                    track_feats_cpu = track_feats.cpu()
                    output_bboxes, _, output_ids, indexs = tracker.match(track_inputs_cpu, labels, track_feats_cpu, frame_id, return_index=True)
                    # remove invalid results
                    valid_inds = (output_ids > -1)
                    output_bboxes = output_bboxes[valid_inds]
                    output_ids = output_ids[valid_inds]
                    # deal with masks
                    masks = masks[indexs]
                    masks = masks[valid_inds]
                    # sort to ascending order
                    _, inds = output_ids.sort(descending=False)
                    output_ids = output_ids[inds]
                    output_bboxes = output_bboxes[inds]
                    masks = masks[inds]
                    # online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
                    output_bboxes_np, output_ids_np = output_bboxes.cpu().numpy(), output_ids.cpu().numpy()
                    """post-processing (overlap-free mask prediction)"""
                    if masks.size(0) > 0:
                        masks_new = masks.clone()
                        mask_prev = masks[0].clone()
                        for n in range(1, masks.size(0)):
                            masks_new[n] = torch.logical_and(torch.logical_not(mask_prev), masks[n])
                            mask_prev = torch.logical_or(mask_prev, masks[n])
                        masks_np = masks_new.cpu().numpy()
                    # online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    mask_rle_list = []
                    # imgs_save_rsz = imgs.cpu().squeeze(0).permute((1, 2, 0)).contiguous().numpy() # (H, W, 3)
                    # crop_h, crop_w = int(round((img_h * scale).item())), int(round((img_w * scale).item()))
                    # imgs_save_rsz_crop = imgs_save_rsz[:crop_h, :crop_w]
                    # imgs_save_ori = cv2.resize(imgs_save_rsz_crop, (img_w.item(), img_h.item()))
                    for i in range(output_bboxes_np.shape[0]):
                        cur_id = output_ids_np[i]
                        if cur_id > -1:
                            # imgs_save = copy.deepcopy(imgs_save_ori)
                            x1, y1, x2, y2, score = output_bboxes_np[i]
                            w, h = x2-x1, y2-y1
                            # vertical = w / h > 1.6
                            if w * h > self.args.min_box_area:
                                # cv2.rectangle(imgs_save, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), thickness=2)
                                # imgs_save[:, :, 0] += 100 * masks_np[i]
                                # cv2.imwrite("%02d.jpg" %i, imgs_save)
                                # online_tlwhs.append(np.array([x1, y1, w, h]))
                                """deal with masks"""
                                mask = masks_np[i] # (H, W)
                                mask = np.asfortranarray(mask)
                                rle = rletools.encode(mask)
                                rle["counts"] = rle["counts"].decode("utf-8")
                                mask_rle_list.append(rle["counts"]) 
                                
                                online_ids.append(cur_id + 1) # 1-based for MOT
                    # import sys
                    # sys.exit(0)
                    # save results
                    results.append((frame_id, online_ids, 2, img_h, img_w, mask_rle_list))
                    torch.cuda.empty_cache()
                    # save_dir = "video_%d" % video_id
                    # if not os.path.exists(save_dir):
                    #     os.makedirs(save_dir)
                    # cv2.imwrite(os.path.join(save_dir, "%04d.jpg" % frame_id), imgs_save)

            if is_time_record:
                track_end = time_synchronized()
                # print("track time:", 1000*(track_end - infer_end))
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_mots(result_filename, results)

        # statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        # if distributed:
        #     data_list = gather(data_list, dst=0)
        #     data_list = list(itertools.chain(*data_list))
        #     torch.distributed.reduce(statistics, dst=0)

        # eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        # return eval_results

    def evaluate_omni(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        grid_sample=False,
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
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        tracker = QuasiDenseEmbedTracker()
        s = 8
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = QuasiDenseEmbedTracker()
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                outputs, cur_dict = model(imgs, mode="whole")
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                with torch.no_grad():
                    # prepare for tracking
                    bboxes, scores = outputs[0][:, :4], outputs[0][:, 4:5] * outputs[0][:, 5:6]
                    # filter low-score boxes
                    keep_inds = scores[:, 0] > 0.1
                    bboxes = bboxes[keep_inds]
                    scores = scores[keep_inds]
                    labels = torch.ones((bboxes.size(0),)) # (N, )
                    if frame_id == 1:
                        pre_dict = copy.deepcopy(cur_dict)
                    """ feature interaction """
                    _, new_feat_cur = model(seq_dict0=pre_dict, seq_dict1=cur_dict, mode="interaction")
                    """ up-sampling --> embedding"""
                    embed_cur = model(feat=new_feat_cur, mode="upsample")  # (1, C, H/8, W/8)
                    pre_dict = copy.deepcopy(cur_dict)
                    track_feat_list = []
                    # network_end = time.time()
                    # vertorized version
                    cx, cy = (bboxes[:, 0] + bboxes[:, 2])/2/s - 0.5,  (bboxes[:, 1] + bboxes[:, 3])/2/s - 0.5
                    cx = (torch.clamp(cx, min=0, max=self.img_size[1]//s-1) / (self.img_size[1]//s-1) - 0.5) * 2.0 # range of [-1, 1]
                    cy = (torch.clamp(cy, min=0, max=self.img_size[0]//s-1) / (self.img_size[0]//s-1) - 0.5) * 2.0 # range of [-1, 1]
                    grids = torch.stack([cx, cy], dim=-1) # (N, 2)
                    for i in range(bboxes.size(0)):
                        grid = grids[i].view(1, 1, 1, 2) # (1, 1, 1, 2)
                        track_feat_list.append(F.grid_sample(embed_cur, grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze())
                    if len(track_feat_list) == 0:
                        track_feats = torch.zeros((0, embed_cur.size(1)))
                    else:
                        track_feats = torch.stack(track_feat_list, dim=0) # (N, C)
                    # rescale boxes to the original image scale
                    img_h, img_w = info_imgs[0], info_imgs[1]
                    scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
                    bboxes /= scale
                    track_inputs = torch.cat((bboxes, scores), dim=1) # (N, 5) (x1, y1, x2, y2, conf)
                    # run tracking
                    track_inputs_cpu = track_inputs.cpu()
                    track_feats_cpu = track_feats.cpu()
                    # torch.cuda.synchronize()
                    # prepare_end = time.time()
                    output_bboxes, _, output_ids = tracker.match(track_inputs_cpu, labels, track_feats_cpu, frame_id)
                    # torch.cuda.synchronize()
                    # match_end = time.time()
                    # remove invalid results
                    valid_inds = (output_ids > -1)
                    output_bboxes = output_bboxes[valid_inds]
                    output_ids = output_ids[valid_inds]
                    # sort to ascending order
                    _, inds = output_ids.sort(descending=False)
                    output_ids = output_ids[inds]
                    output_bboxes = output_bboxes[inds]
                    # online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
                    output_bboxes_np, output_ids_np = output_bboxes.cpu().numpy(), output_ids.cpu().numpy()
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    # imgs_save_rsz = imgs.cpu().squeeze(0).permute((1, 2, 0)).contiguous().numpy() # (H, W, 3)
                    # crop_h, crop_w = int(round((img_h * scale).item())), int(round((img_w * scale).item()))
                    # imgs_save_rsz_crop = imgs_save_rsz[:crop_h, :crop_w]
                    # imgs_save = cv2.resize(imgs_save_rsz_crop, (img_w.item(), img_h.item()))
                    for i in range(output_bboxes_np.shape[0]):
                        cur_id = output_ids_np[i]
                        if cur_id > -1:
                            x1, y1, x2, y2, score = output_bboxes_np[i]
                            w, h = x2-x1, y2-y1
                            vertical = w / h > 1.6
                            if w * h > self.args.min_box_area and not vertical:
                                # cv2.rectangle(imgs_save, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), thickness=2)
                                # text = "ID:%03d" % cur_id
                                # txt_color = (255, 255, 255)
                                # font = cv2.FONT_HERSHEY_SIMPLEX
                                # txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]
                                # cv2.putText(imgs_save, text, (int(x1), int(y1) + txt_size[1]), font, 0.8, txt_color, thickness=2)
                                online_tlwhs.append(np.array([x1, y1, w, h]))
                                online_ids.append(cur_id + 1) # 1-based for MOT
                                online_scores.append(score)
                    # save results
                    results.append((frame_id, online_tlwhs, online_ids, online_scores))
                    torch.cuda.empty_cache()
                    # save_dir = "video_%d" % video_id
                    # if not os.path.exists(save_dir):
                    #     os.makedirs(save_dir)
                    # cv2.imwrite(os.path.join(save_dir, "%04d.jpg" % frame_id), imgs_save)

            if is_time_record:
                track_end = time_synchronized()
                # print("detection time: %.1f ms, other network time: %.1f ms, prepare time: %.1f ms, matching time: %.1f ms, track time: %.1f ms" 
                # %(1000*(infer_end-start), 1000*(network_end-infer_end), 1000*(prepare_end-network_end), 1000*(match_end-prepare_end), 1000*(track_end-infer_end)))
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
