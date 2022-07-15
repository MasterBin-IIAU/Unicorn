import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import cv2
import numpy as np
import copy
import torch.nn.functional as F
from unicorn.utils import postprocess
from mmdet.core import bbox2result, encode_mask_results
from qdtrack.core import track2result
from qdtrack.models import build_tracker
# MOTS
from unicorn.utils.boxes import postprocess_inst
from qdtrack.core.track.transforms_mots import segtrack2result
import pycocotools.mask as mask_util

def encode_track_results(track_results):
    """Encode bitmap mask to RLE code.

    Args:
        track_results (list | tuple[list]): track results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).

    Returns:
        list | tuple: RLE encoded mask.
    """
    for id, roi in track_results.items():
        roi['segm'] = mask_util.encode(
            np.array(roi['segm'][:, :, np.newaxis], order='F',
                     dtype='uint8'))[0]  # encoded with RLE
    return track_results

def multi_gpu_test_omni(model, data_loader, exp, tracker_cfg, tmpdir=None, gpu_collect=False, mots=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    s = 8
    for i, data in enumerate(data_loader):
        frame_id = data["img_metas"][0].data[0][0].get('frame_id', -1)
        if frame_id == 0:
            tracker = build_tracker(tracker_cfg)
        # pre-processing
        imgs, info_imgs = preprocess(data, exp.test_size)
        with torch.no_grad():
            if mots:
                det_outputs, cur_dict = model(imgs, mode="whole")
                head = model.module.head
                if getattr(exp, "use_raft", False):
                    outputs, locations, dynamic_params, fpn_levels, mask_feats, up_masks = det_outputs
                    up_mask_b = up_masks[0:1] # assume batch size = 1
                    outputs, outputs_mask = postprocess_inst(
                        outputs, locations, dynamic_params, fpn_levels, mask_feats, head.mask_head,
                        exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=False, up_masks=up_mask_b, d_rate=exp.d_rate)
                else:
                    outputs, locations, dynamic_params, fpn_levels, mask_feats = det_outputs
                    outputs, outputs_mask = postprocess_inst(
                        outputs, locations, dynamic_params, fpn_levels, mask_feats, head.mask_head,
                        exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=False)
            else:
                if hasattr(model.module.head, "mask_branch"):
                    # use heads with mask branch for box-level tracking (for model trained on four tasks)
                    det_outputs, cur_dict = model(imgs, mode="whole")
                    outputs, locations, dynamic_params, fpn_levels, mask_feats = det_outputs
                else:
                    outputs, cur_dict = model(imgs, mode="whole")
                outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
            if frame_id == 0:
                pre_dict = copy.deepcopy(cur_dict)
            if outputs[0] is not None:
                # with torch.no_grad():
                # prepare for tracking
                bboxes, scores = outputs[0][:, :4], outputs[0][:, 4:5] * outputs[0][:, 5:6]
                det_labels = outputs[0][:, 6].cpu() # (N, )
                """ feature interaction """
                new_feat_pre, new_feat_cur = model(seq_dict0=pre_dict, seq_dict1=cur_dict, mode="interaction")
                """ up-sampling --> embedding"""
                embed_pre = model(feat=new_feat_pre, mode="upsample")  # (1, C, H/8, W/8)
                embed_cur = model(feat=new_feat_cur, mode="upsample")  # (1, C, H/8, W/8)
                pre_dict = copy.deepcopy(cur_dict)
                track_feat_list = []
                for i in range(bboxes.size(0)):
                    x1, y1, x2, y2 = bboxes[i]
                    if exp.grid_sample:
                        cx, cy = (x1+x2)/2/s - 0.5, (y1+y2)/2/s - 0.5
                        cx = (torch.clamp(cx, min=0, max=exp.test_size[1]//s-1) / (exp.test_size[1]//s-1) - 0.5) * 2.0 # range of [-1, 1]
                        cy = (torch.clamp(cy, min=0, max=exp.test_size[0]//s-1) / (exp.test_size[0]//s-1) - 0.5) * 2.0 # range of [-1, 1]
                        grid = torch.stack([cx, cy], dim=-1).view(1, 1, 1, 2) # (1, 1, 1, 2)
                        track_feat_list.append(F.grid_sample(embed_cur, grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze())
                    else:
                        cx = torch.round((x1+x2)/2/s).int().clamp(min=0, max=exp.test_size[1]//s-1)
                        cy = torch.round((y1+y2)/2/s).int().clamp(min=0, max=exp.test_size[0]//s-1)
                        track_feat_list.append(embed_cur[0, :, cy, cx])
                track_feats = torch.stack(track_feat_list, dim=0).cpu() # (N, C)
                # rescale boxes to the original image scale
                img_h, img_w = info_imgs[0], info_imgs[1]
                scale = min(exp.test_size[0] / float(img_h), exp.test_size[1] / float(img_w))
                bboxes /= scale
                det_bboxes = torch.cat([bboxes, scores], dim=1).cpu() # (x1, y1, x2, y2, conf)
                if mots:
                    # track
                    track_bboxes, labels, ids, indexs = tracker.match(
                        bboxes=det_bboxes,
                        labels=det_labels,
                        track_feats=track_feats,
                        frame_id=frame_id,
                        return_index=True)
                    # masks
                    mask_thres = getattr(exp, "mask_thres", 0.3)
                    masks = F.interpolate(outputs_mask[0], scale_factor=1/scale, mode="bilinear", align_corners=False)\
                        [:, 0, :img_h, :img_w] > mask_thres # (N, height, width)
                    # the size of masks may be smaller than the original img !
                    # TODO: fix this problem in a more elegant way
                    tmp_h, tmp_w = masks.size()[1:]
                    masks_full = torch.zeros((len(masks), img_h, img_w), dtype=torch.bool, device=masks.device)
                    masks_full[:, :tmp_h, :tmp_w] = masks[:, :tmp_h, :tmp_w]
                    masks_full_new = masks_full[indexs]
                    masks_np = masks_full_new.cpu().numpy()
                    result = {}
                    result["track_result"] = segtrack2result(track_bboxes, labels, masks_np, ids)

                    if 'track_result' in result:
                        result['track_result'] = (
                            encode_track_results(result['track_result']))
                    # for the instance segmentation metric
                    result['bbox_result'] = bbox2result(det_bboxes, det_labels, exp.num_classes)
                    segm_result = [[] for _ in range(exp.num_classes)]
                    masks_cur = masks_full.cpu().numpy()
                    for n in range(len(det_labels)):
                        label = int(det_labels[n])
                        segm_result[label].append(masks_cur[n])
                    result['segm_result'] = encode_mask_results(segm_result)
                else:
                    # track
                    track_bboxes, labels, ids = tracker.match(
                        bboxes=det_bboxes,
                        labels=det_labels,
                        track_feats=track_feats,
                        frame_id=frame_id)
                    bbox_result = bbox2result(det_bboxes, det_labels, exp.num_classes)
                    track_result = track2result(track_bboxes, labels, ids, exp.num_classes)
                    result = dict(bbox_results=bbox_result, track_results=track_result)
            else:
                if mots:
                    track_result = defaultdict(list)
                    bbox_result = bbox2result(np.zeros((0, 5)), None, exp.num_classes)
                    segm_result = [[] for _ in range(exp.num_classes)]
                    result = {"track_result": track_result, "bbox_result": bbox_result, "segm_result": segm_result}
                else:
                    bbox_result = bbox2result(np.zeros((0, 5)), None, exp.num_classes)
                    track_result = [np.zeros((0, 6), dtype=np.float32) for i in range(exp.num_classes)]
                    result = dict(bbox_results=bbox_result, track_results=track_result)
        for k, v in result.items():
            results[k].append(v)

        if rank == 0:
            batch_size = (
                len(data['img_meta']._data)
                if 'img_meta' in data else data['img'][0].size(0))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        raise NotImplementedError
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = defaultdict(list)
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_file = mmcv.load(part_file)
            for k, v in part_file.items():
                part_list[k].extend(v)
        shutil.rmtree(tmpdir)
        return part_list


def preprocess(data, input_size):
    norm = data["img_metas"][0].data[0][0]["img_norm_cfg"]
    mean, std = norm["mean"].reshape(3, 1, 1), norm["std"].reshape(3, 1, 1)
    # current frame
    img = (data["img"][0][0].numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
    img = np.ascontiguousarray(img)
    # get the original size and the original image (without padding)
    ori_H, ori_W, _ = data["img_metas"][0].data[0][0]["ori_shape"]
    img_ori = img[:ori_H, :ori_W, :]
    # get img_info and id
    img_info = (ori_H, ori_W)
    # preproc (resize), add dimension, to torch CUDA tensor
    img_rsz, _ = preproc(img_ori, input_size)
    img_tensor = torch.from_numpy(img_rsz[np.newaxis, ...]).cuda() # (1, 3, H, W)
    return img_tensor, img_info

def preproc(img, input_size, swap=(2, 0, 1)):
    """add padding and swap dimension"""
    # print("shape before: ", img.shape)
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    # print(r)
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    # print("shape after: ", resized_img.shape)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def get_ori_img(data):
    norm = data["img_metas"][0].data[0][0]["img_norm_cfg"]
    mean, std = norm["mean"].reshape(3, 1, 1), norm["std"].reshape(3, 1, 1)
    # current frame
    img = (data["img"][0][0].numpy() * std + mean).clip(0, 255).transpose(1, 2, 0) # (H, W, 3)
    img = np.ascontiguousarray(img)
    # get the original size and the original image (without padding)
    ori_H, ori_W, _ = data["img_metas"][0].data[0][0]["ori_shape"]
    img_ori = img[:ori_H, :ori_W, :]
    return img_ori


