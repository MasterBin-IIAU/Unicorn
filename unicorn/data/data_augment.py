#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from unicorn.utils import xyxy2cxcywh
import torch

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
    clip_border=True,
    masks=None
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
            if masks is not None:
                masks = cv2.warpPerspective(
                    masks, M, dsize=(width, height), borderValue=(0, 0, 0)
                )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )
            if masks is not None:
                masks = cv2.warpAffine(
                    masks, M[:2], dsize=(width, height), borderValue=(0, 0, 0)
                )    

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        if clip_border:
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]
        if masks is not None:
            masks = masks[:, :, i]

        if not clip_border:
            valid_mask1 = np.logical_and(targets[:, 0] < width, targets[:, 2] > 0)
            valid_mask2 = np.logical_and(targets[:, 1] < height, targets[:, 3] > 0)
            valid_mask = np.logical_and(valid_mask1, valid_mask2)
            targets = targets[valid_mask]
            if masks is not None:
                masks = masks[:, :, valid_mask]
            # targets = targets[targets[:, 0] < width]
            # targets = targets[targets[:, 2] > 0]
            # targets = targets[targets[:, 1] < height]
            # targets = targets[targets[:, 3] > 0]
    if masks is not None:
        return img, targets, masks
    else:
        return img, targets


def _mirror(image, boxes, prob=0.5, mask=None):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        if mask is not None:
            mask = mask[:, ::-1]
    if mask is None:
        return image, boxes
    else:
        return image, boxes, mask


def _mirror_joint(image, boxes, mask=None):
    _, width, _ = image.shape
    image = image[:, ::-1]
    boxes[:, 0::2] = width - boxes[:, 2::-2]
    if mask is None:
        return image, boxes
    else:
        mask = mask[:, ::-1]
        return image, boxes, mask


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

def preproc_mask(mask, input_size, swap=(2, 0, 1)):
    """add padding and swap dimension"""
    # print("shape before: ", img.shape)
    padded_mask = np.zeros((input_size[0], input_size[1], mask.shape[2]), dtype=np.uint8)

    r = min(input_size[0] / mask.shape[0], input_size[1] / mask.shape[1])
    # print(r)
    if r != 1:
        resized_mask = cv2.resize(
            mask,
            (int(mask.shape[1] * r), int(mask.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
    else:
        resized_mask = mask.astype(np.uint8)
    if len(resized_mask.shape) == 2:
        resized_mask = resized_mask[:, :, None]
    # print("shape after: ", resized_img.shape)
    padded_mask[: int(mask.shape[0] * r), : int(mask.shape[1] * r)] = resized_mask

    padded_mask = padded_mask.transpose(swap)
    padded_mask = np.ascontiguousarray(padded_mask, dtype=np.float32)
    return padded_mask, r

def preproc_search(image, boxes, scale_jitter_factor=0.5, center_jitter_factor=4.5, search_area_factor=5, output_sz=640, swap=(2, 0, 1)):
    box_tensor = torch.tensor(boxes[0])
    jittered_anno = get_jittered_box(box_tensor, scale_jitter_factor, center_jitter_factor, jitter=True) # (4, )

    # Avoid too small bounding boxes
    w, h = jittered_anno[2], jittered_anno[3]
    crop_sz = torch.ceil(torch.sqrt(w * h) * search_area_factor)
    if (crop_sz < 1).any():
        jittered_anno = get_jittered_box(torch.tensor(boxes[0]), scale_jitter_factor, center_jitter_factor, jitter=False) # (4, )

    # Crop image region centered at jittered_anno box and get the attention mask
    crop_img, crop_box = jittered_center_crop(image, jittered_anno, box_tensor, search_area_factor, output_sz)
    crop_img_np, crop_box_np = crop_img, crop_box.numpy()

    # Visualize
    # cx, cy, w, h = list(crop_box_np[0])
    # cv2.rectangle(crop_img_np, (int(cx-0.5*w), int(cy-0.5*h)), (int(cx+0.5*w), int(cy+0.5*h)), (0, 0, 255), thickness=2)
    # import torch.distributed as dist
    # cv2.imwrite("%d.jpg"%dist.get_rank(), crop_img_np)
    crop_img_np = crop_img_np.transpose(swap)
    crop_img_np = np.ascontiguousarray(crop_img_np, dtype=np.float32)
    return crop_img_np, crop_box_np

class TrainTransform_local:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, legacy=False):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.legacy = legacy

    def __call__(self, image, targets, input_dim, joint=False, flip=False):
        """joint: whether to jointly flip the reference and the current frame"""
        has_trackid = False
        if targets.shape[1] == 6:
            has_trackid = True
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if has_trackid:
            trackids = targets[:, 5].copy()
        if len(boxes) == 0:
            # if there is no object on the image
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            if self.legacy:
                image = image[::-1, :, :].copy()
                image /= 255.0
                image -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                image /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        if has_trackid:
            trackids_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        if joint:
            if flip:
                image_t, boxes = _mirror_joint(image, boxes)
            else:
                image_t, boxes = image, boxes
        else:
            image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        # image_t, r_ = preproc(image_t, input_dim)
        # # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        # boxes *= r_
        image_t, boxes = preproc_search(image_t, boxes, output_sz=input_dim[0])

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            if has_trackid:
                trackids_t = trackids_o

        labels_t = np.expand_dims(labels_t, 1)
        if has_trackid:
            trackids_t = trackids[mask_b]
            trackids_t = np.expand_dims(trackids_t, 1)
            targets_t = np.hstack((labels_t, boxes_t, trackids_t)) # turn to "label first, box next, track ids last" (N, 6)
            padded_labels = np.zeros((self.max_labels, 6))
        else:
            targets_t = np.hstack((labels_t, boxes_t)) # turn to "label first, box next" (N, 5)
            padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        if self.legacy:
            image_t = image_t[::-1, :, :].copy()
            image_t /= 255.0
            image_t -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image_t /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return image_t, padded_labels


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, legacy=False):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.legacy = legacy

    def __call__(self, image, targets, input_dim, joint=False, flip=False):
        """joint: whether to jointly flip the reference and the current frame"""
        has_trackid = False
        if targets.shape[1] == 6:
            has_trackid = True
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if has_trackid:
            trackids = targets[:, 5].copy()
        if len(boxes) == 0:
            # if there is no object on the image
            if has_trackid:
                targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            else:
                targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            if self.legacy:
                image = image[::-1, :, :].copy()
                image /= 255.0
                image -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                image /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        if has_trackid:
            trackids_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        if joint:
            if flip:
                image_t, boxes = _mirror_joint(image, boxes)
            else:
                image_t, boxes = image, boxes
        else:
            image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            if has_trackid:
                trackids_t = trackids_o

        labels_t = np.expand_dims(labels_t, 1)
        if has_trackid:
            trackids_t = trackids[mask_b]
            trackids_t = np.expand_dims(trackids_t, 1)
            targets_t = np.hstack((labels_t, boxes_t, trackids_t)) # turn to "label first, box next, track ids last" (N, 6)
            padded_labels = np.zeros((self.max_labels, 6))
        else:
            targets_t = np.hstack((labels_t, boxes_t)) # turn to "label first, box next" (N, 5)
            padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        if self.legacy:
            image_t = image_t[::-1, :, :].copy()
            image_t /= 255.0
            image_t -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image_t /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))


def get_jittered_box(box: torch.Tensor, scale_jitter_factor, center_jitter_factor, jitter=True):
    """ Jitter the input box
    args:
        box - input bounding box (cx, cy, w, h) (4, )
        mode - string 'template' or 'search' indicating template or search data

    returns:
        np.array - jittered box (cx, cy, w, h) (4, )
    """
    if jitter:
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * scale_jitter_factor)
    else:
        jittered_size = box[2:4]
    max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor).float())
    jittered_center = box[0:2] + max_offset * (torch.rand(2) - 0.5)

    return torch.cat((jittered_center, jittered_size), dim=0)


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """
    frames_crop, resize_factors = sample_target(frames, box_extract, search_area_factor, output_sz, pad_val=114)

    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = transform_image_to_crop(box_gt, box_extract, resize_factors, crop_sz, normalize=False)
    
    # clip box within the image
    cx, cy ,w, h = box_crop
    x1, y1 = torch.clamp(cx-0.5*w, min=0), torch.clamp(cy-0.5*h, min=0)
    x2, y2 = torch.clamp(cx+0.5*w, max=output_sz), torch.clamp(cy+0.5*h, max=output_sz)
    cx_new, cy_new = (x1+x2)/2, (y1+y2)/2
    w_new, h_new = x2-x1, y2-y1
    return frames_crop, torch.stack([cx_new, cy_new, w_new, h_new], dim=0).unsqueeze(0)


def sample_target(im, target_bb, search_area_factor, output_sz=None, pad_val=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        cx, cy, w, h = target_bb.tolist()
    else:
        cx, cy, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(cx - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(cy - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    if pad_val is not None:
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=(pad_val, pad_val, pad_val))
    else:
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        return im_crop_padded, resize_factor
    else:
        return im_crop_padded, 1.0


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2]

    box_in_center = box_in[0:2]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

class TrainTransform_omni:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, legacy=False):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.legacy = legacy

    def __call__(self, image, targets, input_dim, joint=False, flip=False):
        """joint: whether to jointly flip the reference and the current frame"""
        has_trackid = False
        if targets.shape[1] == 6:
            has_trackid = True
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if has_trackid:
            # MOT
            trackids = targets[:, 5].copy()
        else:
            # SOT
            trackids = np.zeros((len(targets),))
            trackids[0] = 1 
        if len(boxes) == 0:
            # if there is no object on the image
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            if self.legacy:
                image = image[::-1, :, :].copy()
                image /= 255.0
                image -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                image /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        if has_trackid:
            trackids_o = targets_o[:, 5]
        else:
            trackids_o = np.zeros((len(targets),))
            trackids_o[0] = 1 
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        if joint:
            if flip:
                image_t, boxes = _mirror_joint(image, boxes)
            else:
                image_t, boxes = image, boxes
        else:
            image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            trackids_t = trackids_o

        labels_t = np.expand_dims(labels_t, 1)
        
        trackids_t = trackids[mask_b]
        trackids_t = np.expand_dims(trackids_t, 1)
        targets_t = np.hstack((labels_t, boxes_t, trackids_t)) # turn to "label first, box next, track ids last" (N, 6)
        padded_labels = np.zeros((self.max_labels, 6))

        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        if self.legacy:
            image_t = image_t[::-1, :, :].copy()
            image_t /= 255.0
            image_t -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image_t /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return image_t, padded_labels

"""Train Transform for instance segmentation"""
class TrainTransform_Ins:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, legacy=False, d_rate=1/4):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.legacy = legacy
        self.d_rate = d_rate # downsampling rate

    def __call__(self, image, targets, mask, input_dim, joint=False, flip=False):
        # to reduce the latency (copy data from cpu to gpu), we downsample mask by 4x
        input_dim_d = (int(input_dim[0] * self.d_rate), int(input_dim[1] * self.d_rate))
        """joint: whether to jointly flip the reference and the current frame"""
        has_trackid = False
        if targets.shape[1] == 6:
            has_trackid = True
        assert (targets.shape[0] == mask.shape[2]) # assert (num_boxes = num_masks)
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if has_trackid:
            trackids = targets[:, 5].copy()
        if len(boxes) == 0:
            # if there is no object on the image
            if has_trackid:
                targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            else:
                targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            mask = np.zeros((self.max_labels, input_dim_d[0], input_dim_d[1]), dtype=np.float32)
            if self.legacy:
                image = image[::-1, :, :].copy()
                image /= 255.0
                image -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                image /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            return image, targets, mask

        image_o = image.copy()
        targets_o = targets.copy()
        mask_o = mask.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        if has_trackid:
            trackids_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        if joint:
            if flip:
                image_t, boxes, mask_t = _mirror_joint(image, boxes, mask)
            else:
                image_t, boxes, mask_t = image, boxes, mask
        else:
            image_t, boxes, mask_t = _mirror(image, boxes, self.flip_prob, mask)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        mask_t, _ = preproc_mask(mask_t, input_dim_d)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        mask_t = mask_t[mask_b]
        if has_trackid:
            trackids_t = trackids[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            mask_t = np.zeros((self.max_labels, input_dim_d[0], input_dim_d[1]), dtype=np.float32)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            if has_trackid:
                trackids_t = trackids_o

        labels_t = np.expand_dims(labels_t, 1)
        if has_trackid:
            trackids_t = np.expand_dims(trackids_t, 1)
            targets_t = np.hstack((labels_t, boxes_t, trackids_t)) # turn to "label first, box next, track ids last" (N, 6)
            padded_labels = np.zeros((self.max_labels, 6))
        else:
            targets_t = np.hstack((labels_t, boxes_t)) # turn to "label first, box next" (N, 5)
            padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        # masks
        padded_masks = np.zeros((self.max_labels, input_dim_d[0], input_dim_d[1]))
        padded_masks[range(len(mask_t))[:self.max_labels], :, :] = mask_t[:self.max_labels, :, :]
        padded_masks = np.ascontiguousarray(padded_masks, dtype=np.float32)
        if self.legacy:
            image_t = image_t[::-1, :, :].copy()
            image_t /= 255.0
            image_t -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image_t /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return image_t, padded_labels, padded_masks

class TrainTransform_4tasks:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, legacy=False, d_rate=1/4):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.legacy = legacy
        self.d_rate = d_rate # downsampling rate
        self.trans_omni = TrainTransform_omni(max_labels, flip_prob, hsv_prob, legacy)
        self.trans_inst = TrainTransform_Ins(max_labels, flip_prob, hsv_prob, legacy, d_rate)
    
    def __call__(self, image, targets, mask, input_dim, joint=False, flip=False):
        if mask is None:
            image_t, padded_labels = self.trans_omni(image, targets, input_dim, joint, flip)
            return image_t, padded_labels, None
        else:
            return self.trans_inst(image, targets, mask, input_dim, joint, flip)