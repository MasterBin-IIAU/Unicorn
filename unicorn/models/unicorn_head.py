#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2022 ByteDance. All Rights Reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicorn.utils import bboxes_iou

from .losses import IOUloss
from .backbone.network_blocks import BaseConv, DWConv
from .backbone.convnext import Block as Attention_Block

"""Unified Head for SOT-MOT"""
class UnicornHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        use_l1=False,
        use_attention=False,
        n_layer_att=1,
        unshared_obj=False,
        unshared_reg=False,
        mot_weight=1.0,
        scale_all_mot=False,
        fuse_method="sum",
        learnable_fuse=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
            learnable_fuse: whether the broadcast sum is learnable
        """
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.num_classes_sot = 1
        # whether to use unshared obj pred
        self.unshared_obj = unshared_obj
        self.unshared_reg = unshared_reg
        self.mot_weight = mot_weight
        self.scale_all_mot = scale_all_mot
        self.fuse_method = fuse_method
        assert self.fuse_method in ["sum", "mul"]
        assert self.mot_weight >= 1.0
        self.n_layer_att = n_layer_att
        
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList() # MOT classification
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_preds_sot = nn.ModuleList() # SOT classification
        if self.unshared_obj:
            self.obj_preds_sot = nn.ModuleList() # SOT objectness
        if self.unshared_reg:
            self.reg_preds_sot = nn.ModuleList() # SOT regression
        self.stems = nn.ModuleList()
        self.att_layers = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.cls_preds_sot.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes_sot,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ) 
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if self.unshared_reg:
                self.reg_preds_sot.append(
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=4,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if self.unshared_obj:
                self.obj_preds_sot.append(
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.n_anchors * 1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            # newly added attention block
            self.use_attention = use_attention
            self.n_layer_att = n_layer_att
            if use_attention:
                att_list = nn.ModuleList()
                for _ in range(n_layer_att):
                    att_list.append(Attention_Block(int(256 * width), layer_scale_init_value=1.0))
                self.att_layers.append(att_list)
            else:
                self.att_layers.append(nn.Identity())
        self.use_l1 = use_l1
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.learnable_fuse = learnable_fuse
        if self.learnable_fuse:
            for i in range(n_layer_att):
                beta = nn.Parameter(1.0 * torch.ones((int(256 * width), 1, 1)), requires_grad=True)
                setattr(self, "beta_%d"%i, beta)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.cls_preds_sot:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        if self.unshared_obj:
            for conv in self.obj_preds_sot:
                b = conv.bias.view(self.n_anchors, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, mask_in, labels=None, imgs=None, mode=None, return_det=False, return_ota=False):
        """
        mask_in: propagated masks
        mode should be "sot" or "mot" 
        return_det: whether to return detection results during training
        return_ota: whether to return the ota boxes (with corresponding gt_indices)
        """
        self.mode = mode
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        if return_det:
            outputs_test = []
        if return_ota:
            assert return_det is True

        for k, (cls_conv, reg_conv, stride_this_level, x, m) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin, mask_in)
        ):
            x = self.stems[k](x)
            # fusion
            if self.fuse_method == "sum":
                if self.learnable_fuse:
                    beta = getattr(self, "beta_%d"%k)
                    x = x + (m * beta)
                else:
                    x = x + m
            elif self.fuse_method == "mul":
                x = torch.mul(x, m) + x # "+x" is for MOT! 
            else:
                raise ValueError
            # attention
            for n in range(self.n_layer_att):
                x = self.att_layers[k][n](x)
            cls_x = x
            reg_x = x
            
            cls_feat = cls_conv(cls_x)
            if mode == "sot":
                cls_output = self.cls_preds_sot[k](cls_feat)
            elif mode == "mot":
                cls_output = self.cls_preds[k](cls_feat)
            else:
                raise ValueError("""mode has to be 'sot' or 'mot'""")

            reg_feat = reg_conv(reg_x)
            if self.unshared_reg and mode == "sot":
                reg_output = self.reg_preds_sot[k](reg_feat)
            else:
                reg_output = self.reg_preds[k](reg_feat)
            if self.unshared_obj and mode == "sot":
                obj_output = self.obj_preds_sot[k](reg_feat)
            else:
                obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                if return_det:
                    output_test = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                    outputs_test.append(output_test)
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            if return_ota:
                (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), outputs_ota = self.get_losses(
                    imgs,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    labels,
                    torch.cat(outputs, 1),
                    origin_preds,
                    dtype=xin[0].dtype,
                    return_ota=return_ota
                )
            else:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.get_losses(
                    imgs,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    labels,
                    torch.cat(outputs, 1),
                    origin_preds,
                    dtype=xin[0].dtype,
                    return_ota=return_ota
                )
            # deal with unused parameters in cls_preds, obj_preds, reg_preds
            loss_unused = torch.tensor(0.0, device="cuda")
            if self.mode == "sot":
                for module in self.cls_preds:
                    for p in module.parameters():
                        loss_unused += (torch.sum(p) * 0.0)
                if self.unshared_obj:
                    for module in self.obj_preds:
                        for p in module.parameters():
                            loss_unused += (torch.sum(p) * 0.0)
                if self.unshared_reg:
                    for module in self.reg_preds:
                        for p in module.parameters():
                            loss_unused += (torch.sum(p) * 0.0)
            elif self.mode == "mot":
                for module in self.cls_preds_sot:
                    for p in module.parameters():
                        loss_unused += (torch.sum(p) * 0.0)
                if self.unshared_obj:
                    for module in self.obj_preds_sot:
                        for p in module.parameters():
                            loss_unused += (torch.sum(p) * 0.0)
                if self.unshared_reg:
                    for module in self.reg_preds_sot:
                        for p in module.parameters():
                            loss_unused += (torch.sum(p) * 0.0)
            loss += loss_unused
            # Increase the weight of the MOT Loss
            if self.mot_weight > 1.0:
                if self.mode == "mot" and not self.scale_all_mot:
                    loss += conf_loss * (self.mot_weight - 1)
            loss_dict = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            # t_loss = time.time()
            # print("SOT network: %.4f, SOT Loss: %.4f" %(t_net-t_s, t_loss-t_net))
            if return_det:
                self.hw = [x.shape[-2:] for x in outputs_test]
                # [batch, n_anchors_all, 85]
                outputs = torch.cat(
                    [x.flatten(start_dim=2) for x in outputs_test], dim=2
                ).permute(0, 2, 1)
                if self.decode_in_inference:
                    det_results = self.decode_outputs(outputs, dtype=xin[0].type())
                else:
                    det_results = outputs
                """det_results: (2, 21000, 6)"""
                if return_ota:
                    with torch.no_grad():
                        ota_results = []
                        for i in range(det_results.size(0)):
                            if outputs_ota[i] is not None:
                                fg_indices, gt_indices = outputs_ota[i][:, 0], outputs_ota[i][:, 1] # (K, ) K is OTA number
                                det_box_ota = det_results[i][fg_indices, :4] # (K, 4) (cx, cy, w, h)
                                ota_results.append(torch.cat([det_box_ota, gt_indices.unsqueeze(-1)], dim=-1)) # (K, 5)
                            else:
                                ota_results.append(None)
                    return loss_dict, ota_results
                else:
                    return loss_dict, det_results
            else:
                return loss_dict
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        if self.mode == "sot":
            num_classes = self.num_classes_sot
        elif self.mode == "mot":
            num_classes = self.num_classes
        else:
            raise ValueError()
        n_ch = 5 + num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
        return_ota=False
    ):
        # t_s = time.time()
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        # for invalid samples, the dim2=[0, 0, 0, 0, 0]. the coordinates are not normalized.
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        if return_ota:
            outputs_ota = []

        num_fg = 0.0
        num_gts = 0.0
        # t_prepare = time.time()
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                if self.mode == "sot":
                    num_classes = self.num_classes_sot
                elif self.mode == "mot":
                    num_classes = self.num_classes
                else:
                    raise ValueError()
                cls_target = outputs.new_zeros((0, num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                if self.mode == "sot":
                    num_classes = self.num_classes_sot
                elif self.mode == "mot":
                    num_classes = self.num_classes
                else:
                    raise ValueError()
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
            if return_ota:
                if num_gt == 0:
                    outputs_ota.append(None)
                else:
                    with torch.no_grad():
                        output_ota = torch.stack([torch.nonzero(fg_mask, as_tuple=True)[0], matched_gt_inds], dim=-1) # (K, 2) K is the total matched number
                        outputs_ota.append(output_ota.detach())
                # outputs_ota.append()
        # t_assign = time.time()
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        if self.mode == "sot":
            num_classes = self.num_classes_sot
        elif self.mode == "mot":
            num_classes = self.num_classes
        else:
            raise ValueError()
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        # t_loss = time.time()
        # print("SOT---prepare: %.4f, assign: %.4f, loss: %.4f" %(t_prepare-t_s, t_assign-t_prepare, t_loss-t_assign))
        if return_ota:
            return (loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1),), outputs_ota
        else:
            return (
                loss,
                reg_weight * loss_iou,
                loss_obj,
                loss_cls,
                loss_l1,
                num_fg / max(num_gts, 1),
            )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        img_size = imgs.shape[2:]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            img_size
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        if self.mode == "sot":
            num_classes = self.num_classes_sot
        elif self.mode == "mot":
            num_classes = self.num_classes
        else:
            raise ValueError()
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        img_size
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5
        # clip center inside image
        gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
        gt_bboxes_per_image_clip[:, 0] = torch.clamp(gt_bboxes_per_image_clip[:, 0], min=0, max=img_size[1])
        gt_bboxes_per_image_clip[:, 1] = torch.clamp(gt_bboxes_per_image_clip[:, 1], min=0, max=img_size[0])

        gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        del gt_bboxes_per_image_clip
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
