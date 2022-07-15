#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2022 ByteDance. All Rights Reserved.

from loguru import logger
import torch.nn as nn
import torch.distributed as dist
import torch
import torch.nn.functional as F
import copy



class Unicorn(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, pos_emb=None, transformer=None, \
        interact_mode="deform", embed_dim=128, bidirect=True, grid_sample=True, mhs=False, mhs_weight=0.5, \
        scale_all_mot=False, sot_weight=1.0, mot_weight=1.0, mhs_max_inst=1000, init_with_mask=False, d_rate=4):
        """
        mhs_max_inst: max number of instances used in mhs
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        """additional modules"""
        self.bidirect = bidirect
        self.grid_sample = grid_sample
        backbone_dim = self.backbone.in_channels[1]
        hidden_dim = 256
        self.bottleneck = nn.Sequential(
                    nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),)
        nn.init.xavier_uniform_(self.bottleneck[0].weight, gain=1)
        nn.init.constant_(self.bottleneck[0].bias, 0)
        self.upsample_layer = nn.Sequential(nn.PixelShuffle(2),
                                            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1))
        self.pos_emb = pos_emb
        self.transformer = transformer
        self.mhs = mhs
        self.mhs_weight = mhs_weight
        self.interact_mode = interact_mode
        self.scale_all_mot = scale_all_mot
        self.mot_weight = mot_weight
        self.sot_weight = sot_weight
        self.mhs_max_inst = mhs_max_inst
        self.num_classes = head.num_classes
        self.init_with_mask = init_with_mask
        if self.init_with_mask:
            self.d_rate = d_rate
            self.scale_factor = 1/(8//self.d_rate)

    def forward(self, imgs=None, run_fpn=True, seq_dict0=None, seq_dict1=None, feat=None, fpn_feats=None, labels=None,  \
        embed_0=None, embed_1=None, task_ids=None, masks=None, mode="whole"):
        if mode == "backbone":
            return self.forward_backbone(imgs, run_fpn)
        elif mode == "interaction":
            if self.interact_mode == "deform":
                return self.forward_deform_interact(seq_dict0, seq_dict1)
            elif self.interact_mode == "conv":
                return self.forward_conv_interact(seq_dict0, seq_dict1)
            elif self.interact_mode == "full":
                return self.forward_full_attention(seq_dict0, seq_dict1)
            else:
                raise ValueError()
        elif mode == "upsample":
            return self.forward_upsample(feat)
        elif mode == "loss":
            assert torch.all(torch.logical_or(task_ids==1, task_ids==2)) 
            # We adopt alternative training by default.
            # Dduring SOT-MOT training, take "sot" as 1, "mot" as 2
            # During VOS-MOTS training, take "vos" as 1, "mots" as 2
            sot_mask = (task_ids==1)
            mot_mask = (task_ids==2)
            total_loss_aux = torch.tensor(0.0, device="cuda")
            loss_dict_sot = {"total_loss": torch.tensor(0.0, device="cuda"), "iou_loss": torch.tensor(0.0, device="cuda"), 
            "l1_loss": torch.tensor(0.0, device="cuda"), "conf_loss": torch.tensor(0.0, device="cuda"), 
            "cls_loss": torch.tensor(0.0, device="cuda"), "corr_loss": torch.tensor(0.0, device="cuda")}
            if masks is not None:
                loss_dict_sot["condinst_loss"] = torch.tensor(0.0, device="cuda")
                loss_dict_sot["sem_loss"] = torch.tensor(0.0, device="cuda")
            loss_dict_mot = copy.deepcopy(loss_dict_sot)
            # t_start = time.time()
            if torch.sum(sot_mask) > 0:
                fpn_feats_sot = [x[sot_mask] for x in fpn_feats]
                if masks is not None:
                    loss_dict_sot = self.compute_loss_vos(embed_0[sot_mask], embed_1[sot_mask], fpn_feats_sot, labels[sot_mask], imgs[sot_mask], masks[sot_mask])
                else:
                    loss_dict_sot = self.compute_loss_sot(embed_0[sot_mask], embed_1[sot_mask], fpn_feats_sot, labels[sot_mask], imgs[sot_mask])
            # t_sot = time.time()
            if torch.sum(mot_mask) > 0:
                fpn_feats_mot = [x[mot_mask] for x in fpn_feats]
                if masks is not None:
                    loss_dict_mot = self.compute_loss_mot(embed_0[mot_mask], embed_1[mot_mask], fpn_feats_mot, labels[mot_mask], imgs[mot_mask], masks=masks[mot_mask])
                else:
                    loss_dict_mot = self.compute_loss_mot(embed_0[mot_mask], embed_1[mot_mask], fpn_feats_mot, labels[mot_mask], imgs[mot_mask])
                if self.mhs:
                    sot_label = torch.zeros_like(labels[mot_mask])
                    ref_tids = labels[mot_mask][:, 0, :, 5] # (b, M)
                    cur_tids = labels[mot_mask][:, 1, :, 5] # (b, M)
                    for b in range(torch.sum(mot_mask)):
                        for i in range(ref_tids.size(1)):
                            matched = False
                            for j in range(cur_tids.size(1)):
                                if ref_tids[b, i] == cur_tids[b, j]:
                                    sot_label[b, 0, 0, 1:6] = labels[mot_mask][b, 0, i, 1:6]
                                    sot_label[b, 1, 0, 1:6] = labels[mot_mask][b, 1, j, 1:6]
                                    matched = True
                                    break
                            if matched:
                                break
                    total_loss_aux = self.compute_loss_sot(embed_0[mot_mask], embed_1[mot_mask], fpn_feats_mot, sot_label, imgs[mot_mask])["total_loss"] * self.mhs_weight
            if self.scale_all_mot:
                loss_dict_mot["total_loss"] = self.mot_weight * loss_dict_mot["total_loss"]
            if self.sot_weight != 1.0:
                loss_dict_sot["total_loss"] = self.sot_weight * loss_dict_sot["total_loss"]
            total_loss = (torch.sum(sot_mask) * loss_dict_sot["total_loss"] + torch.sum(mot_mask) * loss_dict_mot["total_loss"]) / len(task_ids) + total_loss_aux
            loss_dict_final = {"total_loss": total_loss}
            for k, v in loss_dict_sot.items():
                if k != "total_loss":
                    loss_dict_final[k+"_sot"] = v
            for k, v in loss_dict_mot.items():
                if k != "total_loss":
                    loss_dict_final[k+"_mot"] = v
            return loss_dict_final
        elif mode == "whole":
            bs, _, H, W = imgs.size() # batch size and number of frames
            fpn_outs, seq_dict = self.forward_backbone(imgs, run_fpn=True)
            pred_lbs1_ms = (torch.zeros((bs, 1, H//8, W//8), device="cuda"), 
            torch.zeros((bs, 1, H//16, W//16), device="cuda"),
            torch.zeros((bs, 1, H//32, W//32), device="cuda")) # [8, 16, 32]
            return self.head(fpn_outs, pred_lbs1_ms, mode="mot"), seq_dict
        elif mode == "debug":
            """debug intermediate results"""
            import sys
            import os
            import cv2
            import numpy as np
            save_dir = "/opt/tiger/omnitrack"
            targets = labels.float()
            bs = targets.size(0)
            trackids = targets[:, :, :, 5]
            valid_mask = trackids != 0 # (bs, F, M)
            num_valids = valid_mask.sum(dim=-1) # (bs, F)
            for b in range(bs):
                trackids_0, trackids_1 = trackids[b, 0], trackids[b, 1] # (M, )
                n0, n1 = num_valids[b, 0].item(), num_valids[b, 1].item()
                # get id (one-hot) labels
                if self.bidirect:
                    cur_label_row = torch.full((n0, ), -1, dtype=torch.long, device="cuda")
                    cur_label_col = torch.full((n1, ), -1, dtype=torch.long, device="cuda")
                else:
                    cur_label = torch.full((n0, ), -1, dtype=torch.long, device="cuda")
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                img0, img1 = imgs[b, 0], imgs[b, 1] # (3, H, W)
                img0_np = img0.clamp(0, 255).cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
                img1_np = img1.clamp(0, 255).cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
                img0_np = np.ascontiguousarray(img0_np)
                img1_np = np.ascontiguousarray(img1_np)
                for i in range(n0):
                    if masks is not None:
                        masks0_np = masks[b, 0, i].cpu().numpy() # (H, W)
                        path_mask0 = os.path.join(save_dir, "rank_%d_batch_%d_%d_mask_%02d_ref.jpg" %(dist.get_rank(), b, 0, i))
                        cv2.imwrite(path_mask0, 255*masks0_np)
                    for j in range(n1):
                        if trackids_0[i] == trackids_1[j]:
                            if masks is not None:
                                masks1_np = masks[b, 1, j].cpu().numpy() # (H, W)
                                path_mask1 = os.path.join(save_dir, "rank_%d_batch_%d_%d_mask_%02d_cur.jpg" %(dist.get_rank(), b, 0, i))
                                cv2.imwrite(path_mask1, 255*masks1_np)
                            cx0, cy0, w0, h0 = targets[b, 0, i, 1:5].tolist()
                            cx1, cy1, w1, h1 = targets[b, 1, j, 1:5].tolist()
                            color_idx = (i+j) % 3
                            color = colors[color_idx]
                            cv2.rectangle(img0_np, (int(cx0-w0/2),int(cy0-h0/2)), (int(cx0+w0/2),int(cy0+h0/2)), color, thickness=2)
                            cv2.rectangle(img1_np, (int(cx1-w1/2),int(cy1-h1/2)), (int(cx1+w1/2),int(cy1+h1/2)), color, thickness=2)
                            if self.bidirect:
                                cur_label_row[i] = j
                                cur_label_col[j] = i
                            else:
                                cur_label[i] = j
                            break
                path0 = os.path.join(save_dir, "rank_%d_batch_%d_%d.jpg" %(dist.get_rank(), b, 0))
                path1 = os.path.join(save_dir, "rank_%d_batch_%d_%d.jpg" %(dist.get_rank(), b, 1))
                cv2.imwrite(path0, img0_np)
                cv2.imwrite(path1, img1_np)
            sys.exit(0)
        elif mode == "debug_youtube":
            """debug intermediate results"""
            import sys
            import os
            import cv2
            import numpy as np
            save_dir = "/opt/tiger/omnitrack"
            targets = labels.float()
            bs = targets.size(0)
            trackids = targets[:, :, :, 5]
            valid_mask = trackids != 0 # (bs, F, M)
            num_valids = valid_mask.sum(dim=-1) # (bs, F)
            for b in range(bs):
                trackids_0, trackids_1 = trackids[b, 0], trackids[b, 1] # (M, )
                n0, n1 = num_valids[b, 0].item(), num_valids[b, 1].item()
                print(n0, n1)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                img0, img1 = imgs[b, 0], imgs[b, 1] # (3, H, W)
                img0_np = img0.clamp(0, 255).cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
                img1_np = img1.clamp(0, 255).cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
                img0_np = np.ascontiguousarray(img0_np)
                img1_np = np.ascontiguousarray(img1_np)
                for i in range(n0):
                    cx0, cy0, w0, h0 = targets[b, 0, i, 1:5].tolist()
                    cv2.rectangle(img0_np, (int(cx0-w0/2),int(cy0-h0/2)), (int(cx0+w0/2),int(cy0+h0/2)), colors[-1], thickness=2)
                    path0 = os.path.join(save_dir, "rank_%d_batch_%d_%d.jpg" %(dist.get_rank(), b, 0))
                    cv2.imwrite(path0, img0_np)
                for j in range(n1):      
                    cx1, cy1, w1, h1 = targets[b, 1, j, 1:5].tolist()
                    cv2.rectangle(img1_np, (int(cx1-w1/2),int(cy1-h1/2)), (int(cx1+w1/2),int(cy1+h1/2)), colors[-1], thickness=2)
                    path1 = os.path.join(save_dir, "rank_%d_batch_%d_%d.jpg" %(dist.get_rank(), b, 1))
                    cv2.imwrite(path1, img1_np)
            sys.exit(0)
        else:
            raise ValueError

    def forward_backbone(self, img: torch.Tensor, run_fpn=True):
        """The input type is standard tensor
               - img: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(img, torch.Tensor)
        """run the backbone"""
        if run_fpn:
            fpn_outs, base_outs = self.backbone(img, return_base_feat=True, run_fpn=True)
        else:
            base_outs = self.backbone(img, return_base_feat=True, run_fpn=False)
        feat_s16 = base_outs[1] # pick feature map with stride 16 from [8, 16, 32]

        # features & masks, position embedding for the current image
        bs, _, h, w = feat_s16.size()
        if self.pos_emb is not None:
            """get the positional encoding"""
            pos = self.pos_emb(bs, h, w)  # (B, C, H, W)
            """Interpolate positional encoding according to input size"""
            pos = F.interpolate(pos, size=(h, w), mode='bicubic')
            """adjust dimension and merge feat, pos, and mask"""
            seq_dict = {"feat": feat_s16, "pos": pos, "h": h, "w": w}
        else:
            seq_dict = {"feat": feat_s16, "h": h, "w": w}
        if run_fpn:
            return fpn_outs, seq_dict
        else:
            return seq_dict

    def forward_deform_interact(self, feat_dict1, feat_dict2):
        # Forward the transformer encoder. out_seq shape (bs, hw1+hw2, c)
        feat1, pos1 = feat_dict1["feat"], feat_dict1["pos"]
        feat2, pos2 = feat_dict2["feat"], feat_dict2["pos"]

        srcs = [self.bottleneck(feat1), self.bottleneck(feat2)]
        pos = [pos1, pos2]

        out_seq = self.transformer(srcs, pos)
        # Split the sequence and reshape back to feature maps
        bs, seq_len, c = out_seq.size()
        half_len = seq_len // 2
        seq1, seq2 = out_seq[:, :half_len], out_seq[:, half_len:]
        h, w = feat_dict1["h"], feat_dict1["w"]
        new_feat1 = seq1.permute(0, 2, 1).view(bs, c, h, w)
        new_feat2 = seq2.permute(0, 2, 1).view(bs, c, h, w)
        return new_feat1, new_feat2

    def forward_full_attention(self, feat_dict1, feat_dict2):
        """full attention"""
        # Forward the transformer encoder. out_seq shape (bs, hw1+hw2, c)
        feat1, pos1 = feat_dict1["feat"], feat_dict1["pos"]
        feat2, pos2 = feat_dict2["feat"], feat_dict2["pos"]

        srcs = [self.bottleneck(feat1), self.bottleneck(feat2)]
        pos = [pos1, pos2]

        srcs_new = [x.flatten(-2).permute((2, 0, 1)) for x in srcs]
        pos_new = [x.flatten(-2).permute((2, 0, 1)) for x in pos]

        out_seq = self.transformer(torch.cat(srcs_new, dim=0), mask=None, pos_embed=torch.cat(pos_new, dim=0))
        # Split the sequence and reshape back to feature maps
        seq_len, bs, c = out_seq.size()
        half_len = seq_len // 2
        seq1, seq2 = out_seq[:half_len], out_seq[half_len:]
        h, w = feat_dict1["h"], feat_dict1["w"]
        feat1 = seq1.permute(1, 2, 0).view(bs, c, h, w)
        feat2 = seq2.permute(1, 2, 0).view(bs, c, h, w)
        return feat1, feat2

    def forward_conv_interact(self, feat_dict1, feat_dict2):
        # Forward the conv interaction module. out_seq shape (bs, hw1+hw2, c)
        feat1 = feat_dict1["feat"]
        feat2 = feat_dict2["feat"]

        srcs = [self.bottleneck(feat1), self.bottleneck(feat2)]

        out_seq = self.transformer(srcs)

        return out_seq[0], out_seq[1]

    def forward_upsample(self, x):
        # 2x up-sampling (stride=16 --> stride=8)
        return self.upsample_layer(x)

    def compute_loss_sot(self, embed_0, embed_1, fpn_outs_1, targets, images, s=8, propagate=True):
        bs, nF, _, H, W = images.size() # batch size and number of frames
        H_d, W_d = H//s, W//s
        # t_start = time.time()
        if propagate:
            """ compute correspondence and propagate labels """
            simi_mat = torch.bmm(embed_0.flatten(-2).transpose(-1, -2), embed_1.flatten(-2)) # (b, H/8*W/8, H/8*W/8)
            trans_mat_01 = torch.softmax(simi_mat, dim=1)  # transfer matrix 0->1  (b, H/8*W/8, H/8*W/8)
            """Propagation and running detection heads"""
            gt_lbs_0 = F.interpolate(get_label_map(targets[:, 0, 0, 1:5], H, W, bs), scale_factor=1/8, mode="bilinear", align_corners=False).flatten(-2).cuda()
            # propagation and detection (0->1)
            pred_lbs1 = torch.bmm(gt_lbs_0, trans_mat_01).view(bs, 1, H_d, W_d) # (b, K, H/8*W/8) * (b, H/8*W/8, H/8*W/8) -> (b, K, H/8*W/8) -> (b, K, H/8, W/8)
        else:
            pred_lbs1 = torch.zeros((bs, 1, H_d, W_d), device="cuda")
        pred_lbs1_ms = (pred_lbs1, 
        F.interpolate(pred_lbs1, scale_factor=1/2, mode="bilinear", align_corners=False),
        F.interpolate(pred_lbs1, scale_factor=1/4, mode="bilinear", align_corners=False)) # [8, 16, 32]
        gt_lbs_1 = F.interpolate(get_label_map(targets[:, 1, 0, 1:5], H, W, bs), scale_factor=1/8, mode="bilinear", align_corners=False).flatten(-2).cuda()
        loss_dict1 = self.head(fpn_outs_1, pred_lbs1_ms, targets[:, 1, :, :5], images[:, 1], mode="sot")
        loss_dict1["corr_loss"] = compute_corr_losses(pred_lbs1, gt_lbs_1)
        # merging
        loss_dict1["total_loss"] += loss_dict1["corr_loss"]
        return loss_dict1

    def compute_loss_vos(self, embed_0, embed_1, fpn_outs_1, targets, images, masks, s=8, return_trans_mat=False):
        bs, nF, _, H, W = images.size() # batch size and number of frames
        H_d, W_d = H//s, W//s
        simi_mat = torch.bmm(embed_0.flatten(-2).transpose(-1, -2), embed_1.flatten(-2)) # (b, H/8*W/8, H/8*W/8)
        trans_mat_01 = torch.softmax(simi_mat, dim=1)  # transfer matrix 0->1  (b, H/8*W/8, H/8*W/8)
        loss_dict_list = []
        if self.mhs and self.mhs_wo_head:
            max_matched_inst = self.mhs_max_inst
        else:
            max_matched_inst = 1000 # just a large number
        for b in range(bs):
            num_matched = 0
            stop_flag = False
            tkids_r, tkids_c = targets[b, 0, :, 5], targets[b, 1, :, 5] # (M, )
            for i, id_r in enumerate(tkids_r):
                if stop_flag:
                    break
                matched = False
                if id_r != 0:
                    for j, id_c in enumerate(tkids_c):
                        if (id_c != 0) and (id_c == id_r):
                            matched = True
                            break
                if matched:
                    # num_inst += 1
                    if self.init_with_mask:
                        gt_lbs_0 = F.interpolate(masks[b:b+1, 0, i:i+1], scale_factor=self.scale_factor, mode="bilinear", align_corners=False).flatten(-2).cuda()
                        gt_lbs_1 = F.interpolate(masks[b:b+1, 1, j:j+1], scale_factor=self.scale_factor, mode="bilinear", align_corners=False).flatten(-2).cuda()
                    else:
                        gt_lbs_0 = F.interpolate(get_label_map(targets[b:b+1, 0, i, 1:5], H, W, bs=1, device="cuda"), scale_factor=1/8, mode="bilinear", align_corners=False).flatten(-2)
                        gt_lbs_1 = F.interpolate(get_label_map(targets[b:b+1, 1, j, 1:5], H, W, bs=1, device="cuda"), scale_factor=1/8, mode="bilinear", align_corners=False).flatten(-2)
                    # propagation and detection (0->1)
                    pred_lbs1 = torch.bmm(gt_lbs_0, trans_mat_01[b:b+1]).view(1, 1, H_d, W_d) # (b, K, H/8*W/8) * (b, H/8*W/8, H/8*W/8) -> (b, K, H/8*W/8) -> (b, K, H/8, W/8)
                    pred_lbs1_ms = (pred_lbs1, 
                    F.interpolate(pred_lbs1, scale_factor=1/2, mode="bilinear", align_corners=False),
                    F.interpolate(pred_lbs1, scale_factor=1/4, mode="bilinear", align_corners=False)) # [8, 16, 32]
                    fpn_feats = tuple([x[b:b+1] for x in fpn_outs_1])
                    labels = torch.zeros((1, 1, 5), device="cuda") # (1, M, 5)
                    labels[:, :, 1:5] = targets[b:b+1, 1, j:j+1, 1:5]
                    loss_dict1 = self.head(fpn_feats, pred_lbs1_ms, labels, images[:, 1], masks=masks[b:b+1, 1, j:j+1], mode="sot")
                    loss_dict1["corr_loss"] = compute_corr_losses(pred_lbs1, gt_lbs_1)
                    # merging
                    loss_dict1["total_loss"] += loss_dict1["corr_loss"]
                    loss_dict_list.append(loss_dict1)
                    num_matched += 1
                    if num_matched == max_matched_inst:
                        stop_flag = True
        if len(loss_dict_list) == 0:
            loss_dict_list.append({"total_loss": torch.tensor(0.0, device="cuda"), "iou_loss": torch.tensor(0.0, device="cuda"), 
            "l1_loss": torch.tensor(0.0, device="cuda"), "conf_loss": torch.tensor(0.0, device="cuda"), 
            "cls_loss": torch.tensor(0.0, device="cuda"), "corr_loss": torch.tensor(0.0, device="cuda")})
        return average_dict(loss_dict_list)

    def compute_loss_mot(self, embed_0, embed_1, fpn_outs_1, targets, images, s=8, masks=None):
        bs, nF, _, H, W = images.size() # batch size and number of frames
        H_d, W_d = H//s, W//s
        pred_lbs1_ms = (torch.zeros((bs, 1, H//8, W//8), device="cuda"), 
        torch.zeros((bs, 1, H//16, W//16), device="cuda"),
        torch.zeros((bs, 1, H//32, W//32), device="cuda")) # [8, 16, 32]
        if masks is not None:
            loss_dict = self.head(fpn_outs_1, pred_lbs1_ms, targets[:, 1], images[:, 1], masks=masks[:, 1], mode="mot")
        else:
            loss_dict = self.head(fpn_outs_1, pred_lbs1_ms, targets[:, 1], images[:, 1], mode="mot")
        loss_dict["corr_loss"] = self.compute_loss_mot_corr(embed_0, embed_1, targets, bs, s, H_d, W_d)
        # merging
        loss_dict["total_loss"] += loss_dict["corr_loss"]
        return loss_dict

    def compute_loss_mot_corr(self, embed_0, embed_1, targets, bs, s, H_d, W_d):
        corr_loss_tensor = torch.zeros((bs,)).cuda()
        targets = targets.float()
        trackids = targets[:, :, :, 5]
        valid_mask = trackids != 0 # (bs, F, M)
        num_valids = valid_mask.sum(dim=-1) # (bs, F)
        for b in range(bs):
            trackids_0, trackids_1 = trackids[b, 0], trackids[b, 1] # (M, )
            n0, n1 = num_valids[b, 0].item(), num_valids[b, 1].item()
            # get id (one-hot) labels
            if self.bidirect:
                cur_label_row = torch.full((n0, ), -1, dtype=torch.long, device="cuda")
                cur_label_col = torch.full((n1, ), -1, dtype=torch.long, device="cuda")
            else:
                cur_label = torch.full((n0, ), -1, dtype=torch.long, device="cuda")
            for i in range(n0):
                for j in range(n1):
                    if trackids_0[i] == trackids_1[j]:
                        if self.bidirect:
                            cur_label_row[i] = j
                            cur_label_col[j] = i
                        else:
                            cur_label[i] = j
                        break
            """ get instance embeddings """
            embed_list0 = []
            for i in range(n0):
                if self.grid_sample:
                    cx, cy = targets[b, 0, i, 1:3] / s - 0.5
                    cx = (torch.clamp(cx, min=0, max=W_d-1) / (W_d-1) - 0.5) * 2.0 # range of [-1, 1]
                    cy = (torch.clamp(cy, min=0, max=H_d-1) / (H_d-1) - 0.5) * 2.0 # range of [-1, 1]
                    grid = torch.stack([cx, cy], dim=-1).view(1, 1, 1, 2) # (1, 1, 1, 2)
                    embed_list0.append(F.grid_sample(embed_0[b:b+1], grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze()) \
                        # (1, C, 1, 1) -> (C, )
                else:
                    cx, cy = targets[b, 0, i, 1:3] / s
                    cx = torch.round(torch.clamp(cx, min=0, max=W_d-1)).long()
                    cy = torch.round(torch.clamp(cy, min=0, max=H_d-1)).long()
                    embed_list0.append(embed_0[b, :, cy, cx])
            embed_list1 = []
            for j in range(n1):
                if self.grid_sample:
                    cx, cy = targets[b, 1, j, 1:3] / s - 0.5
                    cx = (torch.clamp(cx, min=0, max=W_d-1) / (W_d-1) - 0.5) * 2.0 # range of [-1, 1]
                    cy = (torch.clamp(cy, min=0, max=H_d-1) / (H_d-1) - 0.5) * 2.0 # range of [-1, 1]
                    grid = torch.stack([cx, cy], dim=-1).view(1, 1, 1, 2) # (1, 1, 1, 2)
                    embed_list1.append(F.grid_sample(embed_1[b:b+1], grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze()) \
                        # (1, C, 1, 1) -> (C, )
                else:
                    cx, cy = targets[b, 1, j, 1:3] / s
                    cx = torch.round(torch.clamp(cx, min=0, max=W_d-1)).long()
                    cy = torch.round(torch.clamp(cy, min=0, max=H_d-1)).long()
                    embed_list1.append(embed_1[b, :, cy, cx])
            # compute losses
            simi_mat = torch.stack(embed_list0, dim=0) @ torch.stack(embed_list1, dim=0).transpose(0, 1) # (M, C) * (C, N) -> (M, N)
            if self.bidirect:
                corr_loss_tensor[b] = 0.5 * (F.cross_entropy(simi_mat, cur_label_row, ignore_index=-1) + F.cross_entropy(simi_mat.transpose(0, 1), cur_label_col, ignore_index=-1))
            else:
                corr_loss_tensor[b] = F.cross_entropy(simi_mat, cur_label, ignore_index=-1)
        return torch.mean(corr_loss_tensor)



class UnicornActor:
    """ The actor class handles the passing of the data through the network and calculation the loss"""
    def __init__(self, net):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net

    def __call__(self, images, targets, task_ids, masks=None):
        """images: (b, F, 3, H, W), targets: (b, F, M, 6), task_ids: (b, 1) M is max number of instances
        targets: [cls, bbox, trackid]
        """
        targets = targets.float() # Half to Float
        task_ids = task_ids.squeeze(-1) # (bs, )
        bs, nF, _, H, W = images.size() # batch size and number of frames
        assert nF == 2 # for now, we require the number of frames equal to 2
        """forward backbone"""
        fpn_outs, out_dict = self.net(images.transpose(0, 1).contiguous().view(-1, 3, H, W), mode="backbone", run_fpn=True) # (2*bs, 3, H, W)
        fpn_outs_1 = (fpn_outs[0][bs:], fpn_outs[1][bs:], fpn_outs[2][bs:]) # pick FPN feature of the current frame
        with torch.cuda.amp.autocast(enabled=False):
            """ feature interaction """
            out_dict_0, out_dict_1 = {}, {}
            out_dict_0["feat"] = out_dict["feat"][:bs].float()
            out_dict_0["h"], out_dict_0["w"] = out_dict["h"], out_dict["w"]
            out_dict_1["feat"] = out_dict["feat"][bs:].float()
            out_dict_1["h"], out_dict_1["w"] = out_dict["h"], out_dict["w"]
            if "pos" in out_dict:
                out_dict_0["pos"] = out_dict["pos"][:bs].float()
                out_dict_1["pos"] = out_dict["pos"][bs:].float()
            new_feat_0, new_feat_1 = self.net(seq_dict0=out_dict_0, seq_dict1=out_dict_1, mode="interaction")
            """ up-sampling --> embedding"""
            embed_0 = self.net(feat=new_feat_0, mode="upsample")  # (b, C, H/8, W/8)
            embed_1 = self.net(feat=new_feat_1, mode="upsample")  # (b, C, H/8, W/8)
        loss_dict_final = self.net(imgs=images, fpn_feats=fpn_outs_1, labels=targets, embed_0=embed_0, embed_1=embed_1, 
        task_ids=task_ids, masks=masks, mode="loss")
        return loss_dict_final

def compute_corr_losses(pred_lbs, gt_lbs):
    return dice_coefficient(pred_lbs, gt_lbs)

def dice_coefficient(pred_lbs, gt_lbs):
    eps = 1e-5
    x = pred_lbs.reshape(-1)
    target = gt_lbs.reshape(-1)
    intersection = (x * target).sum()
    union = (x ** 2.0).sum() + (target ** 2.0).sum() + eps
    loss = 1. - (2 * intersection / union)
    return loss

def get_label_map(boxes, H, W, bs, device="cuda"):
    """boxes: (bs, 4)"""
    boxes_xyxy = torch.round(box_cxcywh_to_xyxy(boxes)).int()
    labels = torch.zeros((bs, 1, H, W), dtype=torch.float32, device=device)
    for b in range(bs):
        x1, y1, x2, y2 = boxes_xyxy[b].tolist()
        x1, y1 = max(0, x1), max(0, y1)
        try:
            labels[b, 0, y1:y2, x1:x2] = 1.0
        except:
            print("too small bounding box")
            pass
    return labels # (bs, 1, H, W)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def average_dict(dict_list):
    n = len(dict_list)
    new_dict = {}
    for k in dict_list[0].keys():
        value_list = [sub_dict[k] for sub_dict in dict_list]
        new_dict[k] = 1/n * sum(value_list)
    return new_dict

