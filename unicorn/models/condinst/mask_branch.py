from typing import Dict
import math

import torch
from torch import nn
from torch.nn import functional as F
from .conv_with_kaiming_uniform import conv_with_kaiming_uniform
from .comm import aligned_bilinear

INF = 100000000


def build_mask_branch(cfg, in_channels=(192, 384, 768), sem_loss_on=True, use_raft=False, up_rate=8):
    return MaskBranch(cfg, in_channels, sem_loss_on=sem_loss_on, use_raft=use_raft, up_rate=up_rate)


class MaskBranch(nn.Module):
    def __init__(self, cfg, in_channels=(192, 384, 768), sem_loss_on=True, use_raft=False, up_rate=8):
        super().__init__()
        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES # ["p3", "p4", "p5"]
        self.sem_loss_on = sem_loss_on # True
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS # 8
        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM # BN
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS # 4
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS # 128
        if use_raft:
            self.out_stride = up_rate
        else:
            self.out_stride = 2 # original value is 8 (compared with 4x downsampled mask, here should be 2)
        self.use_raft = use_raft
        self.up_rate = up_rate

        feature_channels = {"p3": in_channels[0], "p4": in_channels[1], "p5": in_channels[2]}

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature],
                channels, 3, 1
            ))

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

        if self.use_raft:
            self.up_mask_layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, self.up_rate*self.up_rate*9, 1, padding=0))

        if self.sem_loss_on:
            num_classes = cfg.MODEL.FCOS.NUM_CLASSES
            self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

            in_channels = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1)
            )

            self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, features, gt_bitmasks_full=None, gt_classes=None):
        """gt_bitmasks_full: (bs, M, H, W), gt_classes: (bs, M)"""
        # NOTE: gt_bitmasks_full has been downsampled by 4 (to reduce latency)
        # Here CondInst uses multiple-level features (p3, p4, p5)
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.sem_loss_on:
            logits_pred = self.logits(self.seg_head(
                features[self.in_features[0]]
            ))

            # compute semantic targets
            semantic_targets = []
            for (gt_mask, gt_cls) in zip(gt_bitmasks_full, gt_classes):
                # gt_mask: (M, H, W), gt_cls: (M, )
                h, w = gt_mask.size()[-2:]
                valid_mask = (gt_cls > 0) # (M, )
                num_valid = torch.sum(valid_mask)
                if num_valid > 0:
                    # only keep mask and class label of the valid instances
                    gt_mask = gt_mask[valid_mask]
                    gt_cls = gt_cls[valid_mask]
                    # prepare semantic labels
                    areas = gt_mask.sum(dim=-1).sum(dim=-1) # (N, )
                    areas = areas[:, None, None].repeat(1, h, w) # (N, H, W)
                    areas[gt_mask == 0] = INF
                    areas = areas.permute(1, 2, 0).reshape(h * w, -1) # (HW, N)
                    min_areas, inds = areas.min(dim=1) # inds: (HW, ) value is between 0 and N-1
                    per_im_sematic_targets = gt_cls[inds] # (HW, ) gt_cls is between 1 and 91(81)
                    per_im_sematic_targets[min_areas == INF] = 0
                    per_im_sematic_targets = per_im_sematic_targets.reshape(h, w) # (H, W)
                else:
                    per_im_sematic_targets = torch.zeros((h, w), device=gt_mask.device)
                semantic_targets.append(per_im_sematic_targets)
            semantic_targets = torch.stack(semantic_targets, dim=0) # (bs, H, W)

            # resize target to reduce memory
            semantic_targets = semantic_targets[
                               :, None, self.out_stride // 2::self.out_stride,
                               self.out_stride // 2::self.out_stride
                               ] # downsample 1/8 -> (bs, 1, H/8, W/8)

            # prepare one-hot targets
            num_classes = logits_pred.size(1)
            class_range = torch.arange(
                num_classes, dtype=logits_pred.dtype,
                device=logits_pred.device
            )[:, None, None] # (C, 1, 1)
            class_range = class_range + 1
            # print(semantic_targets.size(), class_range.size())
            one_hot = (semantic_targets == class_range).float() # (bs, C, H/8, W/8)
            # print(one_hot.size())
            num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

            loss_sem = sigmoid_focal_loss_jit(
                logits_pred, one_hot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_pos
            losses['loss_sem'] = loss_sem
        if self.use_raft:
            up_masks = self.up_mask_layer(x) # weights used for upsampling the coarse mask predictions
            return mask_feats, losses, up_masks
        else:
            return mask_feats, losses

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_focal_loss)
