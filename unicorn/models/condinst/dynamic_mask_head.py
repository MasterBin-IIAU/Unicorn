import torch
from torch.nn import functional as F
from torch import nn

from .comm import compute_locations, aligned_bilinear

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    # weight_nums: [80, 64, 8], bias_num: [8, 8, 1]
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg, use_raft=False, up_rate=8):
    return DynamicMaskHead(cfg, use_raft=use_raft, up_rate=up_rate)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg, use_raft=False, up_rate=8):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS # 3
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS # 8
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS # 8
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE # 4
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS # False
        self.use_raft = use_raft
        self.up_rate = up_rate
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST # [64, 128, 256, 512]
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2])) # [64, 128, 256, 512, 1024]

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            ) # implemented by Grouped Conv. Each Instance is a group.
            if i < n_layers - 1:
                x = F.relu(x)
        return x
    
    
    def upsample_preds(self, pred, mask):
        """ Upsample pred [N, 1, H/8, W/8] -> [N, 1, H, W] using convex combination """
        N, _, H, W = pred.shape
        mask = mask.view(1, 1, 9, self.up_rate, self.up_rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_pred = F.unfold(pred, [3,3], padding=1)
        up_pred = up_pred.view(N, 1, 9, 1, 1, H, W)

        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, 1, self.up_rate*H, self.up_rate*W)

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, mask_head_params, instance_locations, instance_fpn_levels, up_masks):
        """
        mask_feats: (1, 8, H//8, W//8), mask_feat_stride=8,
        mask_head_params: (N, 169), instance_locations: (N, 2), instance_fpn_levels: (N, ) torch.int, 
        """
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        ) # (hw, 2)
        # n_inst = len(instances)
        n_inst = len(instance_locations)

        # im_inds = instances.im_inds
        im_inds = torch.zeros((n_inst, ), dtype=torch.long, device="cuda")

        bs, _, H, W = mask_feats.size()
        assert bs == 1

        if not self.disable_rel_coords:
            # instance_locations = instances.locations # need more details about instances
            # print("instance_locations:", instance_locations) # (N, 2)
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2) # (N, 1, 2) - (1, HW, 2) -> (N, HW, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float() # (N, 2, HW)
            soi = self.sizes_of_interest.float()[instance_fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1) # (N, 2, HW) / (N, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype) # (N, 2, HW)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1) # (N, 10, HW)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W) # (1, 10*N, H, W)
        # print("mask_head_inputs size:", mask_head_inputs.size()) # (1, N*10, h, w)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )
        # print("weights:", weights)
        # print("bias:", biases)
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W) # (N, 1, h, w)
        if self.use_raft:
            assert up_masks is not None
            mask_logits = self.upsample_preds(mask_logits, up_masks) # upsample by 8 --> the original resolution
        else:
            assert mask_feat_stride >= self.mask_out_stride
            assert mask_feat_stride % self.mask_out_stride == 0
            mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride)) # Upsample 2x -> 1/4 resolution

        return mask_logits

    def __call__(self, mask_feats, mask_feat_stride, \
        mask_head_params=None, instance_locations=None, instance_fpn_levels=None, gt_bitmasks=None, up_masks=None):
        """
        mask_feats: (1, 8, H//8, W//8), mask_feat_stride=8,
        mask_head_params: (N, 169), instance_locations: (N, 2), instance_fpn_levels: (N, ) torch.int, 
        """
        if self.training:
            self._iter += 1


            losses = {}

            # if len(pred_instances) == 0:
            #     dummy_loss = mask_feats.sum() * 0 + mask_head_params.sum() * 0
            #     if not self.boxinst_enabled:
            #         losses["loss_mask"] = dummy_loss
            #     else:
            #         losses["loss_prj"] = dummy_loss
            #         losses["loss_pairwise"] = dummy_loss
            # else:
            mask_logits = self.mask_heads_forward_with_coords(
                mask_feats, mask_feat_stride, mask_head_params, instance_locations, instance_fpn_levels, up_masks)
            mask_scores = mask_logits.sigmoid()

            if self.boxinst_enabled:
                # box-supervised BoxInst losses
                image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                pairwise_losses = compute_pairwise_term(
                    mask_logits, self.pairwise_size,
                    self.pairwise_dilation
                )

                weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                loss_pairwise = loss_pairwise * warmup_factor

                losses.update({
                    "loss_prj": loss_prj_term,
                    "loss_pairwise": loss_pairwise,
                })
            else:
                # fully-supervised CondInst losses
                # print(mask_scores.size(), gt_bitmasks.size())
                mask_losses = dice_coefficient(mask_scores, gt_bitmasks) # (N, 1, H/4, W/4)
                loss_mask = mask_losses.mean()
                losses["loss_mask"] = loss_mask

            return losses
        else:
            mask_logits = self.mask_heads_forward_with_coords(
                mask_feats, mask_feat_stride, mask_head_params, instance_locations, instance_fpn_levels, up_masks)
            mask_scores = mask_logits.sigmoid() # (N, 1, H/4, W/4)
            return mask_scores


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
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
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes