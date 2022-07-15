"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned. (allow users to specify the size)
    """
    def __init__(self, num_pos_feats=256, sz=20):
        super().__init__()
        self.sz = sz
        self.row_embed = nn.Embedding(sz, num_pos_feats)
        self.col_embed = nn.Embedding(sz, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, bs, dh, dw):
        """bs: batch size, dh: destination h, dw: destination w"""
        h, w = self.sz, self.sz
        i = torch.arange(w, device=self.col_embed.weight.device)
        j = torch.arange(h, device=self.row_embed.weight.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1) # (H,W,C) --> (C,H,W) --> (1,C,H,W) --> (B,C,H,W)
        return F.interpolate(pos, (dh, dw), mode="bilinear", align_corners=False) 


def build_position_encoding(hidden_dim=256, sz=40):
    N_steps = hidden_dim // 2
    position_embedding = PositionEmbeddingLearned(N_steps, sz)
    return position_embedding
