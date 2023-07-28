# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      :

import math
import torch
from torch import nn
import torch.nn.functional as F
from ._core import PCEncoderBlock


class PCTransformer(nn.Module):
    def __init__(self, n_layers=2, n_heads=4, d_model=64, d_ff=2048, sub_k_list=[20, 20], epsilon=0.01, dropout_p=0.1):
        super().__init__()
        assert isinstance(sub_k_list, list) and n_layers == len(sub_k_list)
        self.encoder_blocks = nn.ModuleList(
            [PCEncoderBlock(d_model, n_heads, d_ff, sub_k_list[i], epsilon, dropout_p) for i in range(n_layers)]
        )
        self.last_layernorm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, x_embed, pos_embed, neg_embed):
        embed = self._calculate(x_embed)
        pos_logits = (embed * pos_embed).sum(dim=-1)
        neg_logits = (embed * neg_embed).sum(dim=-1)
        return pos_logits, neg_logits, embed

    def predict(self, x_embed):
        embed = self._calculate(x_embed)
        return embed

    def _calculate(self, embed):
        for encoder_block in self.encoder_blocks:
            embed = encoder_block(embed)
        embed = self.last_layernorm(embed)
        return embed
