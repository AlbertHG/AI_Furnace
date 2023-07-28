# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : 

import torch.nn as nn
from ._core import SASTransformerEncoderBlock, _generate_square_subsequent_mask


class SAS(nn.Module):
    """
    SAS Transformer encoders 的实现

    Args:
        __init__():
            n_layers (int): encoder layers 层数
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例

        forward() and predict():
            x_embed (Tensor [batch size, series len, d_model]): 输入 item embed
            pos_embed (Tensor [batch size, series len, d_model]): item embed 对于的正label embed
            neg_embed (Tensor [batch size, series len, d_model]): item embed 对于的负label embed
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask ：1--padding item，0--有效item

    Returns:
        forward():
            pos_logits (Tensor [batch size, series len])
            neg_logits (Tensor [batch size, series len])
            embed (Tensor [batch size, series len, d_model])
        predict()：
            embed (Tensor [batch size, series len, d_model]):
    """

    def __init__(self, n_layers, n_heads, d_model, dropout_p):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(
            [SASTransformerEncoderBlock(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p) for _ in range(n_layers)]
        )
        # 正方形的注意力掩码，防止因果矛盾
        self.attn_mask = None
        self.last_layernorm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, x_embed, pos_embed, neg_embed, padding_mask=None):
        embed = self._calculate(x_embed, padding_mask)
        pos_logits = (embed * pos_embed).sum(dim=-1)
        neg_logits = (embed * neg_embed).sum(dim=-1)
        return pos_logits, neg_logits, embed

    def predict(self, x_embed, padding_mask=None):
        embed = self._calculate(x_embed, padding_mask)
        return embed

    def _calculate(self, embed, padding_mask):
        # embed.shape = [batch size, serise len, d_model]
        if self.attn_mask is None or self.attn_mask.size(0) != embed.shape[1]:
            # 如果没有定义mark矩阵，则根据embed的尺寸生成它
            mask = _generate_square_subsequent_mask(embed.shape[1]).to(embed.device)
            self.attn_mask = mask  # [S, S]

        for encoder_block in self.encoder_blocks:
            embed = encoder_block(embed, self.attn_mask, padding_mask)

        embed = self.last_layernorm(embed)

        return embed
