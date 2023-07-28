# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      :

import torch.nn as nn
from ._core import TransformerEncoderBlock


class Transformer(nn.Module):
    """
    Vanilla Transformer encoders 的实现

    Args:
        __init__():
            n_layers (int): encoder layers 层数
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例

        forward() and predict():
            x_embed (Tensor [batch size, series len, d_model]): 输入 item embed
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask ：1--padding item，0--有效item

    Returns:
        forward():
            embed (Tensor [batch size, series len, d_model])
    """

    def __init__(self, n_layers, n_heads, d_model, dropout_p):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p) for _ in range(n_layers)]
        )
        # 在完整的Transformer encoder-decoder实现中，返回的memory在经过若干个TransformerEncoderBlock之后
        # 最后还得经过一层LayerNor()， 这是因为memory会作为src-tgt的attention layer 的 K V，但是我们此处只是用了encoder
        # 这种情况下，是否还需要LN呢？？
        self.last_layernorm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, x_embed, padding_mask=None):
        embed = self._calculate(x_embed, padding_mask)
        return embed

    def _calculate(self, embed, padding_mask):
        for encoder_block in self.encoder_blocks:
            embed = encoder_block(embed, padding_mask=padding_mask)

        embed = self.last_layernorm(embed)

        return embed
