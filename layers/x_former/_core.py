# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : 

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import math

from ..activation import GELU


class PCEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, sub_k, epsilon, dropout_p=0.1):
        super().__init__()
        self.attention_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.attention_layer = PCMultiHeadedAttention(d_model, n_heads, sub_k, epsilon, dropout_p=dropout_p)
        self.forward_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = PointWiseFeedForwardByLinear(d_model, d_ff, dropout_p)

    def forward(self, embed):
        Q = self.attention_layernorm(embed)
        K, V = embed, embed
        mha_outputs = self.attention_layer(Q, K, V)[0]
        embed = Q + mha_outputs
        embed = self.forward_layernorm(embed)
        pff_output = self.forward_layer(embed)
        embed = pff_output + embed

        return embed


class BertEncoderBlock(nn.Module):
    """
    Bert Transformer 一层 encoder 的实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            embed (Tensor [batch size, series len, d_model]): 输入 item embed
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """

    def __init__(self, n_heads, d_model, dropout_p=0.1):
        super().__init__()

        self.attention = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.feed_forward = PointWiseFeedForwardByLinear(d_model=d_model, dropout_p=dropout_p)
        self.input_sublayer = SublayerConnection(size=d_model, dropout_p=dropout_p)
        self.output_sublayer = SublayerConnection(size=d_model, dropout_p=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, embed, attn_mask=None, padding_mask=None):
        embed = self.input_sublayer(
            embed, lambda _x: self.attention.forward(_x, _x, _x, attn_mask=attn_mask, padding_mask=padding_mask)
        )
        embed = self.output_sublayer(embed, self.feed_forward)
        embed = self.dropout(embed)

        return embed


class SASTransformerEncoderBlock(nn.Module):
    """
    SAS Transformer 一层 encoder 的实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            embed (Tensor [batch size, series len, d_model]): 输入 item embed
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """

    def __init__(self, n_heads, d_model, dropout_p=0.1):
        super().__init__()

        self.attention_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.attention_layer = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.forward_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = PointWiseFeedForwardByLinear(d_model=d_model, dropout_p=dropout_p)

    def forward(self, embed, attn_mask=None, padding_mask=None):

        Q = self.attention_layernorm(embed)
        K, V = embed, embed
        mha_outputs = self.attention_layer(Q, K, V, attn_mask=attn_mask, padding_mask=padding_mask)[0]
        embed = Q + mha_outputs
        embed = self.forward_layernorm(embed)
        pff_output = self.forward_layer(embed)
        embed = pff_output + embed

        return embed


class TransformerEncoderBlock(nn.Module):
    """
    Attention is all you need Transformer 一层 encoder 的标准实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            X (Tensor [batch size, series len, d_model]): 输入 item embed
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """

    def __init__(self, n_heads, d_model, dropout_p=0.1):

        super().__init__()
        self.attention_layer = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.attention_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = PointWiseFeedForwardByLinear(d_model=d_model, d_ff=d_model * 4, dropout_p=dropout_p)
        self.forward_layernorm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, embed, attn_mask=None, padding_mask=None):
        Q, K, V = embed, embed, embed
        mha_outputs = self.attention_layer(Q, K, V, attn_mask=attn_mask, padding_mask=padding_mask)[0]
        embed = Q + mha_outputs
        embed = self.attention_layernorm(embed)
        pff_output = self.forward_layer(embed)
        embed = embed + pff_output
        embed = self.forward_layernorm(embed)

        return embed


class TiSASTransformerEncoderBlock(nn.Module):
    """
    TiSAS Transformer 一层 encoder 的实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            embed (Tensor [batch size, series len, d_model]): 输入 item Embed
            time_matrix_K (Tensor [batch size, n head, query's len, key's len, head_dim]): Q 对 K 的时间间隔矩阵 Embed
            time_matrix_V (Tensor [batch size, n head, query's len, value's len, head_dim]): Q 对 V 的时间间隔矩阵 Embed
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """

    def __init__(self, n_heads, d_model, dropout_p=0.1):
        super().__init__()

        self.attention_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.attention_layer = TimeAwareMultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.forward_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = PointWiseFeedForwardByLinear(d_model=d_model, dropout_p=dropout_p)

    def forward(self, embed, time_matrix_K, time_matrix_V, attn_mask=None, padding_mask=None):

        Q = self.attention_layernorm(embed)
        K, V = embed, embed
        mha_outputs = self.attention_layer(
            Q, K, V, time_matrix_K, time_matrix_V, attn_mask=attn_mask, padding_mask=padding_mask
        )[0]
        embed = Q + mha_outputs
        embed = self.forward_layernorm(embed)
        pff_output = self.forward_layer(embed)
        embed = pff_output + embed

        return embed


class MultiHeadedAttention(nn.Module):
    """
    attention is all your need 多头实现

    Args:
        __init__():
            d_model (int): App item 编码的隐向量维度
            n_head (int): head number
            dropout_p (int): dropout失活比例。

        forward():
            Q (Tensor [batch size, Q length, d_model]): Target sequence Embed
            K (Tensor [batch size, K length, d_model]): Src sequence Embed
            V (Tensor [batch size, V length, d_model]): Src sequence Embed
            attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None.
            padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None.

    Returns:
        forward():
            [Tensor]: [batch size, Q len, d_model]

    """

    def __init__(self, d_model, n_heads, dropout_p):
        super().__init__()

        assert d_model % n_heads == 0, "[!] item 隐向量的维数必须能被head num 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProduct()
        self.dropout = nn.Dropout(dropout_p)

        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, attn_mask=None, padding_mask=None):
        """
        Args:
            Q (Tensor): Target sequence Embed [batch size, Q length, d_model]
            K (Tensor): Src sequence Embed [batch size, K length, d_model]
            V (Tensor): Src sequence Embed [batch size, V length, d_model]
            attn_mask (Tensor, optional): 注意力掩码 [Q length, K length]. Defaults to None.
            padding_mask (Tensor, optional): padding 掩码 [batch size, K length] Defaults to None.

        Returns:
            [Tensor]: [batch size, Q len, d_model]
        """
        batch_size = Q.shape[0]

        # Q,K,V计算与变形：
        # query,key,value = [batch size, (query,key,value)'s len, head dim]
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(V)

        # Q,K,V = [batch size, n head, (query,key,value)'s len, head dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        O, attn = self.attention(Q, K, V, attn_mask, padding_mask, self.dropout)
        O = O.permute(0, 2, 1, 3).contiguous()

        # O = [batch size, Q len, hid dim]
        O = O.view(batch_size, -1, self.d_model)
        O = self.fc_o(O)
        # O = self.dropout2(O) # 效果貌似差一丢丢，pytorch官方实现带了 dropout
        # O = [batch size, Q len, d_model]
        return O, attn


class TimeAwareMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_p):
        """
            TiSAS 时间感知的多头注意力机制实现

            Args:
                __init__():
                    d_model (int): App item 编码的隐向量维度
                    n_head (int): head number
                    dropout_p (int): dropout失活比例。

                forward():
                    Q (Tensor [batch size, Q length, d_model]): Target sequence Embed
                    K (Tensor [batch size, K length, d_model]): Src sequence Embed
                    V (Tensor [batch size, V length, d_model]): Src sequence Embed
                    time_matrix_K (Tensor [batch size, n head, query's len, key's len, head_dim]): Q 对 K 的时间间隔矩阵 Embed
                    time_matrix_V (Tensor [batch size, n head, query's len, value's len, head_dim]): Q 对 V 的时间间隔矩阵 Embed
                    attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None.
                    padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None.

        Returns:
            forward():
                [Tensor]: [batch size, Q len, d_model]


        """
        super().__init__()

        assert d_model % n_heads == 0, "[!] item 隐向量的维数必须能被head num 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.attention = TimeAwareScaledDotProduct()
        self.dropout = nn.Dropout(dropout_p)

        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, time_matrix_K, time_matrix_V, padding_mask=None, attn_mask=None):
        batch_size = Q.shape[0]
        q_len, k_len, v_len = Q.shape[1], K.shape[1], V.shape[1]

        # Q,K,V计算与变形：
        # query,key,value = [batch size, (query,key,value)'s len, head dim]
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(V)

        # Q,K,V = [batch size, n head, (query,key,value)'s len, head dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # time_matrix_K/V = [batch size, n head, query's len, (key,value)'s len, head_dim]
        time_matrix_K = time_matrix_K.view(batch_size, q_len, k_len, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        time_matrix_V = time_matrix_V.view(batch_size, q_len, v_len, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        O, attn = self.attention(Q, K, V, time_matrix_K, time_matrix_V, attn_mask, padding_mask, self.dropout)

        O = O.permute(0, 2, 1, 3).contiguous()

        # O = [batch size, Q len, hid dim]
        O = O.view(batch_size, -1, self.d_model)
        O = self.fc_o(O)

        # O = self.dropout2(O) # 效果貌似差一丢丢，pytorch官方实现带了 dropout

        # O = [batch size, Q len, d_model]
        return O, attn


class PCMultiHeadedAttention(nn.Module):
    """
    PC-transformer 多头实现

    Args:
        __init__():
            d_model (int): App item 编码的隐向量维度
            n_heads (int): head number
            sub_k (int): k 个 Local Aware
            epsilon (float):  LocalAwareSequeeze的正则化系数
            max_iter (int): LocalAwareSequeeze的sinkhorn迭代算法的迭代次数
            bidirectional (bool): LocalAware 是否进行序列的双向感知
            dropout_p (int): dropout失活比例。

        forward():
            Q (Tensor [batch size, Q length, d_model]): Target sequence Embed
            K (Tensor [batch size, K length, d_model]): Src sequence Embed
            V (Tensor [batch size, V length, d_model]): Src sequence Embed
            attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None.
            padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None.

    Returns:
        forward():
            [Tensor]: [batch size, Q len, d_model]

    """

    def __init__(self, d_model, n_heads, sub_k, epsilon, max_iter=100, bidirectional=True, dropout_p=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "[!] d_model must be divisible by n_heads!"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.las_layer = LocalAwareSequeeze(d_model, sub_k, epsilon, max_iter, bidirectional)
        self.attention_layer = ScaledDotProduct()
        self.dropout = nn.Dropout(dropout_p)

        self.fc_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size = Q.shape[0]

        Q, K, V = self.fc_q(Q), self.fc_k(self.las_layer(K)[0]), self.fc_v(self.las_layer(V)[0])
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        O, attn = self.attention_layer(Q, K, V, dropout=self.dropout)
        O = O.permute(0, 2, 1, 3).contiguous()
        O = O.view(batch_size, -1, self.d_model)
        O = self.fc_o(O)
        return O, attn


class ScaledDotProduct(nn.Module):
    """
    计算 'Scaled Dot Product Attention'

    Args:
        Q (Tensor [batch size, n head, Q length, head dim]): Target sequence Embed
        K (Tensor [batch size, n head, K length, head dim]): Src sequence Embed
        V (Tensor [batch size, n head, V length, head dim]): Src sequence Embed
            d_model = head dim * n head
        attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None. 1--mask，0--有效item
        padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None. 1--mask，0--有效item
        dropout (Dropout, optional): 传入 dropout 的实例化对象

    Returns:
        (Tensor): [batch size, n heads, Q len, head dim]
        (Tensor): [description]
    """

    def forward(self, Q, K, V, attn_mask=None, padding_mask=None, dropout=None):

        scale = math.sqrt(Q.size(-1))
        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        # energy = [batch size, n head, Q length, K length]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale

        if attn_mask is not None:
            # [Q len, K len] -> [1, 1, Q len, K len]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            energy += attn_mask

        if padding_mask is not None:
            # [batch size, K len] -> [batch size, 1, 1, K len]
            # padding矩阵中元素为 True 会被 (-2**32+1)极小值 替代
            """
            这个位置，如果是置换为float('-inf')，会使得softmax的时候出现 nan
            出现原因是，如果padding mask导致一整行都是-inf的时候，无法求softmax
            （-inf,-inf,-inf,-inf）-->softmax-->(nan,nan,nan,nan)
            而 [(-2**32+1)，(-2**32+1)，(-2**32+1)，(-2**32+1)]-->softmax-->[0.25, 0.25, 0.25, 0.25]
            所以置换为一个极小值就好了。比如 (-2**32+1)
            我感觉pytorch的源代码这个位置是有问题的：
            当年第一次写出现nan的时候，百思不得其解，终于找到原因了
            """
            energy = energy.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), (-(2 ** 32) + 1))

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        # attention = [batch size, n heads, Q length, K length]
        attention_weight = torch.softmax(energy, dim=-1)

        if dropout is not None:
            attention_weight = dropout(attention_weight)

        # 第三步，attention结果与V相乘
        # [batch size, n heads, Q length, head dim]
        O = torch.matmul(attention_weight, V)

        return O, attention_weight


class TimeAwareScaledDotProduct(nn.Module):
    """
    计算 'Relative time matrix Scaled Dot Product Attention'

    Args:
        Q (Tensor [batch size, n head, Q length, head dim]): Target sequence Embed
        K (Tensor [batch size, n head, K length, head dim]): Src sequence Embed
        V (Tensor [batch size, n head, V length, head dim]): Src sequence Embed
            d_model = head dim * n head
        time_matrix_K (Tensor [batch size, n head, query's len, key's len, head_dim]): Q 对 K 的时间间隔矩阵 Embed
        time_matrix_V (Tensor [batch size, n head, query's len, value's len, head_dim]): Q 对 V 的时间间隔矩阵 Embed
        attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None. 1--mask，0--有效item
        padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None. 1--mask，0--有效item
        dropout (Dropout, optional): 传入 dropout 的实例化对象

    Returns:
        (Tensor): [batch size, n heads, Q len, head dim]
        (Tensor): [description]
    """

    def forward(self, Q, K, V, time_matrix_K, time_matrix_V, attn_mask=None, padding_mask=None, dropout=None):
        scale = math.sqrt(Q.size(-1))

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        # energy = [batch size, n head, Q length, K length]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        energy += torch.matmul(time_matrix_K, Q.unsqueeze(-1)).squeeze(-1)

        energy = energy / scale

        if attn_mask is not None:
            # [Q len, K len] -> [1, 1, Q len, K len]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            energy += attn_mask

        if padding_mask is not None:
            # [batch size, K len] -> [batch size, 1, 1, K len]
            # padding矩阵中元素为 True 会被 (-2**32+1)极小值 替代
            energy = energy.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), (-(2 ** 32) + 1))

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        # attention = [batch size, n heads, Q length, K length]
        attention_weight = torch.softmax(energy, dim=-1)

        if dropout is not None:
            attention_weight = dropout(attention_weight)

        # 第三步，attention结果与V相乘
        # [batch size, n heads, Q length, head dim]
        O = torch.matmul(attention_weight, V)
        O += torch.matmul(attention_weight.unsqueeze(3), time_matrix_V).reshape(O.shape).squeeze(3)

        return O, attention_weight


class LocalAwareSequeeze(nn.Module):
    def __init__(self, d_model, sub_k, epsilon, max_iter, bidirectional):
        super().__init__()
        self.sub_k = sub_k
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.local_aware = nn.GRU(d_model, d_model, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(d_model * 2, 1) if bidirectional else nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.sequeeze = Squeeze(sub_k, epsilon, max_iter)

    def forward(self, X):
        B, L, _ = X.size()
        A = self.relu(self.fc(self.local_aware(X)[0]).view(B, L))
        gamma = self.sequeeze(A)
        X_topk = torch.matmul(X.permute(0, 2, 1), gamma).permute(0, 2, 1).contiguous()
        return X_topk, A, gamma


class Squeeze(nn.Module):
    def __init__(self, sub_k, epsilon, max_iter=200):
        super().__init__()
        self.sub_k = sub_k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([i for i in range(sub_k + 1)]).view([1, 1, sub_k + 1])
        self.max_iter = max_iter

    def forward(self, A):
        self.anchors = self.anchors.to(A.device)
        B, L = A.size()
        A = A.view([B, L, 1])

        A_ = A.clone().detach()
        max_score = torch.max(A_).detach()
        A_[A_ == float("-inf")] = float("inf")
        min_score = torch.min(A_).detach()
        filled_value = min_score - (max_score - min_score)
        mask = A == float("-inf")
        A = A.masked_fill(mask, filled_value)

        C = (A - self.anchors) ** 2
        C = C / (C.max().detach())

        P_A = torch.ones([1, L, 1], requires_grad=False) / L
        P_K = [(L - self.sub_k) / L] + [1.0 / L for _ in range(self.sub_k)]
        P_K = torch.FloatTensor(P_K).view([1, 1, self.sub_k + 1])

        P_A = P_A.to(A.device)
        P_K = P_K.to(A.device)

        Gamma = _DifferentialFunction.apply(C, P_A, P_K, self.epsilon, self.max_iter)
        gamma = Gamma[:, :, 1:] * L
        return gamma


class _DifferentialFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, P_A, P_K, epsilon, max_iter):
        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = _sinkhorn_forward(C, P_A, P_K, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    print("Nan appeared in Gamma,re-computing...")
                    Gamma = _sinkhorn_forward_stablized(C, P_A, P_K, epsilon, max_iter)
            else:
                Gamma = _sinkhorn_forward_stablized(C, P_A, P_K, epsilon, max_iter)

            ctx.save_for_backward(P_A, P_K, Gamma)
            ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):
        epsilon = ctx.epsilon
        P_A, P_K, Gamma = ctx.saved_tensors
        with torch.no_grad():
            grad_C = _sinkhorn_backward(grad_output_Gamma, Gamma, P_A, P_K, epsilon)
        return grad_C, None, None, None, None


def _sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    v = (torch.ones([bs, 1, k_]) / (k_)).to(C.device)
    G = torch.exp(-C / epsilon)
    for _ in range(max_iter):
        u = mu / (G * v).sum(-1, keepdim=True)
        v = nu / (G * u).sum(-2, keepdim=True)
    Gamma = u * G * v
    return Gamma


def _sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    k = k_ - 1

    f = torch.zeros([bs, n, 1]).to(C.device)
    g = torch.zeros([bs, 1, k + 1]).to(C.device)

    epsilon_log_mu = epsilon * torch.log(mu)
    epsilon_log_nu = epsilon * torch.log(nu)

    def min_epsilon_row(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

    def min_epsilon_col(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

    for _ in range(max_iter):
        f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

    Gamma = torch.exp((-C + f + g) / epsilon)
    return Gamma


def _sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    nu_ = nu[:, :, :-1]
    Gamma_ = Gamma[:, :, :-1]
    bs, n, k_ = Gamma.size()
    inv_mu = 1.0 / (mu.view([1, -1]))
    Kappa = torch.diag_embed(nu_.squeeze(-2)) - torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)

    inv_Kappa = torch.inverse(Kappa)
    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = Gamma_mu.matmul(inv_Kappa)
    G1 = grad_output_Gamma * Gamma
    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma
    g1_L = g1.unsqueeze(-2).matmul(L)
    G22 = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma
    G23 = -F.pad(g1_L, pad=(0, 1), mode="constant", value=0) * Gamma
    G2 = G21 + G22 + G23

    del g1, G21, G22, G23, Gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1)
    g2 = g2[:, :-1, :]
    G31 = -L.matmul(g2) * Gamma
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode="constant", value=0) * Gamma
    G3 = G31 + G32
    grad_C = (-G1 + G2 + G3) / epsilon
    return grad_C


class PointWiseFeedForwardByLinear(nn.Module):
    """
    Attention is all you need Transformer FeedForward 实现
    基于 nn.Linear() 实现 Attention机制中的 Feed Forward 函数

    Args:
        d_model (int): 编码的隐向量维度
        d_ff (int, optional): 隐层的维数，通常 4 * hidden_size . Defaults to None.
        dropout_p (float, optional): dropout失活比例. Defaults to 0.1.
    """

    def __init__(self, d_model, d_ff=None, dropout_p=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.activation = GELU()
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.dropout2(self.w_2(self.dropout1(self.activation(self.w_1(x)))))


class PointWiseFeedForwardByConv(nn.Module):
    """
    基于 nn.Conv1d() 实现 Attention机制中的 Feed Forward 函数

    Args:
        d_model (int): 编码的隐向量维度
        dropout_p (float, optional): dropout失活比例. Defaults to 0.1.
    """

    def __init__(self, d_model, dropout_p):
        super().__init__()

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.activation = GELU()  # bert用gelu，原生transformer Relu
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        outputs = self.dropout2(self.conv2(self.activation(self.dropout1(self.conv1(x.transpose(-1, -2))))))
        # Conv1D 输入维度要求 (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        return outputs


class SublayerConnection(nn.Module):
    """
    层归一化之后进行残差连接
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout_p):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def _generate_square_subsequent_mask(sz):
    """
    vanilla attetnion mask
    注意力掩码生成函数，生成 shape = [sz, sz] 的矩阵

    Args:
        sz (int): 矩阵尺寸

    Returns:
        [Tensor]: [sz, sz]
    """
    # float('-inf') -2**32+1
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, (-(2 ** 32) + 1)).masked_fill(mask == 1, float(0.0))
    return mask
