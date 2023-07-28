# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/12/23 12:00
# @Function      :

import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ItemEmbedding(nn.Embedding):
    """
    item embedding 层，继承了nn.Embedding类

    Args:
        app_num ([int]): App的个数
        embed_size (int, optional): App Embed 的隐向量维度. Defaults to 64.
        padding_idx ([int], optional): padding_idx是索引值，其索引值对应的位置的embed会被填充为 0. Defaults to None.
        embed_path ([str], optional): embedding 权重文件的路径. Defaults to None.
        requires_train (bool, optional): 是否要求权重参与训练. Defaults to True.
    """

    def __init__(self, item_num, embed_size=64, padding_idx=None, embed_path=None, requires_train=True):
        super().__init__(item_num, embed_size, padding_idx)
        self.embed_path = embed_path
        self.requires_train = requires_train
        if self.embed_path:
            # with open(self.embed_path, 'r') as fn:
            appitem_vector = np.load(self.embed_path)
            self.weight.data.copy_(torch.from_numpy(appitem_vector))

            if not self.requires_train:
                # 如果要固定embedding层参数不参与训练，则切记不能在embedding层用优化器
                self.weight.requires_grad = False


class LearnedPositionalEmbedding(nn.Embedding):
    """
    带参数的 Position Embedding 层。
    继承一个nn.Embedding，再续上一个dropout。
    因为nn.Embedding中包含了一个可以按索引取向量的权重矩阵weight。
    Args:
        d_model (int): 位置编码的隐向量维度
        max_len (int, optional): 编码允许的最长位置. Defaults to 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__(max_len, d_model)

    def forward(self, x):
        # x.shape = [Batch size, Series Length]
        weight = self.weight.data.unsqueeze(0)  # [max_len,E]->[1,max_len,E]
        x = weight[:, : x.size(1), :].repeat(x.size(0), 1, 1)  # [N, S, E]
        return x


class AbsolutePositionalEmbedding(nn.Module):
    """
    Transformer 位置编码，位置编码与 嵌入 具有相同的d_model维度，因此可以将两者相加。
    该模块没有需要训练的参数

    Args:
        d_model (int): 位置编码的隐向量维度
        max_len (int, optional): 编码允许的最长位置. Defaults to 5000.
        position (Tensor, optional): 位置编码. Defaults to None.
    """

    def __init__(self, d_model, max_len=5000, position=None):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        if position is None:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe = pe.unsqueeze(0).transpose(0, 1) # pytorch自带的transformer需要用这招
        self.register_buffer("pe", pe)  # 将模块添加到持久缓冲区

    def forward(self, x):
        # x.shape = [Batch size, Series Length]
        x = self.pe[:, : x.size(1), :].repeat(x.size(0), 1, 1)
        # x.shape = [batch size, Series Length, d_model]
        return x


class DenseFeatureEncoding(nn.Module):
    """
    论文：time-dependent representation for neural event sequence prediction。
    对dense feature的编码: Event-time joint Embedding。

    Args:
        d_model (int): dense feature的隐向量维度
        padding_value ([int], optional): padding_value 是填充值，其填充值对应的embed会被填充为 0. Defaults to None.
        hidden_embedding_dim ([type]):
        dropout (float, optional): dropout失活比例. Defaults to 0.1.
    """

    def __init__(self, d_model, padding_value=None, hidden_embed_dim=None, dropout=0.1):

        super().__init__()
        self.d_model = d_model
        self.padding_value = padding_value
        self.hidden_embed_dim = d_model if hidden_embed_dim is None else hidden_embed_dim
        self.w = Parameter(torch.Tensor(self.hidden_embed_dim))
        self.b = Parameter(torch.Tensor(self.hidden_embed_dim))
        self.embedding_matrix = Parameter(torch.Tensor(self.hidden_embed_dim, self.d_model))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.embedding_matrix, gain=1.0)
        nn.init.uniform_(self.w)
        nn.init.uniform_(self.b)

    def forward(self, x):
        x1 = torch.unsqueeze(x, 2).float()  # [batch_size, series_size, 1]
        x1 = x1 * self.w + self.b  # [batch_size, series_size, hidden_embed_dim]
        x1 = F.softmax(x1, dim=-1)

        if self.padding_value is not None:
            mask = torch.unsqueeze(x.ne(self.padding_value), 2).repeat(1,1,x1.shape[-1])
            x1 = x1 * mask

        # [batch_size, series_size, d_model]
        output = torch.einsum("bsv,vi->bsi", x1, self.embedding_matrix)
        return output


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="default"):
        super().__init__()

        hour_size = 25
        mimute_size = 61
        second_size = 61
        weekday_size = 8
        day_size = 32

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)

    def forward(self, batch):

        hour_e = self.hour_embed(batch["hour"])
        week_e = self.weekday_embed(batch["week"])

        ret_time = {"week": week_e, "hour": hour_e}

        return ret_time
