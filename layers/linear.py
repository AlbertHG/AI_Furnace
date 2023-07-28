# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : 

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import activation_layer


class DNN(nn.Module):
    """
    多层感知机

    x-->Linear-->bn-->activation-->Linear-->bn-->activation-->...-->Linear-->softmax/sigmoid-->output

    Args:
        args参数，内含下列四项参数
            input_dim (int): 全连接输入层维度
            hidden_dim_list (list): 全连接隐藏层维度
            use_bn (bool): 是否在网络中启用 BN 正则化
            activation (str, optional): 隐藏层层激活函数. Defaults to 'relu'.
            dropout_p (float, optional): 隐藏层 dropout 失活比例. Defaults to 0.
    """

    def __init__(self, input_dim, hidden_dim_list, activation="relu", use_bn=True, dropout_p=0.0):
        super().__init__()
        if use_bn and dropout_p > 0.0:
            warnings.warn("[!] 在 DNN 网络中同时开启了 BN 和 Dropout. ")

        self.input_dim = input_dim
        self.hidden_dim_list = hidden_dim_list[:-1]
        self.output_dim = hidden_dim_list[-1]
        self.dropout_p = dropout_p
        self.activation = activation
        self.use_bn = use_bn

        # 创建全连接网络
        self.mlp = self._make_layers(self.input_dim, self.hidden_dim_list, self.output_dim)

    def forward(self, x):
        x = self.mlp(x)  # shape=[batch_size, n_token]
        return x

    def _make_layers(self, input_dim, hidden_layer_list, output_dim):
        layers = []
        input_dim = input_dim
        if len(hidden_layer_list) > 0:
            for i in hidden_layer_list:
                i = int(i)
                layers += [nn.Linear(input_dim, i)]
                if self.use_bn:
                    layers += [nn.BatchNorm1d(i)]
                layers += [activation_layer(self.activation)]
                if self.dropout_p > 0:
                    layers += [nn.Dropout(self.dropout_p)]
                input_dim = i
        layers += [nn.Linear(input_dim, output_dim)]
        return nn.Sequential(*layers)


class PredictionLayer(nn.Module):
    """
    任务模型的输出激活层，softmax or sigmoid

      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task="binary", use_bias=False, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary, multiclass or regression")

        super().__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        output = x
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        if self.task == "multiclass":
            output = F.softmax(output, dim=-1)
        return output
