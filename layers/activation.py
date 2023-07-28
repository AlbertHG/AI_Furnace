# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : 

import math
import torch
import torch.nn as nn


def activation_layer(act_name):
    """
    Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == "sigmoid":
            act_layer = nn.Sigmoid()
        elif act_name.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == "prelu":
            act_layer = nn.PReLU()
        elif act_name.lower() == "tanh":
            act_layer = nn.Tanh()
        elif act_name.lower() == "gelu":
            act_layer = nn.GELU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
