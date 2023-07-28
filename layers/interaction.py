# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      :

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import itertools
from .activation import activation_layer


class CrossNet(nn.Module):
    """
    https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/interaction.py
    The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.

        Input shape
            - 2D tensor with shape: ``(batch_size, units)``.
        Output shape
            - 2D tensor with shape: ``(batch_size, units)``.
        Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **layer_num**: Positive integer, the cross layer number
            - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        References
            - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
            - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization="vector"):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == "vector":
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == "matrix":
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == "vector":
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == "matrix":
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

    Input shape
        - 3D tensor with shape: ``(batch_size,field_size, embedding_size)``.
    Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
    References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class AFMLayer(nn.Module):
    """Attentonal Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

    Input shape
        - A list of 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
    Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
    Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **attention_factor** : Positive integer, dimensionality of the attention network output space.
        - **dropout_p** : float between in [0,1). Fraction of the attention net output units to dropout.
    References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(self, in_features, attention_factor=4, dropout_p=0.0):
        super().__init__()
        self.attention_factor = attention_factor
        embedding_size = in_features

        self.attention_W = nn.Parameter(torch.Tensor(embedding_size, self.attention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))
        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(
                tensor,
            )

        for tensor in [self.attention_b]:
            nn.init.zeros_(
                tensor,
            )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q

        bi_interaction = inner_product
        attention_temp = F.relu(torch.tensordot(bi_interaction, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(
            torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1
        )
        attention_output = torch.sum(self.normalized_att_score * bi_interaction, dim=1)
        attention_output = self.dropout(attention_output)  # training
        afm_out = torch.tensordot(attention_output, self.projection_p, dims=([-1], [0]))
        return afm_out


class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.

    Input shape
      - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.
    Output shape
      - 2D tensor with shape: ``(batch_size, featuremap_num)``
                              ``featuremap_num = sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
    Arguments
      - **filed_size** : Positive integer, number of feature groups.
      - **layer_size** : list of int.Feature maps in each layer.
      - **activation** : activation function name used on feature maps.
      - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
    References
      - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(
        self,
        field_size,
        layer_size=[128, 128],
        activation="relu",
        split_half=True,
    ):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError("[!] layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)

        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "[!] layer_size must be even number except for the last layer when split_half=True"
                    )
                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("[!] Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            x = torch.einsum("bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0])
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation == "linear":
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)

        return result


class AIT(nn.Module):
    """Attention Network used in AITM.

    Input shape
      - 3D tensor with shape: ``(batch_size, 2, dim)``.
    Output shape
      - 2D tensor with shape: ``(batch_size, dim)``

    Arguments
      - **dim** : the output dim of tower network
    References
      - [Xi D, Chen Z, Yan P, et al. Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising[J]. arXiv preprint arXiv:2105.08489, 2021.]
    """

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        Q, K, V = self.fc_q(inputs), self.fc_k(inputs), self.fc_v(inputs)
        O = torch.sum(torch.mul(Q, K), -1) / math.sqrt(inputs.shape[-1])
        O = self.softmax(O)
        outputs = torch.sum(torch.mul(torch.unsqueeze(O, -1), V), dim=1)
        return outputs
