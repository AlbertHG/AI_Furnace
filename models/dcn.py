# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/12/02 12:00
# @Function      :

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import DNN, CrossNet, PredictionLayer
from inputs import (
    LinearLogits,
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    CreateFeatureColumns,
)


class DCNModel(AbstractModel):
    """Deep&Cross Network and Deep&Cross V2 Network architecture.

    Args:
        AbstractModel (class): 父类继承.
        yaml 配置文件必备参数:
            - cross_num (int): Cross Net 的层数.
            - cross_parameterization (str): ``"vector"`` or ``"matrix"``, 切换 DCN V1 和 DCN V2.
            - dnn_hidden_units (list): DNN 各个隐藏层的层数.
            - activation (str): 用于 DNN 各个隐藏层的激活函数.
            - task (str): ``"binary"`` for  binary logloss
            - feature_msg (str): dataFrame 数据集的说明文件 json 路径
    References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[M]//Proceedings of the ADKDD'17. 2017: 1-7.]
        - [Wang R, Shivanna R, Cheng D, et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems[C]//Proceedings of the Web Conference 2021. 2021: 1785-1797.]
    """

    def __init__(self, args):
        super().__init__(args)
        self.cross_num = args.cross_num
        self.cross_parameterization = args.cross_parameterization
        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.task = args.task

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.input_feature_columns = feature_fun.get_feature_columns()
        self.feature_index = feature_fun.get_feature_index()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

        # Deep
        self.dnn = DNN(
            compute_input_dim(self.input_feature_columns),
            self.dnn_hidden_units,
            activation=self.activation,
            dropout_p=args.dropout_p,
        )
        # Cross
        self.crossnet = CrossNet(
            in_features=compute_input_dim(self.input_feature_columns),
            layer_num=self.cross_num,
            parameterization=self.cross_parameterization,
        )

        # Linear
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = compute_input_dim(self.input_feature_columns) + self.dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = self.dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = compute_input_dim(self.input_feature_columns)
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)

        # linear Logit
        self.logit = LinearLogits(self.input_feature_columns, self.feature_index)
        self.out = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "dcn"

    def forward(self, batch):
        x = batch["x"].float()
        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )
        logit = self.logit(x)
        dnn_input = combined_dnn_input(feature_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:  # Only Cross
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:  # Error
            pass
        y_pred = self.out(logit)
        return {"y_pred": y_pred}
