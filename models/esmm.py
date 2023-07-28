# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      :

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import DNN, PredictionLayer
from inputs import (
    LinearLogits,
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    CreateFeatureColumns,
)


class ESMMModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        assert len(args.task) == len(args.metrics_list) == 2, "[!] The task num must equal metrics_list element num"

        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.metrics_list = args.metrics_list
        self.task = args.task

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.input_feature_columns = feature_fun.get_feature_columns()
        self.feature_index = feature_fun.get_feature_index()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

        # Deep
        self.ctr_dnn = DNN(
            compute_input_dim(self.input_feature_columns), self.dnn_hidden_units, activation=self.activation
        )
        self.cvr_dnn = DNN(
            compute_input_dim(self.input_feature_columns), self.dnn_hidden_units, activation=self.activation
        )

        # Linear
        self.ctr_dnn_linear = nn.Linear(self.dnn_hidden_units[-1], 1, bias=False)
        self.cvr_dnn_linear = nn.Linear(self.dnn_hidden_units[-1], 1, bias=False)
        self.ctr_out = PredictionLayer(self.task[0])
        self.cvr_out = PredictionLayer(self.task[1])

    @classmethod
    def code(cls):
        return "esmm"

    def forward(self, batch):
        x = batch["x"].float()
        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )
        dnn_input = combined_dnn_input(feature_embedding_list, dense_value_list)

        ctr_out = self.ctr_dnn(dnn_input)
        cvr_out = self.cvr_dnn(dnn_input)
        ctr_logit = self.ctr_dnn_linear(ctr_out)
        cvr_logit = self.cvr_dnn_linear(cvr_out)
        ctr_pred = self.ctr_out(ctr_logit)
        cvr_pred = self.cvr_out(cvr_logit)
        ctcvr_out = torch.mul(ctr_pred, cvr_pred)

        return {self.metrics_list[0]: ctr_pred, self.metrics_list[1]: ctcvr_out}
