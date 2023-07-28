import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import CIN, DNN, PredictionLayer
from inputs import (
    LinearLogits,
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    CreateFeatureColumns,
)


class xDeepFMModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.cin_hidden_units = args.cin_hidden_units
        self.cin_split_half = args.cin_split_half
        self.cin_activation = args.cin_activation

        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.task = args.task

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.input_feature_columns = feature_fun.get_feature_columns()
        self.feature_index = feature_fun.get_feature_index()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

        # DNN
        self.use_dnn = len(self.input_feature_columns) > 0 and len(self.dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(
                compute_input_dim(self.input_feature_columns), self.dnn_hidden_units, activation=self.activation
            )
            # Linear
            self.dnn_linear = nn.Linear(self.dnn_hidden_units[-1], 1, bias=False)

        # CIN
        self.use_cin = len(self.cin_hidden_units) > 0 and len(self.input_feature_columns) > 0
        if self.use_cin:
            feature_embedding = len(self.embedding_dict)
            if self.cin_split_half == True:
                self.featuremap_num = sum(self.cin_hidden_units[:-1]) // 2 + self.cin_hidden_units[-1]
            else:
                self.featuremap_num = sum(self.cin_hidden_units)
            self.cin = CIN(feature_embedding, self.cin_hidden_units, self.cin_activation, self.cin_split_half)
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False)

        # linear Logit
        self.logit = LinearLogits(self.input_feature_columns, self.feature_index)
        self.out = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "xdeepfm"

    def forward(self, batch):
        x = batch["x"].float()
        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )

        logit = self.logit(x)  # LR
        if self.use_cin and len(feature_embedding_list) > 0:  # LR + CIN
            cin_input = torch.cat(feature_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
            logit += cin_logit

        if self.use_dnn:  # LR + DNN + CIN
            dnn_input = combined_dnn_input(feature_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)
        return {"y_pred": y_pred}
