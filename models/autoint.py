import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import DNN, CrossNet, PredictionLayer, Transformer
from inputs import (
    LinearLogits,
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    concat_fun,
    CreateFeatureColumns,
)


class AutoInt(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        self.att_layer_num = args.att_layer_num
        self.att_head_num = args.att_head_num
        self.d_model = args.d_model
        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.task = args.task

        if len(self.dnn_hidden_units) <= 0 and self.att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        # 参与 DNN 的 feature
        self.input_feature_columns = feature_fun.get_feature_columns()
        # 参与 attn 的 feature
        self.embed_feature_columns = (
            feature_fun.get_emb_dense_feature_columns()
            + feature_fun.get_sparse_feature_columns()
            + feature_fun.get_val_sparse_feature_columns()
        )
        self.feature_name = feature_fun.get_feature_name()
        self.feature_index = feature_fun.get_feature_index()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

        # DNN 全量 feature 进去
        self.use_dnn = len(self.input_feature_columns) > 0 and len(self.dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(
                compute_input_dim(self.input_feature_columns), self.dnn_hidden_units, activation=self.activation
            )
        # Attention
        self.att_layers = Transformer(self.att_layer_num, self.att_head_num, self.d_model, args.dropout_p)

        # Linear
        attn_feature_dim = compute_input_dim(self.embed_feature_columns)
        if len(self.dnn_hidden_units) and self.att_layer_num > 0:
            dnn_linear_in_feature = self.dnn_hidden_units[-1] + attn_feature_dim
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = self.dnn_hidden_units[-1]
        elif self.att_layer_num > 0:
            dnn_linear_in_feature = attn_feature_dim
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)

        # linear Logit
        self.logit = LinearLogits(self.input_feature_columns, self.feature_index)
        self.out = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "autoint"

    def forward(self, batch):
        x = batch["x"].float()
        logit = self.logit(x)

        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )

        att_output = self.att_layers(concat_fun(feature_embedding_list, axis=1))
        att_output = torch.flatten(att_output, start_dim=1)
        dnn_input = combined_dnn_input(feature_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Attention
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Attention
            logit += self.dnn_linear(att_output)
        else:  # Error
            pass
        y_pred = self.out(logit)
        return {"y_pred": y_pred}
