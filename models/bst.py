import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import DNN, Transformer, PredictionLayer
from inputs import (
    LinearLogits,
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    concat_fun,
    CreateFeatureColumns,
    embedding_lookup,
    get_varlen_group_and_mask,
)


class BSTModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        # 声明 seq 特征
        self.sequence_features_list = args.sequence_features_columns

        self.att_layer_num = args.att_layer_num
        self.att_head_num = args.att_head_num
        self.d_model = args.d_model
        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.task = args.task
        if len(self.dnn_hidden_units) <= 0 and self.att_layer_num <= 0:
            raise ValueError("[!] Either hidden_layer or att_layer_num must > 0")

        # feature
        feature_func = CreateFeatureColumns(args.feature_msg)
        self.feature_name = feature_func.get_feature_name()
        self.feature_index = feature_func.get_feature_index()
        # val sequence
        if self.sequence_features_list:
            self.sequence_features_columns = feature_func.find_feature_columns(self.sequence_features_list)
        else:
            self.sequence_features_columns = feature_func.get_val_sparse_feature_columns()
        self.other_feature_columns = feature_func.find_feature_columns(self.sequence_features_list, negation=True)

        # embedding
        self.embedding_dict = create_embedding_matrix(self.sequence_features_columns + self.other_feature_columns)

        if not (len(self.sequence_features_columns) > 0 and self.att_layer_num > 0):
            raise ValueError("[!] BST model 中 序列数据是必须的！")
        # Attention 有多少组seq数据,就得多少个 transformer
        group_names = set(feats.group_name for feats in self.sequence_features_columns)
        self.att_layers_group = nn.ModuleList(
            [Transformer(self.att_layer_num, self.att_head_num, self.d_model, args.dropout_p) for _ in group_names]
        )
        # DNN
        # 计算总共有多少组 attention 输出 + other 特征
        dnn_input_dim = compute_input_dim(
            [next(filter(lambda x: x.group_name == g, self.sequence_features_columns)) for g in group_names]
            + self.other_feature_columns,
            varlen_feature_pooling=False,
        )
        self.dnn = DNN(dnn_input_dim, self.dnn_hidden_units, activation=self.activation)
        # Linear
        self.dnn_linear = nn.Linear(self.dnn_hidden_units[-1], 1)
        self.out = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "bst"

    def forward(self, batch):
        x = batch["x"].float()
        sparse_feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.other_feature_columns, self.feature_index, self.embedding_dict
        )
        sequence_embed_dict = embedding_lookup(
            x, self.embedding_dict, self.feature_index, self.sequence_features_columns
        )
        val_sparse_feature_embedding_dict, padding_mask_dict = get_varlen_group_and_mask(
            sequence_embed_dict, x, self.feature_index, self.sequence_features_columns
        )

        att_output_list = [
            att_net(emb, mask)
            for (att_net, emb, mask) in zip(
                self.att_layers_group, val_sparse_feature_embedding_dict.values(), padding_mask_dict.values()
            )
        ]
        att_output = concat_fun(att_output_list, 1)
        att_output = torch.flatten(att_output, start_dim=1)

        other_input = combined_dnn_input(sparse_feature_embedding_list, dense_value_list)
        logit = self.dnn_linear(self.dnn(concat_fun([other_input, att_output])))

        y_pred = self.out(logit)
        return {"y_pred": y_pred}
