import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import FM, AFMLayer, PredictionLayer
from inputs import LinearLogits, create_embedding_matrix, input_from_feature_columns, CreateFeatureColumns


class AFMModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.use_attention = True
        self.attention_factor = args.attention_factor
        self.dropout_p = args.dropout_p
        self.task = args.task

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.input_feature_columns = feature_fun.get_sparse_feature_columns()
        self.feature_name = feature_fun.get_feature_name()
        self.feature_index = feature_fun.get_feature_index()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

        # FM or AFM
        if self.use_attention:
            self.fm = AFMLayer(args.d_model, self.attention_factor, self.dropout_p)
        else:
            self.fm = FM()

        # linear Logit
        self.logit = LinearLogits(self.input_feature_columns, self.feature_index)
        self.out = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "afm"

    def forward(self, batch):
        x = batch["x"].float()
        feature_embedding_list, _ = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict, support_dense=False
        )

        logit = self.logit(x)
        if len(feature_embedding_list) > 0:
            if self.use_attention:
                logit += self.fm(feature_embedding_list)
            else:
                logit += self.fm(torch.cat(feature_embedding_list, dim=1))
        y_pred = self.out(logit)
        return {"y_pred": y_pred}
