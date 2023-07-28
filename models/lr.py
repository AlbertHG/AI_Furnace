import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import DNN, CrossNet, PredictionLayer
from inputs import (
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    CreateFeatureColumns,
)


class LRCTRModel(AbstractModel):
    """LR Network architecture.

    Args:
        AbstractModel (class): 父类继承.
        yaml 配置文件必备参数:
            - task (str): ``"binary"`` for  binary logloss
            - feature_msg (str): dataFrame 数据集的说明文件 json 路径
    """

    def __init__(self, args):
        super().__init__(args)
        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.input_feature_columns = feature_fun.get_feature_columns()
        self.feature_index = feature_fun.get_feature_index()
        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)
        linear_in_feature = compute_input_dim(self.input_feature_columns)
        # linear Logit
        self.logit = nn.Linear(linear_in_feature, 1)
        self.out = PredictionLayer(args.task)

    @classmethod
    def code(cls):
        return "lr_ctr"

    def forward(self, batch):
        x = batch["x"].float()
        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )
        x = combined_dnn_input(feature_embedding_list, dense_value_list)
        y_pred = self.out(self.logit(x))
        return {"y_pred": y_pred}
