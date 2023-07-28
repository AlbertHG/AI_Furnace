import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import FM, DNN, PredictionLayer
from inputs import (
    LinearLogits,
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    CreateFeatureColumns,
)


class DeepFMModel(AbstractModel):
    """ DeepFM Network architecture.

    Args:
        AbstractModel (class): 父类继承.
        yaml 配置文件必备参数:
            - use_fm (bool): 是否启用 FM Net.
            - dnn_hidden_units (list): DNN 各个隐藏层的层数.
            - activation (str): 用于 DNN 各个隐藏层的激活函数.
            - task (str): ``"binary"`` for  binary logloss
            - feature_msg (str): dataFrame 数据集的说明文件 json 路径
    References
        - [Guo H, Tang R, Ye Y, et al. DeepFM: a factorization-machine based neural network for CTR prediction[J]. arXiv preprint arXiv:1703.04247, 2017.]
    """

    def __init__(self, args):
        super().__init__(args)

        self.use_fm = args.use_fm
        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.task = args.task

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.feature_index = feature_fun.get_feature_index()
        self.input_feature_columns = feature_fun.get_feature_columns()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns, include_dense=True)

        # FM
        if args.use_fm:
            self.fm = FM()
        # DNN
        self.use_dnn = len(self.input_feature_columns) > 0 and len(self.dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(
                compute_input_dim(self.input_feature_columns), self.dnn_hidden_units, activation=self.activation
            )

        # Linear
        self.dnn_linear = nn.Linear(self.dnn_hidden_units[-1], 1, bias=False)

        # linear Logit
        self.logit = LinearLogits(self.input_feature_columns, self.feature_index)
        self.out = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "deepfm"

    def forward(self, batch):
        x = batch["x"].float()
        logit = self.logit(x)
        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )

        if self.use_fm and len(feature_embedding_list) > 0:
            fm_input = torch.cat(feature_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(feature_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)
        return {"y_pred": y_pred}
