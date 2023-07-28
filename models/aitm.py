import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models import AbstractModel
from layers import DNN, PredictionLayer, AIT
from inputs import (
    LinearLogits,
    compute_input_dim,
    create_embedding_matrix,
    combined_dnn_input,
    input_from_feature_columns,
    CreateFeatureColumns,
)


class AITMModel(AbstractModel):
    """AITM Network architecture.

    Args:
        AbstractModel (class): 父类继承.
        yaml 配置文件必备参数:
            - dnn_hidden_units (list): 每一个任务塔 DNN 各个隐藏层的层数.
            - activation (str): 用于 DNN 各个隐藏层的激活函数.
            - metrics_list (list): 每一个任务塔的任务名称
            - task (list): 每一个任务塔的具体任务. 可选择的任务类型：``"binary", "regression"``
            - feature_msg (str): dataFrame 数据集的说明文件 json 路径
    References
        - [Xi D, Chen Z, Yan P, et al. Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising[J]. arXiv preprint arXiv:2105.08489, 2021.]
    """
    def __init__(self, args):
        super().__init__(args)
        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.task = args.task
        self.metrics_list = args.metrics_list

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.input_feature_columns = feature_fun.get_feature_columns()
        self.feature_index = feature_fun.get_feature_index()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

        # task layers
        self.tower_layers = nn.ModuleList(
            [
                DNN(
                    compute_input_dim(self.input_feature_columns),
                    self.dnn_hidden_units,
                    activation=self.activation,
                    dropout_p=args.dropout_p,
                )
                for _ in self.task
            ]
        )

        # info transform layers
        self.info_transform_layers = nn.ModuleList(
            [nn.Linear(self.dnn_hidden_units[-1], self.dnn_hidden_units[-1]) for _ in range(len(self.task) - 1)]
        )

        # ait layers
        self.ait_layers = nn.ModuleList([AIT(dim=self.dnn_hidden_units[-1]) for _ in range(len(self.task) - 1)])

        # linear layers
        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dnn_hidden_units[-1], 1, bias=False),
                    PredictionLayer(task=t),
                )
                for t in self.task
            ]
        )

    @classmethod
    def code(cls):
        return "aitm"

    def forward(self, batch):
        x = batch["x"].float()
        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )
        tower_input = combined_dnn_input(feature_embedding_list, dense_value_list)
        Q = [t(tower_input) for t in self.tower_layers]

        pred_output = OrderedDict()
        p = Q[0]
        pred_output[self.metrics_list[0]] = self.linear_layers[0](p)
        for t in range(len(self.task) - 1):
            p = torch.unsqueeze(p, 1)
            q = torch.unsqueeze(Q[t + 1], 1)
            ait_input = torch.cat([p, q], 1)
            z = self.ait_layers[t](ait_input)
            pred_output[self.metrics_list[t + 1]] = self.linear_layers[t + 1](z)
            p = self.info_transform_layers[t](z)

        return pred_output
