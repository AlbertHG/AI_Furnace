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


class MMOEModel(AbstractModel):
    """MMOE Network architecture.

    Args:
        AbstractModel (class): 父类继承.
        yaml 配置文件必备参数:
            - num_experts (int): 专家网络的个数.
            - expert_dnn_hidden_units (list): 每一个专家 DNN 各个隐藏层的层数.
            - gate_dnn_hidden_units (list): 每一个门控 DNN 各个隐藏层的层数.
            - tower_dnn_hidden_units (list): 每一个任务塔 DNN 各个隐藏层的层数.
            - activation (str): 用于 DNN 各个隐藏层的激活函数.
            - metrics_list (list): 每一个任务塔的任务名称
            - task (list): 每一个任务塔的具体任务. 可选择的任务类型：``"binary", "regression"``
            - feature_msg (str): dataFrame 数据集的说明文件 json 路径
    References
        - [Ma J, Zhao Z, Yi X, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 1930-1939.]
    """

    def __init__(self, args):
        super().__init__(args)

        assert args.num_experts >= 1, "[!] Experters num must be greater than 1"
        assert isinstance(args.task, (list, tuple)) and len(args.task) >= 1, "[!] Task num must be greater than 1"
        assert (
            args.num_experts == args.gate_dnn_hidden_units[-1]
        ), "[!] The output dimension of the gate layers must equal experts num"
        assert len(args.task) == len(args.metrics_list), "[!] The task num must equal metrics_list element num"

        self.num_experts = args.num_experts
        self.expert_dnn_hidden_units = args.expert_dnn_hidden_units
        self.gate_dnn_hidden_units = args.gate_dnn_hidden_units
        self.tower_dnn_hidden_units = args.tower_dnn_hidden_units
        self.activation = args.activation
        self.metrics_list = args.metrics_list
        self.task = args.task

        # feature
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.input_feature_columns = feature_fun.get_feature_columns()
        self.feature_index = feature_fun.get_feature_index()

        # embedding
        self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

        # expert layers
        self.expert_layers = nn.ModuleList(
            [
                DNN(
                    compute_input_dim(self.input_feature_columns),
                    self.expert_dnn_hidden_units,
                    activation=self.activation,
                    dropout_p=args.dropout_p,
                )
                for _ in range(self.num_experts)
            ]
        )

        # gate layers
        self.gate_layers = nn.ModuleList(
            [
                nn.Sequential(
                    DNN(
                        compute_input_dim(self.input_feature_columns),
                        self.gate_dnn_hidden_units,
                        activation=self.activation,
                        dropout_p=args.dropout_p,
                    ),
                    PredictionLayer(task="multiclass"),
                )
                for _ in range(len(self.task))
            ]
        )

        # task layers
        self.tower_layers = nn.ModuleList(
            [
                nn.Sequential(
                    DNN(
                        self.expert_dnn_hidden_units[-1],
                        self.tower_dnn_hidden_units,
                        activation=self.activation,
                        dropout_p=args.dropout_p,
                    ),
                    nn.Linear(self.tower_dnn_hidden_units[-1], 1, bias=False),
                    PredictionLayer(task=t),
                )
                for t in self.task
            ]
        )

    @classmethod
    def code(cls):
        return "mmoe"

    def forward(self, batch):
        x = batch["x"].float()
        feature_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.input_feature_columns, self.feature_index, self.embedding_dict
        )
        expert_input = combined_dnn_input(feature_embedding_list, dense_value_list)  # [B, dim]

        expert_out = [
            e(expert_input) for e in self.expert_layers
        ]  # len(num_expert) --> [B, expert_dnn_hidden_units[-1]]
        expert_out = torch.stack(expert_out, dim=0)  # [num_experts, B, dim]
        gates_out = [g(expert_input) for g in self.gate_layers]  # len(num_gate) --> [B, tower_num]
        tower_input = [
            go.t().unsqueeze(2).expand(-1, -1, expert_out.shape[-1]) * expert_out for go in gates_out
        ]  # len(num_gate) --> [num_expert, B, expert_dnn_hidden_units[-1]]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        tower_pred = [tl(ti) for tl, ti in zip(self.tower_layers, tower_input)]  # len(num_tower) --> [B, 1]

        return {k: v for k, v in zip(self.metrics_list, tower_pred)}
