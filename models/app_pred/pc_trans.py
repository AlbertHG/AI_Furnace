import torch.nn as nn
import torch.nn.functional as F

from models import AbstractModel
from layers import (
    ItemEmbedding,
    AbsolutePositionalEmbedding,
    TemporalEmbedding,
    DenseFeatureEncoding,
    PredictionLayer,
    DNN,
    SAS,
    PCTransformer,
)


class PCTransModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.n_layers = args.n_layers
        self.n_heads = args.n_heads

        self.class_num = args.class_num
        self.embedding_dim = args.embedding_dim
        self.d_model = args.embedding_dim

        self.task_inputs_series_len = args.task_inputs_series_len
        self.task_inputs_dim = self.task_inputs_series_len * self.d_model
        self.task_hidden_units = args.task_hidden_units
        self.activation = args.activation
        self.task = args.task

        self.app_embedding = ItemEmbedding(item_num=self.class_num + 5, embed_size=self.embedding_dim)
        self.position_embedding = AbsolutePositionalEmbedding(d_model=self.d_model)
        self.time_embedding = TemporalEmbedding(self.d_model)
        self.delta_t_embedding = DenseFeatureEncoding(d_model=self.d_model, hidden_embed_dim=self.d_model)
        self.dropout = nn.Dropout(args.dropout_p)

        self.trans_layer = PCTransformer(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_ff=self.d_model,
            sub_k_list=args.sub_k_list,
            epsilon=args.epsilon,
            dropout_p=args.dropout_p,
        )

        self.task_layer = DNN(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        """
        基于PC_transformer的 App 序列预测模型, dummy 训练方式

        """
        return "pc_trans_dummy"

    def forward(self, batch):
        _return = {}
        embed = self._item_add_feature(batch)
        if self.training:
            pos_embed = self.app_embedding(batch["pos"])
            neg_embed = self.app_embedding(batch["neg"])
            pos_logits, neg_logits, embed = self.trans_layer(embed, pos_embed, neg_embed)
            _return["pos_logits"] = pos_logits
            _return["neg_logits"] = neg_logits
        else:
            embed = self.trans_layer.predict(embed)

        output_logits, y_pred = self._task(embed, self.task_inputs_series_len)
        _return["output_logits"] = output_logits
        _return["y_pred"] = y_pred

        return _return

    def _item_add_feature(self, batch):
        """
        为 item embed 添加 特征 embed

        Args:
            batch (dict): 来自dataloader的一个 batch 的数据

        Returns:
            [Tensor]: [batch size, series len, d_model]
        """
        app_embed = self.app_embedding(batch["app"])
        hour_embed = self.time_embedding(batch)["hour"]
        deltat_embed = self.delta_t_embedding(batch["delta_t"])

        embed = app_embed + hour_embed + deltat_embed

        position_embed = self.position_embedding(batch["app"])
        embed = self.dropout(embed + position_embed)

        return embed

    def _task(self, embed, inputs_series_len):

        # --- concat ---#
        embed = embed[:, -inputs_series_len:, :].squeeze(1)
        embed = embed.view(embed.size(0), -1)

        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)

        return output_logits, y_pred


class SASTransModel(PCTransModel):
    def __init__(self, args):
        super().__init__(args)
        self.trans_layer = SAS(self.n_layers, self.n_heads, self.d_model, args.dropout_p)

    @classmethod
    def code(cls):
        """
        基于 SAS 的 App 序列预测模型, dummy 训练方式

        """
        return "sas_dummy"
