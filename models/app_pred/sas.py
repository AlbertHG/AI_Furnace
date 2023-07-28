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


class SASTransModel(AbstractModel):
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
        self.dropout = nn.Dropout(args.dropout_p)

        self.trans_layer = SAS(self.n_layers, self.n_heads, self.d_model, args.dropout_p)
        self.task_layer = DNN(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        """
        基于 SAS 的 App 序列预测模型, sas 论文中的数据构造和训练方式

        """
        return "sas_sas"

    def forward(self, batch):
        _return = {}
        padding_mask = batch["padding_mask"]
        embed = self._item_add_feature(batch)
        if self.training:
            pos_embed = self.app_embedding(batch["pos"])
            neg_embed = self.app_embedding(batch["neg"])
            pos_logits, neg_logits, embed = self.trans_layer(embed, pos_embed, neg_embed, padding_mask=padding_mask)
            _return["pos_logits"] = pos_logits
            _return["neg_logits"] = neg_logits
        else:
            embed = self.trans_layer.predict(embed, padding_mask=padding_mask)

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
        position_embed = self.position_embedding(batch["app"])
        embed = self.dropout(app_embed + position_embed)
        return embed

    def _task(self, embed, inputs_series_len):

        # --- concat ---#
        embed = embed[:, -inputs_series_len:, :].squeeze(1)
        embed = embed.view(embed.size(0), -1)

        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)

        return output_logits, y_pred
