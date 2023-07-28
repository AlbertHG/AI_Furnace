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
)


class LSTMModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.class_num = args.class_num
        self.embedding_dim = args.embedding_dim
        self.d_model = args.embedding_dim
        self.series_len = args.series_len
        self.n_layers = args.n_layers

        self.task_inputs_dim = self.d_model
        self.task_hidden_units = args.task_hidden_units
        self.activation = args.activation
        self.task = args.task

        self.app_embedding = ItemEmbedding(item_num=self.class_num + 5, embed_size=self.embedding_dim)
        self.position_embedding = AbsolutePositionalEmbedding(d_model=self.d_model)
        self.time_embedding = TemporalEmbedding(self.d_model)
        self.delta_t_embedding = DenseFeatureEncoding(d_model=self.d_model, hidden_embed_dim=self.d_model)
        self.dropout = nn.Dropout(args.dropout_p)

        self.lstm_layer = nn.LSTM(self.embedding_dim, self.embedding_dim, num_layers=self.n_layers, batch_first=True)

        self.task_layer = DNN(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "lstm"

    def forward(self, batch):
        _return = {}
        embed = self._item_add_feature(batch)

        embed, (_, _) = self.lstm_layer(embed)
        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)
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
