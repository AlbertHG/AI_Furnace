import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from loguru import logger as loguru_logger
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json


from ._base import AbstractAnalyzer


class CTRAnalyzer(AbstractAnalyzer):
    def __init__(self, args, model, test_loader, export_root):
        super().__init__(args, model, test_loader, export_root)
        assert isinstance(args.task_label, str) and len(self.metrics_list) == 1
        self.task_label = args.task_label
        loguru_logger.info("PLEASE DUBBLE CHECK.  CTR Label: {}".format(self.task_label))
        self.count = []
        self.predict_list = []
        self.label_list = []
        self.args.metrics_meter_type = "value"
        self.training_auc_mode = self.args.training_auc_mode

    @classmethod
    def code(cls):
        return "ctr"

    def calculate_metrics(self, batch, scores):
        label = np.squeeze(batch[self.task_label].cpu().data.numpy().astype("float64")).tolist()
        y_pred = np.squeeze(scores["y_pred"].cpu().data.numpy()).tolist()
        self._auc_input(label, y_pred, mode="add_up")
        try:
            auc = roc_auc_score(np.array(self.label_list), np.array(self.predict_list))
        except:
            auc = 0.5
        metrics = {"AUC": auc}

        imei = np.expand_dims(batch["imei"].cpu().data.numpy().astype("int32"), axis=1)
        label = batch[self.task_label].cpu().data.numpy().astype("int32")
        y_pred = np.expand_dims(np.squeeze(scores["y_pred"].cpu().data.numpy()), axis=1)

        cache = np.concatenate((imei, label, y_pred), axis=1)

        cache_df = pd.DataFrame(cache)
        cache_df.to_csv(
            os.path.join(
                "./experiments",
                self.args.experiment_dir,
                self.args.experiment_description,
                "imei_label_pred_result.csv",
            ),
            mode="a",
            header=False,
        )

        return metrics

    def _auc_input(self, label=[], pred=[], mode="add_up"):
        if mode == "add_up":
            self.label_list += label
            self.predict_list += pred
        elif mode == "value":
            self.label_list = label
            self.predict_list = pred
        elif mode == "reset":
            self.label_list = []
            self.predict_list = []
        else:
            raise ValueError('[!] mode Error, choise = ["add", "reset", "value"]')
