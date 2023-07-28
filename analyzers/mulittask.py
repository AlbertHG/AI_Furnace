import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from loguru import logger as loguru_logger
import torch.nn.functional as F
import numpy as np
import json


from ._base import AbstractAnalyzer


class MMOEAnalyzer(AbstractAnalyzer):
    def __init__(self, args, model, test_loader, export_root):
        super().__init__(args, model, test_loader, export_root)

        assert len(set(args.metrics_list)) == len(args.task_label) == len(args.task)
        loguru_logger.info("PLEASE DUBBLE CHECK!!  Label List: {}".format(args.task_label))

        self.task = args.task
        self.task_label = args.task_label
        self.args.metrics_meter_type = "value"
        self.training_auc_mode = self.args.training_auc_mode

        self.count = []
        # 存储每一个batch的预测值，用来计算AUC的
        self.predict_list = {ml: [] for ml, t in zip(self.metrics_list, self.task)}
        # 存储每一个batch的label值，用来计算AUC的
        self.label_list = {ml: [] for ml, t in zip(self.metrics_list, self.task)}

    @classmethod
    def code(cls):
        return "mmoe"

    def calculate_metrics(self, batch, scores):
        metrics = {}
        for ml, t, tl in zip(self.metrics_list, self.task, self.task_label):
            label = np.squeeze(batch[tl].cpu().data.numpy().astype("float64")).tolist()
            pred = np.squeeze(scores[ml].cpu().data.numpy()).tolist()
            if self.model.training:
                self._metric_input(label, pred, mode=self.training_auc_mode, ml=ml)
            else:
                self._metric_input(label, pred, mode="add_up", ml=ml)

            if t == "binary":  # 计算 AUC
                try:
                    auc = roc_auc_score(np.array(self.label_list[ml]), np.array(self.predict_list[ml]))
                except:
                    auc = 0.5
                metrics[ml] = auc
            elif t == "regression":  # 计算 mse
                mse = mean_squared_error(np.array(self.label_list[ml]), np.array(self.predict_list[ml]))
                metrics[ml] = mse
            else:
                raise ValueError

        return metrics

    def _metric_input(self, label=[], pred=[], mode="add_up", ml=None):
        """存储 metric 计算需要的 label 和 pred 值

        Args:
            label (list, optional): [description]. Defaults to [].
            pred (list, optional): [description]. Defaults to [].
            mode (str, optional): 存储模式切换,
                -- ``add_up`` 表示累计存储;
                -- ``value`` 表示只存储当前batch;
                -- ``reset`` 表示重置list . Defaults to "add_up".
        """

        if mode == "add_up":
            self.label_list[ml] += label
            self.predict_list[ml] += pred
        elif mode == "value":
            self.label_list[ml] = label
            self.predict_list[ml] = pred
        elif mode == "reset":
            for ml in self.metrics_list:
                self.label_list[ml] = []
                self.predict_list[ml] = []
        else:
            raise ValueError('[!] mode Error, choise = ["add", "reset", "value"]')
