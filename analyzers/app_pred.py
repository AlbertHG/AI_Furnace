import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from trainers.utils.metrics import accuracy_topk_add_bias, accuracy_top1
from trainers.app_pred import get_ignore_class
from config import *
from ._base import AbstractAnalyzer
from post_processor import *


class E2EAnalyzer(AbstractAnalyzer):
    def __init__(self, args, model, test_loader, export_root):
        super().__init__(args, model, test_loader, export_root)
        self.args.metrics_meter_type = "avg"
        self.wechat_ratio = args.dataset_wechat_ratio
        self.ingore_classes = get_ignore_class(args.class_num, args.ignore_class, self.device)

    @classmethod
    def code(cls):
        return "e2e"

    def calculate_metrics(self, batch, scores):
        label = batch["label"]
        y_pred = scores["y_pred"]
        metrics = accuracy_topk_add_bias(
            label, y_pred, ks=self.args.ks, bias=self.wechat_ratio, ignore_idxs=self.ingore_classes
        )

        return metrics


class LSTMAnalyzer(AbstractAnalyzer):
    def __init__(self, args, model, test_loader, export_root):
        super().__init__(args, model, test_loader, export_root)
        self.args.metrics_meter_type = "avg"
        self.wechat_ratio = args.dataset_wechat_ratio
        self.ingore_classes = get_ignore_class(args.class_num, args.ignore_class, self.device)

    @classmethod
    def code(cls):
        return "lstm"

    def calculate_metrics(self, batch, scores):
        label = batch["label"]
        y_pred = scores["y_pred"][:, -1]
        metrics = accuracy_topk_add_bias(
            label, y_pred, ks=self.args.ks, bias=self.wechat_ratio, ignore_idxs=self.ingore_classes
        )

        return metrics


class E2E_Auxiliary_Analyzer(AbstractAnalyzer):
    def __init__(self, args, model, test_loader, export_root):
        super().__init__(args, model, test_loader, export_root)
        self.args.metrics_meter_type = "avg"
        self.wechat_ratio = args.dataset_wechat_ratio
        self.ingore_classes = get_ignore_class(args.class_num, args.ignore_class, self.device)

    @classmethod
    def code(cls):
        return "e2e_auxiliary"

    def calculate_metrics(self, batch, scores):
        label = batch["label"]
        y_pred = scores["y_pred"]
        metrics = accuracy_topk_add_bias(
            label, y_pred, ks=self.args.ks, bias=self.wechat_ratio, ignore_idxs=self.ingore_classes
        )
        return metrics

    def calculate_metrics111(self, batch, scores):
        label = batch["label"]
        y_pred = scores["y_pred"]
        batch_size = y_pred.size(0)
        
        longtail = batch["longtail"]
        hist_series = batch["hist_series"]
        app = hist_series[:, -100:].tolist()
        # app1 = batch['app'].tolist()

        # for a,a1 in zip(app,app1):
        #     print(a,'\n',a1)
        #     print(a==a1)



        hist_series = hist_series.tolist()

        # markov_pred
        markov_pred = []
        for i in hist_series:
            i = [_i for _i in i if _i != -1]
            markov_pred.append(first_order_markov_prediction_method(i, 2000, max(self.args.ks)))

        # # freq_pred
        # freq_pred = []
        # for i in hist_series:
        #     i = [_i for _i in i if _i != -1]
        #     freq_pred.append(statistical_frequency(hist_series[-100:], max(self.args.ks)))

        # model_pred
        _, model_pred = torch.topk(y_pred, k=max(self.args.ks), dim=-1)
        model_pred = replace(model_pred, 0, longtail)  # 把预测出来的0类用，用户的长尾app替换
        y_pred = model_pred
        

        # 聚合
        # _model_pred = model_pred.tolist()
        # _y_pred = []
        # for h, i, j in zip(app, _model_pred, markov_pred):
        #     _y_pred.append(stra_fun(h, i[::-1] , j[::-1], 35, max(self.args.ks)))
        # _y_pred = torch.tensor(_y_pred).type_as(model_pred).to(model_pred.device)
        # y_pred = _y_pred

        label = label.view(-1, 1)  # [n] --> [n,1]
        metrics = {}
        for k in self.args.ks:
            top = (label == y_pred[:, 0:k]).sum().item()
            metrics["Top@{}".format(k)] = top / batch_size

        return metrics
