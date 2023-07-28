# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      :

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config

from trainers import AbstractTrainer
from .utils.metrics import accuracy_topk_add_bias, accuracy_top1
from .utils.loss import SmoothTopkSVM
from loggers import MetricTensorBoardPrinter


class E2E_Trainer(AbstractTrainer):
    """
    端到端的训练方式

    Args:
        AbstractTrainer ([type]): [description]
    """
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.args.metrics_meter_type = "avg"  # 设定 log meter 类型
        self.wechat_ratio = args.dataset_wechat_ratio

        self.task_loss = F.cross_entropy
        # self.task_loss = SmoothTopkSVM(n_classes=args.class_num, alpha=1.0, tau=1.0, k=max(args.ks))
        # self.task_loss.cpu() if self.device == "cpu" else self.task_loss.cuda()

        self.ingore_classes = get_ignore_class(args.class_num, args.ignore_class, self.device)

    @classmethod
    def code(cls):
        return "e2e"

    def calculate_loss(self, batch, scores):
        label = batch["label"]
        output_logits = scores["output_logits"]
        loss = self.task_loss(output_logits, label)
        return loss

    def calculate_metrics(self, batch, scores):
        label = batch["label"]
        y_pred = scores["y_pred"]
        metrics = accuracy_topk_add_bias(
            label, y_pred, ks=self.args.ks, bias=self.wechat_ratio, ignore_idxs=self.ingore_classes
        )

        return metrics

    def _get_state_dict(self, epoch, accum_iter):
        """
        获取模型的参数，通过dict保存

        Returns:
            dict: [description]
        """
        return {
            # 保存模型model的参数
            config.STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            config.STEPS_DICT_KEY: (epoch, accum_iter),
        }


class LSTM_Trainer(E2E_Trainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

    @classmethod
    def code(cls):
        return "lstm"

    def calculate_loss(self, batch, scores):
        label = batch["pos"].view(-1)
        output_logits = scores["output_logits"]
        output_logits = output_logits.view(output_logits.size()[0] * output_logits.size()[1], -1)
        loss = self.task_loss(output_logits, label)
        return loss

    def calculate_metrics(self, batch, scores):
        label = batch["label"]
        y_pred = scores["y_pred"][:, -1]
        metrics = accuracy_topk_add_bias(
            label, y_pred, ks=self.args.ks, bias=self.wechat_ratio, ignore_idxs=self.ingore_classes
        )

        return metrics


class E2E_Auxiliary_Trainer(AbstractTrainer):
    """
    端到端多分类 + item 级别的二分类 的训练方式

    Args:
        AbstractTrainer ([type]): [description]
    """

    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.args.metrics_meter_type = "avg"  # 设定 log meter 类型
        self.feature_loss = nn.BCEWithLogitsLoss()
        self.task_loss = F.cross_entropy

        # self.task_loss = SmoothTopkSVM(n_classes=args.class_num, alpha=1.0, tau=1.0, k=max(args.ks))
        # self.task_loss.cpu() if self.device == "cpu" else self.task_loss.cuda()

        self.wechat_ratio = args.dataset_wechat_ratio
        self.ingore_classes = get_ignore_class(args.class_num, args.ignore_class, self.device)

    @classmethod
    def code(cls):
        return "e2e_auxiliary"

    def add_extra_loggers(self):
        self.train_loggers.append(
            MetricTensorBoardPrinter(self.writer, key="att_loss", graph_name="Att_Loss", group_name="Train")
        )
        self.train_loggers.append(
            MetricTensorBoardPrinter(self.writer, key="task_loss", graph_name="Task_Loss", group_name="Train")
        )

    def calculate_loss(self, batch, scores):
        padding_mask = batch["padding_mask"]
        label = batch["label"]
        pos_logits = scores["pos_logits"]
        neg_logits = scores["neg_logits"]
        output_logits = scores["output_logits"]

        pos_labels = torch.ones(pos_logits.shape, device=self.device)
        neg_labels = torch.zeros(neg_logits.shape, device=self.device)
        indices = np.where(padding_mask.cpu() == False)

        att_loss = self.feature_loss(pos_logits[indices], pos_labels[indices])
        att_loss += self.feature_loss(neg_logits[indices], neg_labels[indices])
        task_loss = self.task_loss(output_logits, label)
        loss = att_loss + task_loss
        self.train_average_meter_set.update("att_loss", att_loss.item())
        self.train_average_meter_set.update("task_loss", task_loss.item())

        return loss

    def calculate_metrics(self, batch, scores):
        label = batch["label"]
        y_pred = scores["y_pred"]
        metrics = accuracy_topk_add_bias(
            label, y_pred, ks=self.args.ks, bias=self.wechat_ratio, ignore_idxs=self.ingore_classes
        )
        return metrics

    def _get_state_dict(self, epoch, accum_iter):
        """
        获取模型的参数，通过dict保存

        Returns:
            dict: [description]
        """
        return {
            # 保存模型model的参数
            config.STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            config.STEPS_DICT_KEY: (epoch, accum_iter),
        }


def get_ignore_class(class_num, ignore_class, device):
    """计算 类别排序的 mask，值为 0 的位置的class不参与 tok k 排序

    example：ignore_class=[1,3]--->ignore_idxs = [1, 0, 1, 0, 1, 1, 1, 1]，类别 1 和 3 不参与 top k 计算

    Args:
        class_num ([type]): 总类别数
        ignore_class ([type]): 需要忽略的类别
        device ([type]):

    Returns:
        [type]: [description]
    """

    ingore_idxs = torch.ones(class_num).to(device)
    if isinstance(ignore_class, list) and len(ignore_class) != 0:
        for i in ignore_class:
            ingore_idxs[i] = 0
    return ingore_idxs
