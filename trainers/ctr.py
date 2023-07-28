# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : CTR 训练器

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from loguru import logger as loguru_logger

from trainers import AbstractTrainer
from utils import fix_random_seed_as
from pathlib import Path
from config import *
from loggers import HistoryModelLogger
from .utils.loss import BSCELoss


class CTRTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        assert isinstance(args.task_label, str) and len(self.metrics_list) == 1
        self.task_label = args.task_label
        loguru_logger.warning("PLEASE DUBBLE CHECK.  CTR Label: {}".format(self.task_label))
        self.args.metrics_meter_type = "value"  # 设定 log meter 类型
        self.training_auc_mode = self.args.training_auc_mode  # 设定 AUC 计算需要的 label 和 pred 的存储模式
        self.predict_list = []
        self.label_list = []
        # self.task_loss = F.binary_cross_entropy
        # self.task_loss = BSCELoss(1, 1)
        self.task_loss = nn.BCELoss()

    @classmethod
    def code(cls):
        return "ctr"

    def train(self):
        """
        定义了多轮训练的流程, 比 _base.py 中增添了 各种中间结果集合的 reset 过程.
        """
        epoch = self.epoch_start
        # 全局累加计数器，记录已经遍历过的样本数
        accum_iter = self.accum_iter_start

        for epoch in range(self.epoch_start, self.num_epochs):
            fix_random_seed_as(epoch)
            self._auc_input(mode="reset")
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self._auc_input(mode="reset")
            self.validate(epoch, accum_iter, mode="val")

        self._auc_input(mode="reset")
        self.validate(epoch, accum_iter, mode="test")

        # 训练结束之后，Log 的善后
        self.logger_service.complete({"state_dict": (self._get_state_dict(epoch, accum_iter))})

        self.writer.close()

    def add_extra_loggers(self):
        # model_checkpoint = Path(self.export_root).joinpath("models")
        # self.val_loggers.append(HistoryModelLogger(model_checkpoint, self.best_metric))
        pass

    def calculate_loss(self, batch, scores):
        label = batch[self.task_label].float()
        y_pred = scores["y_pred"]
        loss = self.task_loss(y_pred, label)
        return loss

    def calculate_metrics(self, batch, scores):
        label = np.squeeze(batch[self.task_label].cpu().data.numpy().astype("float64")).tolist()
        y_pred = np.squeeze(scores["y_pred"].cpu().data.numpy()).tolist()
        if self.model.training:
            self._auc_input(label, y_pred, mode=self.training_auc_mode)
        else:
            self._auc_input(label, y_pred, mode="add_up")
        try:
            auc = roc_auc_score(np.array(self.label_list), np.array(self.predict_list))
        except:
            auc = 0.5
        metrics = {self.metrics_list[0]: auc}
        return metrics

    def _get_state_dict(self, epoch, accum_iter):
        """
        获取模型的参数，通过dict保存

        Returns:
            dict: [description]
        """
        return {
            # 保存模型model的参数
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            STEPS_DICT_KEY: (epoch, accum_iter),
        }

    def _auc_input(self, label=[], pred=[], mode="add_up"):
        """存储 AUC 计算需要的 label 和 pred 值

        Args:
            label (list, optional): [description]. Defaults to [].
            pred (list, optional): [description]. Defaults to [].
            mode (str, optional): 存储模式切换,
                -- ``add_up`` 表示累计存储;
                -- ``value`` 表示只存储当前batch;
                -- ``reset`` 表示重置list . Defaults to "add_up".
        """
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
