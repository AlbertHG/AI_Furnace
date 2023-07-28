# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2022/01/05 12:00
# @Function      : multi task 训练器

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from loguru import logger as loguru_logger
import numpy as np

from trainers import AbstractTrainer
from utils import fix_random_seed_as
from pathlib import Path

import config
import loggers


class MMOETrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

        assert len(set(args.metrics_list)) == len(args.task_label) == len(args.task)

        # self.metrics_list = args.metrics_list
        self.task = args.task
        self.task_label = args.task_label
        self.args.metrics_meter_type = "value"  # 设定 log meter 类型
        self.training_auc_mode = self.args.training_auc_mode  # 设定 AUC 计算需要的 label 和 pred 的存储模式

        # 存储每一个batch的预测值，用来计算AUC的
        self.predict_list = {ml: [] for ml, t in zip(self.metrics_list, self.task)}
        # 存储每一个batch的label值，用来计算AUC的
        self.label_list = {ml: [] for ml, t in zip(self.metrics_list, self.task)}
        loss_func = {"binary": "bce", "regression": "mse"}
        self.task_loss = {ml: self._create_loss_func(loss_func[t]) for ml, t in zip(self.metrics_list, self.task)}

    @classmethod
    def code(cls):
        return "mmoe"

    def train(self):
        """
        重载 基类_base.py 的 train()  定义了多轮训练的流程,
        比 _base.py 中增添了 各种中间结果集合的 reset 过程.
        """
        epoch = self.epoch_start
        # 全局累加计数器，记录已经遍历过的样本数
        accum_iter = self.accum_iter_start

        for epoch in range(self.epoch_start, self.num_epochs):
            fix_random_seed_as(epoch)
            self._metric_input(mode="reset")
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self._metric_input(mode="reset")
            self.validate(epoch, accum_iter, mode="val")

        self._metric_input(mode="reset")
        self.validate(epoch, accum_iter, mode="test")

        # 训练结束之后，Log 的善后
        self.logger_service.complete({"state_dict": (self._get_state_dict(epoch, accum_iter))})
        self.writer.close()

    def add_extra_loggers(self):
        # model_checkpoint = Path(self.export_root).joinpath("models")
        # self.val_loggers.append(loggers.HistoryModelLogger(model_checkpoint, self.best_metric))

        self.train_loggers += [
            loggers.MetricTensorBoardPrinter(
                self.writer, key="{}_loss".format(ml), graph_name="{}_Loss".format(ml), group_name="Train"
            )
            for ml in self.metrics_list
        ]

    def calculate_loss(self, batch, scores):
        loss = 0
        for ml, tl in zip(self.metrics_list, self.task_label):
            cache_loss = self.task_loss[ml](scores[ml], batch[tl].float())
            self.train_average_meter_set.update("{}_loss".format(ml), cache_loss.item())
            loss += cache_loss

        return loss

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


class AITMTrainer(MMOETrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.constraint_weight = args.constraint_weight

    @classmethod
    def code(cls):
        return "aitm"

    def calculate_loss(self, batch, scores):
        loss = 0
        i = 0
        for ml, tl in zip(self.metrics_list, self.task_label):
            this_output = scores[ml]
            this_label = batch[tl]
            this_loss = self.task_loss[ml](this_output, this_label.float())

            self.train_average_meter_set.update("{}_loss".format(ml), this_loss.item())

            loss += this_loss

            # Behavioral Expectation Calibrator loss
            # 保证前一个任务应该比后一个任务具有更高的端到端转换概率
            # 当前任务转换概率 - 上一个任务转换概率 <= 0 , 则符合要求 ， loss = 0
            if i > 0:
                loss += self.constraint_weight * torch.sum(
                    torch.maximum(this_output - prev_output, torch.zeros_like(this_output))
                )
            prev_output = this_output
            i += 1

        return loss


class ESMMTrainer(MMOETrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        assert isinstance(args.task_label, list) and len(args.task_label) == 2
        for i in self.task:
            assert i == "binary", "[!] ESMM task must be `binary`. "
        loguru_logger.warning(
            "PLEASE DUBBLE CHECK!!  CTR Label: {};  CTCVR Label: {}".format(self.task_label[0], self.task_label[1])
        )

    @classmethod
    def code(cls):
        return "esmm"
