# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : mnist 训练器

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from trainers import AbstractTrainer
from .utils.metrics import accuracy_topk, accuracy_top1
from .utils.loss import SCELoss
from utils import fix_random_seed_as
from config import *


class ImageClassificationTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.args.metrics_meter_type = "avg"  # 设定 log meter 类型
        # nnl_loss + log_softmax  等价于  F.cross_entropy
        # self.task_loss = F.nll_loss
        self.task_loss = F.cross_entropy
        # self.task_loss = SCELoss(alpha=0.1, beta=1, num_classes=10)

    @classmethod
    def code(cls):
        return "image_classification"

    def _batch_to_device(self, batch):
        batch_size = batch[0].size(0)
        batch = [batch[0].to(self.device), batch[1].to(self.device)]
        return batch_size, batch

    def calculate_loss(self, batch, scores):
        _, label = batch
        output_logits = scores["y_pred"]
        loss = self.task_loss(output_logits, label)
        return loss

    def calculate_metrics(self, batch, scores):
        _, label = batch
        y_pred = scores["y_pred"]
        metrics = accuracy_top1(label, y_pred)
        # metrics = accuracy_topk(label, y_pred, ks=self.args.ks)
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
