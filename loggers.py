# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/26 12:00
# @Function      : logger.py

import os
import torch
from abc import ABCMeta, abstractmethod
from loguru import logger as loguru_logger


def save_state_dict(state_dict, path, filename):
    """
    将对象保存到磁盘文件中。

    Args:
        state_dict (object): 模型参数 dict
        path (str): 文件路径
        filename (str): 文件名
    """
    torch.save(state_dict, os.path.join(path, filename))


def save_model(net, path, filename, input=None):
    """
    将整个模型对象保存到磁盘文件中。

    Args:
        state_dict (object): 模型对象
        path (str): 文件路径
        filename (str): 文件名
    """
    # net.eval()
    # traced_script_module = torch.jit.trace(net, input)
    # traced_script_module.save(os.path.join(path, filename))
    torch.save(net, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None):
        """
        Args:
            object ([type]): [description]
            train_loggers ([type], optional): [description]. Defaults to None.
            val_loggers ([type], optional): [description]. Defaults to None.
        """
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)


class AbstractLogger(metaclass=ABCMeta):
    """
    Logger 的抽象类
    """

    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class HistoryModelLogger(AbstractLogger):
    """
    保存每一次的模型训练的参数（增量保存）

    Args:
        AbstractLogger (class): 父类
        checkpoint_path (str): 文件路径
    """

    def __init__(self, checkpoint_path, metric_key):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.metric_key = metric_key

    def log(self, *args, **kwargs):
        epoch = kwargs["epoch"]
        metric = kwargs[self.metric_key]
        filename = "checkpoint-epoch{}-metric{:.5f}".format(epoch, metric)

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs["state_dict"]
            state_dict["epoch"] = kwargs["epoch"]
            save_state_dict(state_dict, self.checkpoint_path, filename)


class RecentModelLogger(AbstractLogger):
    """
    保存当前次的模型训练的参数

    Args:
        AbstractLogger (class): 父类
        checkpoint_path (str): 文件路径
        filename (str, optional): 文件名. Defaults to 'checkpoint-recent.pth'.
    """

    def __init__(self, checkpoint_path, filename="checkpoint-recent.pth"):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs["epoch"]

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs["state_dict"]
            state_dict["epoch"] = kwargs["epoch"]
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs["state_dict"], self.checkpoint_path, self.filename + ".final")


class BestModelLogger(AbstractLogger):
    """
    保存整个训练过程中最好的那一版模型参数

    Args:
        AbstractLogger (class): 父类
        checkpoint_path (str): 文件路径
        metric_key (str, optional): 基于什么指标下的最佳值. Defaults to 'Acc@4'.
        filename (str, optional): 文件名. Defaults to 'best_acc_model.pth'.
    """

    def __init__(self, checkpoint_path, metric_key="Acc@4", filename="best_acc_model.pth"):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.0
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            loguru_logger.info("Update Best {} Model at Epoch {}".format(self.metric_key, kwargs["epoch"]))
            self.best_metric = current_metric
            save_state_dict(kwargs["state_dict"], self.checkpoint_path, self.filename)


class MetricTensorBoardPrinter(AbstractLogger):
    """
    负责训练/验证/测试过程中的各类指标数据 整合tensorboard writer的写入

    Args:
        AbstractLogger (class): 父类
        writer (object): tensorboard writer类对象
        key (str, optional): 指标的值. Defaults to 'train_loss'.
        graph_name (str, optional): 指标的名称. Defaults to 'Train Loss'.
        group_name (str, optional): 指标所在的组的名称，tensorboard上会进行分组展示. Defaults to 'metric'.
    """

    def __init__(self, writer, key="train_loss", graph_name="Train Loss", group_name="metric"):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(
                self.group_name + "/" + self.graph_label,
                kwargs[self.key],
                kwargs["accum_iter"],
            )
        else:
            self.writer.add_scalar(self.group_name + "/" + self.graph_label, 0, kwargs["accum_iter"])

    def complete(self, *args, **kwargs):
        self.writer.close()
