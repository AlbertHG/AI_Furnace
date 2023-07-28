# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : 配置信息读取

import argparse
from .template import set_template
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS


class Parser(object):
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv

    def parse(self):
        conf = {}
        conf.update(self.parse_top())
        conf.update(self.parse_dataloader())
        conf.update(self.parse_dataset())
        conf.update(self.parse_trainer())
        conf.update(self.parse_model())
        conf.update(self.parse_experiment())

        set_template(conf)
        return conf

    def parse_top(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--run_mode", type=str, choices=["train", "analyse"], help="程序的运行模式")
        parser.add_argument("--template", nargs="+", type=str, default=["poi/transformer"], help="templates的配置文件名")
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_experiment(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--experiment_dir", type=str, help="实验数据存放的根目录名称")
        parser.add_argument("--experiment_description", type=str, help="trainer 侧实验数据存放文件夹的描述")
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_dataset(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--dataset_path", type=str, help="数据集路径")
        parser.add_argument("--trainset_path", type=str, help="训练集文件路径")
        parser.add_argument("--valset_path", type=str, help="验证集文件路径")
        parser.add_argument("--testset_path", type=str, help="测试集文件路径")
        parser.add_argument("--class_num", type=int, help="分类数")
        parser.add_argument("--series_len", type=int, help="模型的输入样本长度")
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_dataloader(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--dataloader_code", type=str, choices=DATALOADERS.keys())
        parser.add_argument("--batch_size", type=int, help="Batch Size")
        parser.add_argument("--train_batch_size", type=int, help="训练集Batch Size")
        parser.add_argument("--val_batch_size", type=int, help="验证集Batch Size")
        parser.add_argument("--test_batch_size", type=int, help="测试集Batch Size")
        parser.add_argument("--dataloader_random_seed", type=float, default=10086, help="随机负样本构造种子数")
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_trainer(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--trainer_code", type=str, choices=TRAINERS.keys())
        parser.add_argument("--resume_training", type=str, help="是否从断点处开始继续训练模型")
        parser.add_argument("--resume_node", type=str,  help="选择加载 `recent` 或者 `best` 断点")
        # device #
        parser.add_argument("--device", type=str, choices=["cpu", "cuda"])
        parser.add_argument("--num_gpu", type=int)
        parser.add_argument("--device_idx", type=str)
        # mixed precision #
        parser.add_argument("--use_amp", type=bool, default=False, help="Automatic Mixed Precision.")
        # optimizer #
        parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"])
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--weight_decay", type=float, help="l2 regularization")
        parser.add_argument("--momentum", type=float, help="SGD momentum")
        # lr scheduler #
        parser.add_argument("--enable_lr_schedule", type=bool)
        parser.add_argument("--decay_step", type=int, help="Decay step for StepLR")
        parser.add_argument("--gamma", type=float, help="Gamma for StepLR")
        # epochs #
        parser.add_argument("--num_epochs", type=int, help="Number of epochs for training")
        # logger #
        parser.add_argument("--log_period_as_iter", type=int, default=10, help="每隔多少Batch_size打印一次log")
        parser.add_argument("--metrics_meter_type", type=str, choices=["sum", "avg", "val"], help="metrics呈现的数据类型")
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_model(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--model_code", type=str, choices=MODELS.keys())
        parser.add_argument("--model_init_seed", type=float, default=0, help="固定模型训练的随机种子")

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)
