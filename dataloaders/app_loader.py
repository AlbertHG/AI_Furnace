# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : app 预测算法 dalaloader

import os
import ast
import json
import lmdb
import pickle
import sqlite3
import linecache
import itertools
import pandas as pd
import numpy as np
from loguru import logger as loguru_logger
from tqdm import tqdm
from abc import *

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .utils import *
from dataloaders import AbstractDataloader


class DummyData(object):
    def __init__(self, series_len, class_num, rng):
        self.series_len = series_len
        self.class_num = class_num
        self.rng = rng
        self.dummy = class_num
        self.one_example = {}

    def create_data(self, data, mode):
        """app 序列数据的构造"""
        hist_list = [int(v) for v in data["app"]] if not isinstance(data["app"][0], int) else data["app"]
        hist_list = np.array(hist_list)
        self.one_example = {}
        if mode == "train":
            self.__train_data(hist_list)
        elif mode == "val":
            self.__val_data(hist_list)
        elif mode == "test":
            self.__test_data(hist_list)
        else:
            raise ValueError
        self.__add_extra_datas(data)

    def __train_data(self, hist_list):
        raw_hist_list = hist_list.copy()
        # 将序列中大于等于 self.class_num 的类别统一归为 0 类，当作长尾App
        hist_list[hist_list >= self.class_num] = 0

        app = np.full([self.series_len], 0, dtype=np.int32)
        pos = np.full([self.series_len], 0, dtype=np.int32)
        neg = np.full([self.series_len], 0, dtype=np.int32)
        padding_mask = np.ones([self.series_len], dtype=np.bool_)

        nxt = hist_list[-1]
        idx = self.series_len - 2

        app[-1] = self.dummy
        pos[-1] = nxt
        neg[-1] = random_neg(nxt, self.class_num, self.rng)
        padding_mask[-1] = False

        for i in reversed(hist_list[:-1]):
            app[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neg(nxt, self.class_num, self.rng)
            padding_mask[idx] = False
            nxt = i
            idx -= 1
            if idx == -1:
                break
        self.one_example.update(
            {
                "app": torch.tensor(app, dtype=torch.long),
                "pos": torch.tensor(pos, dtype=torch.long),
                "neg": torch.tensor(neg, dtype=torch.long),
                "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
                "label": torch.tensor(hist_list[-1], dtype=torch.long),
                # "raw_app": torch.tensor(raw_hist_list, dtype=torch.long),
            }
        )

    def __val_data(self, hist_list):
        """验证集数据构造"""

        raw_hist_list = hist_list.copy()
        label = hist_list[-1]
        # 将整段历史序列中除了最后一个item之外的其他item，其大于等于 self.class_num 的类别统一归为 0 类
        hist_list[:-1][hist_list[:-1] >= self.class_num] = 0
        hist_list[-1] = self.dummy
        app = padding1D(hist_list, self.series_len, 0, np.int32)
        padding_mask = padding1D(np.zeros_like(hist_list), self.series_len, 1, np.bool_)

        self.one_example.update(
            {
                "app": torch.tensor(app, dtype=torch.long),
                "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
                "label": torch.tensor(label, dtype=torch.long),
                "raw_app": torch.tensor(raw_hist_list, dtype=torch.long),
            }
        )

    def __test_data(self, hist_list):
        return self.__val_data(hist_list)

    def __add_extra_datas(self, data):
        """
        添加额外数据，比如时间数据
        """
        week = data["week"]
        hour = data["hour"]
        weekend = data["weekend"]
        delta_t = data["delta_t"][1:] + data["delta_t"][-1:]
        # duration = data["duration"]
        # timestamp = data["timestamp"]
        hist_list = data["app"]

        self.one_example.update(
            {
                # "timestamp": torch.tensor(padding1D(timestamp, self.series_len, 0, np.int32), dtype=torch.long),
                "hour": torch.tensor(padding1D(hour, self.series_len, 0, np.int32), dtype=torch.long),
                "week": torch.tensor(padding1D(week, self.series_len, 0, np.int32), dtype=torch.long),
                "weekend": torch.tensor(padding1D(weekend, self.series_len, 0, np.int32), dtype=torch.long),
                # "duration": torch.tensor(padding1D(duration, self.series_len, 0, np.int32), dtype=torch.long),
                "delta_t": torch.tensor(padding1D(delta_t, self.series_len, 0, np.int32), dtype=torch.long),
            }
        )

        imei = int(data["imei"])
        longtail = next((x1 for x1 in reversed(hist_list) if x1 >= self.class_num), 0)
        self.one_example.update(
            {
                "hist_series": torch.tensor(padding1D(data["hist_series"], 2000, -1, np.int32), dtype=torch.long),
                "longtail": torch.tensor(longtail, dtype=torch.long),
                "imei": torch.tensor(imei, dtype=torch.long),
            }
        )


class SasData(object):
    def __init__(self, series_len, class_num, rng):
        self.series_len = series_len
        self.class_num = class_num
        self.rng = rng
        self.one_example = {}

    def create_data(self, data, mode):
        """app 序列数据的构造"""
        hist_list = [int(v) for v in data["app"]] if not isinstance(data["app"][0], int) else data["app"]
        hist_list = np.array(hist_list)
        self.one_example = {}
        if mode == "train":
            self.__train_data(hist_list)
        elif mode == "val":
            self.__val_data(hist_list)
        elif mode == "test":
            self.__test_data(hist_list)
        else:
            raise ValueError
        self.__add_extra_datas(data)

    def __train_data(self, hist_list):

        raw_hist_list = hist_list.copy()
        # 将序列中大于等于 self.class_num 的类别统一归为 0 类，当作长尾App
        hist_list[hist_list >= self.class_num] = 0

        app = np.full([self.series_len], 0, dtype=np.int32)
        pos = np.full([self.series_len], 0, dtype=np.int32)
        neg = np.full([self.series_len], 0, dtype=np.int32)

        padding_mask = np.ones([self.series_len], dtype=np.bool_)
        nxt = hist_list[-1]
        idx = self.series_len - 1

        for i in reversed(hist_list[:-1]):
            app[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neg(nxt, self.class_num, self.rng)
            padding_mask[idx] = False
            nxt = i
            idx -= 1
            if idx == -1:
                break

        self.one_example.update(
            {
                "app": torch.tensor(app, dtype=torch.long),
                "pos": torch.tensor(pos, dtype=torch.long),
                "neg": torch.tensor(neg, dtype=torch.long),
                "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
                "label": torch.tensor(hist_list[-1], dtype=torch.long),
                "raw_app": torch.tensor(raw_hist_list, dtype=torch.long),
            }
        )

    def __val_data(self, hist_list):
        raw_hist_list = hist_list.copy()
        # 将整段历史序列中除了最后一个item之外的item，其大于等于 self.class_num 的类别统一归为 0 类
        hist_list[:-1][hist_list[:-1] >= self.class_num] = 0

        app = padding1D(hist_list[:-1], self.series_len, 0, np.int32)
        padding_mask = padding1D(np.zeros_like(hist_list[:-1]), self.series_len, 1, np.bool_)

        self.one_example.update(
            {
                "app": torch.tensor(app, dtype=torch.long),
                "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
                "label": torch.tensor(hist_list[-1], dtype=torch.long),
                "raw_app": torch.tensor(raw_hist_list, dtype=torch.long),
            }
        )

    def __test_data(self, hist_list):
        return self.__val_data(hist_list)

    def __add_extra_datas(self, data):
        week = data["week"][:-1]
        hour = data["hour"][:-1]
        weekend = data["weekend"][:-1]
        delta_t = data["delta_t"][:-1]
        # duration = data["duration"][:-1]
        timestamp = data["timestamp"][:-1]
        hist_list = data["app"][:-1]

        self.one_example.update(
            {
                "timestamp": torch.tensor(padding1D(timestamp, self.series_len, 0, np.int32), dtype=torch.long),
                "hour": torch.tensor(padding1D(hour, self.series_len, 0, np.int32), dtype=torch.long),
                "week": torch.tensor(padding1D(week, self.series_len, 0, np.int32), dtype=torch.long),
                "weekend": torch.tensor(padding1D(weekend, self.series_len, 0, np.int32), dtype=torch.long),
                # "duration": torch.tensor(padding1D(duration, self.series_len, 0, np.int32), dtype=torch.long),
                "delta_t": torch.tensor(padding1D(delta_t, self.series_len, 0, np.int32), dtype=torch.long),
            }
        )

        imei = int(data["imei"])
        longtail = next((x1 for x1 in reversed(hist_list) if x1 >= self.class_num), 0)
        self.one_example.update(
            {
                "hist_series": torch.tensor(padding1D(data["hist_series"], 2000, -1, np.int32), dtype=torch.long),
                "longtail": torch.tensor(longtail, dtype=torch.long),
                "imei": torch.tensor(imei, dtype=torch.long),
            }
        )


class SqliteDataLoader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)
        self.fetchmany = args.fetchmany
        if self.fetchmany:
            self.fetch_size = self.bz.copy()
            self.bz = {"train": 1, "val": 1, "test": 1}
        else:
            self.fetch_size = {"train": None, "val": None, "test": None}

        if args.data_constructor.lower() == "dummy":
            self.data_constructor = DummyData
        elif args.constructor.lower() == "sas":
            self.data_constructor = SasData
        else:
            raise ValueError("[!] 参数 `data_constructor` 传参错误，可选：`dummy`、`sas`")

    @classmethod
    def code(cls):
        return "sqlite"

    def _get_dataset(self, mode):
        if mode == "train":
            dataset = SqliteDataset(
                "train",
                self.args.dataset_path,
                self.args.series_len,
                self.args.class_num,
                self.rng,
                self.fetchmany,
                self.data_constructor,
                self.fetch_size["train"],
            )
        elif mode == "val":
            dataset = SqliteDataset(
                "val",
                self.args.dataset_path,
                self.args.series_len,
                self.args.class_num,
                self.rng,
                self.fetchmany,
                self.data_constructor,
                self.fetch_size["val"],
            )
        elif mode == "test":
            dataset = SqliteDataset(
                "test",
                self.args.dataset_path,
                self.args.series_len,
                self.args.class_num,
                self.rng,
                self.fetchmany,
                self.data_constructor,
                self.fetch_size["test"],
            )
        else:
            raise ValueError
        return dataset


class SqliteDataset(Dataset):
    def __init__(self, mode, dataset_path, series_len, class_num, rng, fetchmany, constructor, fetch_size):
        super().__init__()
        self.constructor = constructor(series_len, class_num, rng)

        self.db_path = os.path.join(dataset_path, "%s.sqlite3" % mode)  # sqlite 文件
        self.table_name = "series_data"
        self.db = SqliteEngine(self.db_path, self.table_name)

        row_num = self.db.size()
        loguru_logger.info("{} 的样本总数为 {}".format(self.db_path, row_num))

        self.mode = mode
        self.fetchmany = fetchmany

        if fetchmany:
            index_list = list(range(1, row_num + 1))
            self.rng.shuffle(index_list)
            index_dict = {}
            for k, i in enumerate(range(0, len(index_list), fetch_size)):
                v = index_list[i : i + fetch_size]
                if len(v) >= fetch_size:
                    index_dict.update({k: v})
            self.example_num = len(index_dict.keys())
            self.index_dict = index_dict.copy()
            del index_dict
            del index_list
        else:
            self.example_num = row_num

    def __len__(self):
        return self.example_num

    def __getitem__(self, index):
        if self.fetchmany:
            examples = []
            raw_data = self.read_batch_sample(self.index_dict[index])
            for one in raw_data:
                self.constructor.create_data(one, self.mode)
                examples.append(self.constructor.one_example)
            self.constructor.one_example = examples
            del examples
        else:
            raw_data = self.read_one_sample(index)
            self.constructor.create_data(raw_data, self.mode)
        return self.constructor.one_example

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"

    def read_one_sample(self, index):
        """读取一条样本，需要 return"""
        unpacked = self.db.search(index)[2]
        unpacked = ast.literal_eval(unpacked)

        return unpacked

    def read_batch_sample(self, batch_index):
        raw_batch = self.db.search(batch_index)
        unpacked = []
        for one in raw_batch:
            unpacked.append(ast.literal_eval(one[3]))
        return unpacked

    def collate_fn(self, batch):
        if self.fetchmany:
            return default_collate(batch[0])
        else:
            return default_collate(batch)


class LmdbDataLoader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)

        if args.data_constructor.lower() == "dummy":
            self.data_constructor = DummyData
        elif args.constructor.lower() == "sas":
            self.data_constructor = SasData
        else:
            raise ValueError("[!] 参数 `data_constructor` 传参错误，可选：`dummy` 、`sas`")

    @classmethod
    def code(cls):
        return "lmdb"

    def _get_dataset(self, mode):
        if mode == "train":
            dataset = LmdbDataset(
                "train",
                self.args.dataset_path,
                self.args.series_len,
                self.args.class_num,
                self.data_constructor,
                self.rng,
            )
        elif mode == "val":
            dataset = LmdbDataset(
                "val",
                self.args.dataset_path,
                self.args.series_len,
                self.args.class_num,
                self.data_constructor,
                self.rng,
            )
        elif mode == "test":
            dataset = LmdbDataset(
                "val",
                self.args.dataset_path,
                self.args.series_len,
                self.args.class_num,
                self.data_constructor,
                self.rng,
            )
        else:
            raise ValueError
        return dataset


class LmdbDataset(Dataset):
    def __init__(self, mode, dataset_path, series_len, class_num, constructor, rng):
        super().__init__()
        self.constructor = constructor(series_len, class_num, rng)
        self.db_path = os.path.join(dataset_path, "%s.lmdb" % mode)  # lmdb文件
        self.db = LmdbEngine(self.db_path)
        self.example_num = self.db.search("__len__")
        self.keys = list(range(0, self.example_num))
        loguru_logger.info("{} 的样本总数为 {}".format(self.db_path, self.example_num))
        self.mode = mode

    def __len__(self):
        return self.example_num

    def __getitem__(self, index):
        raw_data = self.db.search(index)
        self.constructor.create_data(raw_data, self.mode)
        return self.constructor.one_example

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


class TxtDataLoader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)

        if args.data_constructor.lower() == "dummy":
            self.data_constructor = DummyData
        elif args.constructor.lower() == "sas":
            self.data_constructor = SasData
        else:
            raise ValueError("[!] 参数 `data_constructor` 传参错误，可选：`dummy` 、`sas`")

    @classmethod
    def code(cls):
        return "txt"

    def _get_dataset(self, mode):
        if mode == "train":
            dataset = TxtDataset(
                "train",
                self.args.dataset_path,
                self.args.data_keys,
                self.args.from_memory,
                self.args.series_len,
                self.args.class_num,
                self.data_constructor,
                self.rng,
            )
        elif mode == "val":
            dataset = TxtDataset(
                "val",
                self.args.dataset_path,
                self.args.data_keys,
                self.args.from_memory,
                self.args.series_len,
                self.args.class_num,
                self.data_constructor,
                self.rng,
            )
        elif mode == "test":
            dataset = TxtDataset(
                "val",
                self.args.dataset_path,
                self.args.data_keys,
                self.args.from_memory,
                self.args.series_len,
                self.args.class_num,
                self.data_constructor,
                self.rng,
            )
        else:
            raise ValueError
        return dataset


class TxtDataset(Dataset):
    def __init__(self, mode, dataset_path, keys, from_memory, series_len, class_num, constructor, rng):
        super().__init__()
        self.constructor = constructor(series_len, class_num, rng)
        self.file_path = os.path.join(dataset_path, "%s.txt" % mode)
        loguru_logger.info("正在计算 {} 样本数量".format(self.file_path))
        self.db = FileEngine(self.file_path)
        self.example_num = self.db.size()
        loguru_logger.info("{} 的样本总数为 {}".format(self.file_path, self.example_num))
        self.keys = keys
        self.mode = mode
        self.from_memory = from_memory

    def __getitem__(self, index):
        raw_data = self.read_one_sample(index)
        self.constructor.create_data(raw_data, self.mode)
        return self.constructor.one_example

    def __len__(self):
        self.db.reset()
        return self.example_num

    def read_one_sample(self, index):
        data = {}
        values = self.db.get(index)

        for (key, value) in zip(self.keys, values):
            value = value.split(",")
            if value == "" or value is None:
                line = index if self.from_memory else self.example_num
                loguru_logger.error("第 {} 行出现了空值，请检查数据集！".format(line))
                raise ValueError
            data[key] = value[0] if len(value) == 1 else value

        return data
