import os
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from .utils import *
from dataloaders import SplitCSVDataLoader
from inputs import CreateFeatureColumns


class CriteoDataLoader(SplitCSVDataLoader):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def code(cls):
        return "criteo_loader"

    def _get_dataset(self, mode):
        if mode == "train":
            dataset = CriteoDataSet(self.args, self.train_data)
        elif mode == "val":
            dataset = CriteoDataSet(self.args, self.test_data)
        elif mode == "test":
            dataset = CriteoDataSet(self.args, self.test_data)
        else:
            raise ValueError
        return dataset


class CriteoDataSet(Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.task_label = args.task_label
        assert isinstance(self.task_label, (str, list))

        # feature index define
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.feature_names = feature_fun.get_feature_name()
        self.sparse_names = feature_fun.get_sparse_feature_names() + feature_fun.get_val_sparse_feature_names()
        self.dense_names = feature_fun.get_emb_dense_feature_names() + feature_fun.get_dense_feature_names()

        # self.data = self.data[self.feature_names + ["label"]]  # 重新排列 列 按照 feature_names 的顺序
        self.data[self.sparse_names] = self.data[self.sparse_names].fillna("-1")
        self.data[self.dense_names] = self.data[self.dense_names].fillna(0)

        # preprocessing
        for feat in self.sparse_names:
            lbe = LabelEncoder()
            self.data[feat] = lbe.fit_transform(self.data[feat])
        mms = MinMaxScaler()
        self.data[self.dense_names] = mms.fit_transform(self.data[self.dense_names])
        self.one_example = {}

    def __getitem__(self, index):
        self.one_example = {}
        one_line = self.data.iloc[index]
        x = [one_line[fn] for fn in self.feature_names]

        self.one_example.update({"x": torch.tensor(x)})

        self.task_label = [self.task_label] if isinstance(self.task_label, str) else self.task_label
        self.one_example.update({k: torch.tensor(one_line[[k]]) for k in self.task_label})
        return self.one_example

    def _collate_fn(self, batch):
        pass

    def __len__(self):
        return self.data.shape[0]
