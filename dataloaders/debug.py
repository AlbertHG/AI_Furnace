import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from itertools import chain
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger as loguru_logger
from abc import *

from .utils import *
from dataloaders import AbstractDataloader
from inputs import DenseFeat, SparseFeat, CreateFeatureColumns


class TestDataLoader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def code(cls):
        return "test"

    def _get_dataset(self, mode):
        if mode == "train":
            dataset = DebugDataset(self.args, self.args.dataset_path)
        elif mode == "val":
            dataset = DebugDataset(self.args, self.args.dataset_path)
        elif mode == "test":
            dataset = DebugDataset(self.args, self.args.dataset_path)
        else:
            raise ValueError
        return dataset


class DebugDataset(Dataset):
    def __init__(self, args, dataset_path=None):
        self.args = args
        # feature index define
        feature_fun = CreateFeatureColumns(args.feature_msg)
        self.feature_names = feature_fun.get_feature_name()
        self.input_feature_columns = feature_fun.get_feature_columns()
        self.sparse_names = feature_fun.get_sparse_feature_names() + feature_fun.get_val_sparse_feature_names()
        self.dense_names = feature_fun.get_emb_dense_feature_names() + feature_fun.get_dense_feature_names()

        loguru_logger.info("正在读取数据集...")
        self.data = pd.read_csv(dataset_path)

        # preprocessing
        self.data.D3 = self.data.D3.apply(lambda x: [float(i) for i in x.split(",")])
        self.data.VED1 = self.data.VED1.apply(lambda x: [float(i) for i in x.split(",")])
        self.data.VED2 = self.data.VED2.apply(lambda x: [int(i) for i in x.split(",")])
        self.data.VED3 = self.data.VED3.apply(lambda x: [int(i) for i in x.split(",")])
        self.data.VS1 = self.data.VS1.apply(lambda x: [int(i) for i in x.split(",")])
        self.data.VS2 = self.data.VS2.apply(lambda x: [int(i) for i in x.split(",")])
        for feat in feature_fun.get_sparse_feature_names():
            lbe = LabelEncoder()
            self.data[feat] = lbe.fit_transform(self.data[feat])

        self.data = self.data[self.feature_names + ["label"]]  # 重新排列 列 按照 feature_names 的顺序
        self.target = ["label"]

        self.one_example = {}

    def __getitem__(self, index):
        self.one_example = {}
        one_line = self.data.iloc[index]
        x = []
        for fn in self.feature_names:
            x += one_line[fn] if isinstance(one_line[fn], list) else [one_line[fn]]

        self.one_example.update({"x": torch.tensor(x)})
        self.one_example.update({"label": torch.tensor(one_line[self.target])})
        return self.one_example

    def _collate_fn(self, batch):
        pass

    def __len__(self):
        return self.data.shape[0]
