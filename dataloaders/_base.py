# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : dataloader 基类

import os
import random
import linecache
import pandas as pd
import numpy as np

from abc import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from loguru import logger as loguru_logger
from sklearn.model_selection import train_test_split

from .utils import *


class AbstractDataloader(metaclass=ABCMeta):
    """Dataloader 抽象类

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, args):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)

        self.bz = {
            "train": self.args.train_batch_size,
            "val": self.args.val_batch_size,
            "test": self.args.test_batch_size,
        }

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def get_dataloaders(self):
        train_loader = self._loader("train")
        val_loader = self._loader("val")
        test_loader = self._loader("test")
        return train_loader, val_loader, test_loader

    def get_val_loader(self):
        return self._loader("val")

    def get_train_loader(self):
        return self._loader("train")

    def get_test_loader(self):
        return self._loader("test")

    def _loader(self, mode):
        # mode = train ,val, test
        batch_size = self.bz[mode]
        dataset = self._get_dataset(mode)
        shuffle = True if mode == "train" else False
        try:
            num_workers = self.args.num_workers
            dl = DataLoaderX if num_workers > 0 else DataLoader
        except:
            num_workers = 0
            dl = DataLoader

        invert_op = getattr(dataset, "collate_fn", None)
        if callable(invert_op):
            dataloader = dl(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=num_workers,
                collate_fn=dataset.collate_fn,
            )
        else:
            dataloader = dl(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=num_workers,
            )
        return dataloader

    @abstractmethod
    def _get_dataset(self, mode):
        pass


class SplitCSVDataLoader(AbstractDataloader):
    """附带切分 CSV 功能的dataloader

    Args:
        AbstractDataloader ([type]): [description]
    """

    def __init__(self, args):
        super().__init__(args)
        loguru_logger.info("正在读取和切分数据集, Train Val Ratio [{} : {}]".format(1 - args.test_size, args.test_size))
        self.data = pd.read_csv(args.dataset_path)
        self.train_data, self.test_data = train_test_split(
            self.data,
            test_size=args.test_size,
            stratify=self.data["label"],
            random_state=args.dataloader_random_seed,
        )
        loguru_logger.info("Train Set Size : {}".format(self.train_data.shape[0]))
        loguru_logger.info("Val Set Size   : {}".format(self.test_data.shape[0]))

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def _get_dataset(self, mode):
        pass


class CSVChunkDataLoader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)

    def _loader(self, mode):
        # mode = train ,val, test
        dataset = self._get_dataset(mode)
        shuffle = True if mode == "train" else False
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def collate_fn(self, batch):
        batch = default_collate(batch)
        batch = {k: v.squeeze(0) for k, v in batch.items()}
        return batch
