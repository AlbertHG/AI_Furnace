# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : dataloader 和 dataset 工具类

import os
import lmdb
import sqlite3
import pickle
import linecache
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CsvEngine(object):
    def __init__(self, db_path, fetchmany, fetch_size):
        self.db_path = db_path
        self.fetchmany = fetchmany
        self.fetch_size = fetch_size
        self.example_num = None
        self.open()

    def size(self):
        return self.example_num if self.example_num is not None else wc_count(self.db_path) - 1

    def open(self):
        if self.fetchmany:
            self.example_num = self.example_num // self.fetch_size
            self.sum_counter = 0  # 计数器
            self.file = open(self.db_path, encoding="utf-8")
            self.data = pd.read_csv(self.file, iterator=True)
        else:
            self.example_num = self.example_num
            self.data = pd.read_csv(self.db_path)
        return self

    def get(self, index):
        if self.fetchmany:
            if self.sum_counter == self.example_num:
                self.file.close()
                raise StopIteration
            self.sum_counter += 1
            values = self.data.get_chunk(self.fetch_size)
        else:
            values = self.data.iloc[index].copy().to_dict()
        return values

    def reset(self):
        if self.fetchmany and self.file.closed:
            self.open()


class FileEngine(object):
    def __init__(self, db_path, from_memory):
        self.db_path = db_path
        self.from_memory = from_memory
        self.example_num = None
        self.open()

    def size(self):
        return self.example_num if self.example_num is not None else wc_count(self.db_path)

    def open(self):
        self.size()
        if not self.from_memory:
            self.sum_counter = 0  # 计数器
            self.dataset_file = open(self.db_path)
        else:
            self.dataset_file = linecache.getlines(self.db_path)
        return self

    def get(self, index):
        if not self.from_memory:
            self.sum_counter += 1
            if self.sum_counter == self.example_num:
                self.dataset_file.close()
                raise StopIteration
            # index 实际上没有真正用上
            values = self.dataset_file.readline()
        else:
            values = self.dataset_file[index]
        return values

    def reset(self):
        if not self.from_memory and self.dataset_file.closed:
            self.open()


class LmdbEngine(object):
    """Lmdb格式数据库读写引擎"""

    def __init__(self, db_path):
        """初始化方法，接受一个数据库路径作为参数

        Args:
            db_path (str): 数据库路径
        """
        self.db_path = db_path
        self.isdir = os.path.isdir(db_path)

        self.env = None
        self.init_env()

    def search(self, sid):
        """搜索方法，接受一个sid作为参数，返回对应的值

        Args:
            sid (str or bytes): 要搜索的键

        Returns:
            object: 键对应的值，反序列化后的对象
        """
        b_sid = str(sid).encode("ascii") if not isinstance(sid, bytes) else sid
        with self.env.begin(write=False) as txn:
            value = txn.get(b_sid)
        return pickle.loads(value)

    def insert(self, sid, value):
        """插入方法，接受一个sid和一个value作为参数，将它们存入数据库

        Args:
            sid (str or bytes): 要插入的键
            value (object): 要插入的值，任意可序列化的对象
        """
        txn = self.env.begin(write=True)
        txn.put(str(sid).encode("ascii"), pickle.dumps(value))
        txn.commit()

    def insert_many(self, sid, value):
        """批量插入方法，接受两个列表作为参数，分别是sid列表和value列表，要求两个列表长度相等，将它们一一对应地存入数据库

        Args:
            sid (list of str or bytes): 要插入的键列表
            value (list of object): 要插入的值列表，每个元素都是可序列化的对象

        Raises:
            AssertionError: 如果两个列表长度不相等
        """
        assert isinstance(sid, list) and isinstance(value, list), len(sid) == len(value)
        txn = self.env.begin(write=True)
        for k, v in zip(sid, value):
            txn.put(str(k).encode("ascii"), pickle.dumps(v))
        txn.commit()

    def size(self):
        """大小方法，返回数据库中的条目数

        Returns:
            int: 数据库中的条目数
        """
        with self.env.begin(write=False) as txn:
            return txn.stat()["entries"]

    def reset(self):
        """重置方法，关闭并重新打开环境变量"""
        self.close_env()
        self.init_env()

    def init_env(self):
        """初始化环境变量方法，如果环境变量为None，则创建一个新的lmdb环境并设置相关参数

        Returns:
            LmdbEngine: 返回自身对象
        """
        if self.env is None:
            self.env = lmdb.open(
                self.db_path,
                subdir=self.isdir,
                map_size=1099511627776 / 2,  # 1T/m
                readonly=False,
                meminit=False,
                lock=False,
                # map_async=False,
            )
        return self

    def close_env(self):
        """关闭环境变量方法，如果环境变量不为None，则同步并关闭环境变量，并将其设为None"""
        if self.env is not None:
            self.env.sync()
            self.env.close()

            del self.env
            self.env = None
        return self

    def __enter__(self):
        """进入方法，用于支持with语句，返回自身对象

        Returns:
            LmdbEngine: 返回自身对象
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出方法，用于支持with语句，同步并关闭环境变量"""
        self.env.sync()
        self.env.close()


class SqliteEngine(object):
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.conn = None
        self.establish_conn()
        self.table_name = table_name

    def execute(self, sql, value=None):
        try:
            if value:
                self.cursor.execute(sql, value)
            else:
                self.cursor.execute(sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception("execute failed: {}".format(str(e)))

    def execute_many(self, sql, data_list):
        # 一次性插入多条样本，相比于 execute(), 可以极快提高插入速度
        # example:
        # sql = 'insert into filelist (pkgKey, dirname, filenames, filetypes) values (?, ?, ?, ?);'
        # data_list = [(1, '/etc/sysconfig', 'openshift_option', 'f'), (1, '/usr/share/doc', 'adb-utils-1.6', 'd')]
        try:
            self.cursor.executemany(sql, data_list)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception("execute failed: {}".format(str(e)))

    def search(self, index):
        search_sql = f"select * from {self.table_name} where rowid=?"
        self.execute(search_sql, (index + 1,))  # rowid 从 1 开始计数
        unpacked = self.cursor.fetchone()
        return unpacked

    def search_many(self, indexs):
        assert isinstance(indexs, list)
        search_sql = "select * from {} where rowid IN ({})".format(
            self.table_name, ",".join(map(lambda x: str(x), indexs))
        )
        self.execute(search_sql)
        unpacked = self.cursor.fetchall()
        assert len(unpacked) == len(indexs)
        return unpacked

    def size(self):
        self.execute(f"select max(rowid) from {self.table_name}")
        row_num = self.cursor.fetchall()[0][0]
        return row_num

    def establish_conn(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, cached_statements=1024)
            self.cursor = self.conn.cursor()
        return self

    def close_conn(self):
        if self.conn is not None:
            self.cursor.close()
            self.conn.close()

            del self.conn
            self.conn = None
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        self.conn.commit()
        self.conn.close()


def random_neg_from_retention(s, _list, rng):
    """
    从给定的列表 _list 中随机取 不等于 s 的值
    Args:
        s (int): 例外
        _list (list): 取值列表
        rng (object): Random包实例化对象

    Returns:
        [int]: [description]
    """
    # neg_item = random.randint(0, self.item_num-1)
    neg_item = rng.choice(_list)
    while neg_item == s:
        neg_item = rng.choice(_list)
    return neg_item


def random_neg(s, item_num, rng):
    """
    从区间中随机取 不等于 s 的值

    Args:
        s (int): 例外
        item_num (int): 取值区间 [0, item_num)
        rng (object): Random包实例化对象

    Returns:
        [int]: [description]
    """
    neg_item = rng.randint(0, item_num - 1)
    while neg_item == s:
        neg_item = rng.randint(0, item_num - 1)
    return neg_item


def random_bbox(img_shape, margin, bbox_shape):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    img_height = img_shape[0]
    img_width = img_shape[1]
    height = bbox_shape[0]
    width = bbox_shape[1]
    margin_u = margin[0]
    margin_d = margin[1]
    margin_l = margin[2]
    margin_r = margin[3]

    maxt = img_height - margin_u - height
    maxl = img_width - margin_r - width
    t = np.random.randint(low=margin_d, high=maxt)
    l = np.random.randint(low=margin_l, high=maxl)
    h = height
    w = width
    return (t, l, h, w)


def bbox2mask(img_shape, margin, bbox_shape, times):
    """Generate mask tensor from bbox.
    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    bboxs = []
    for _ in range(times):
        bbox = random_bbox(img_shape, margin, bbox_shape)
        bboxs.append(bbox)
    height = img_shape[0]
    width = img_shape[1]
    mask = np.zeros((height, width), np.float32)
    for bbox in bboxs:
        h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
        w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
        mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.0
    return mask.astype(np.float32)


def padding1D(data, need_len, v, _type):
    """
    根据指定长度 切割 或者 补齐 1维 List
    Args:
        data ([type]): 待处理 List
        need_len ([type]): 指定长度
        v ([type]): 填充数据
        _type ([type]): numpy type

    Returns:
        [type]: [description]
    """
    data_len = len(data)
    data_ = np.full([need_len], v, dtype=_type)
    if data_len <= need_len:  # padding序列
        data_[need_len - data_len :] = data
    else:  # 截断序列
        data_ = data[data_len - need_len :]
    return data_


def padding2D(data, need_len, v, _type):
    """
    根据指定长度 切割 或者 补齐 2维 List
    Args:
        data ([type]): 待处理 List
        need_len ([type]): 指定长度
        v ([type]): 填充数据
        _type ([type]): numpy type

    Returns:
        [type]: [description]
    """
    data_len = len(data)
    data_ = np.full([need_len, need_len], v, dtype=_type)
    if data_len <= need_len:  # padding序列
        data_[need_len - data_len :, need_len - data_len :] = data[:, :]
    else:  # 截断序列
        data_[:, :] = data[data_len - need_len :, data_len - need_len :]
    return data_


def wc_count(file_name):
    """调用使用 wc 命令计算 文件行数行

    Args:
        file_name (str): 文件路径

    """
    import subprocess

    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])
