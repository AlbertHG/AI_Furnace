# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/12/26 15:00
# @Function      : inputs.py

import json
import inspect
import torch
import torch.nn as nn
import numpy as np
from itertools import chain
from collections import OrderedDict, namedtuple, defaultdict
from layers import SequencePoolingLayer, DenseFeatureEncoding

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(
    namedtuple(
        "SparseFeat",
        [
            "column_name",
            "vocabulary_size",
            "embedding_dim",
            "use_hash",
            "dtype",
            "embedding_name",
            "padding_idx",
            "group_name",
            "trainable",
            "kwargs",
        ],
    )
):
    """Define Fixlen Sparse feature

    Args:
        column_name: feature name,
        vocabulary_size: Number of the feature.
        embedding_dim: Embedding dimension of feature.
        dtype: dtype of the feature, default="float32".
        embedding_name: Name of the embedding Matrix (object of nn.Embedding()), default=None, 其名字与name保持一致.
        padding_idx: padding_idx 是索引值，其索引值对应的位置的embed会被填充为 0. Defaults to None.
        group_name: 特征列所属的组.
        trainable: embedding Matrix是否可训练，默认True.
    """

    __slots__ = ()

    def __new__(
        cls,
        column_name,
        vocabulary_size,
        embedding_dim,
        use_hash=False,
        dtype="int32",
        embedding_name=None,
        padding_idx=None,
        group_name=DEFAULT_GROUP_NAME,
        trainable=True,
        **kwargs
    ):
        if embedding_name is None:
            embedding_name = column_name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print("Notice! Feature Hashing on the fly currently!")
        return super(SparseFeat, cls).__new__(
            cls,
            column_name,
            vocabulary_size,
            embedding_dim,
            use_hash,
            dtype,
            embedding_name,
            padding_idx,
            group_name,
            trainable,
            kwargs,
        )

    def __hash__(self):
        return self.column_name.__hash__()


class DenseFeat(namedtuple("DenseFeat", ["column_name", "dimension", "dtype", "group_name", "transform_fn", "kwargs"])):
    """Dense feature (varlen and fixlen Dense feature)

    Args:
        column_name: feature name,
        dimension: dimension of the feature (length of the feature), default = 1.
        dtype: dtype of the feature, default="float32".
        group_name: 特征列所属的组.
        transform_fn: 转换函数，可以是归一化函数，也可以是其它的线性变换函数，以张量作为输入，经函数处理后，返回张量
                        比如： lambda x: (x - mean) / std, 支持 eval() 函数转换
        kwargs: 其他一些附属参数，比如 mean std etc.
    """

    __slots__ = ()

    def __new__(
        cls, column_name, dimension, dtype="float32", group_name=DEFAULT_GROUP_NAME, transform_fn=None, **kwargs
    ):
        transform_fn = eval(transform_fn) if isinstance(transform_fn, str) else transform_fn
        return super(DenseFeat, cls).__new__(cls, column_name, dimension, dtype, group_name, transform_fn, kwargs)

    def __hash__(self):
        return self.column_name.__hash__()


class EmbDenseFeat(
    namedtuple(
        "EmbDenseFeat",
        ["densefeat", "embedding_dim", "embedding_name", "padding_value", "combiner", "trainable", "length_name"],
    )
):
    """Embedding Dense feature (varlen and fixlen Dense feature)

    Args:
        densefeat: DenseFeat 实例化对象.
        embedding_dim: Embedding dimension of feature.
        embedding_name: name of the emiedding, default=None.
        padding_value: 是填充值，其填充值对应的 embed 会被填充为 0. Defaults to None.
    """

    __slots__ = ()

    def __new__(
        cls,
        densefeat,
        embedding_dim,
        embedding_name=None,
        padding_value=None,
        combiner="mean",
        trainable=True,
        length_name=None,
    ):
        if embedding_name is None:
            embedding_name = densefeat.column_name
        return super(EmbDenseFeat, cls).__new__(
            cls, densefeat, embedding_dim, embedding_name, padding_value, combiner, trainable, length_name
        )

    @property
    def column_name(self):
        return self.densefeat.column_name

    @property
    def dimension(self):
        return self.densefeat.dimension

    @property
    def dtype(self):
        return self.densefeat.dtype

    @property
    def transform_fn(self):
        return self.densefeat.transform_fn

    @property
    def kwargs(self):
        return self.densefeat.kwargs

    @property
    def group_name(self):
        return self.densefeat.group_name

    def __hash__(self):
        return self.column_name.__hash__()


class VarLenSparseFeat(namedtuple("VarLenSparseFeat", ["sparsefeat", "dimension", "combiner", "length_name"])):

    """varlen Sparse Feature，比如序列特征


    Args:
        sparsefeat: SparseFeat 实例化对象.
        dimension: dimension of the feature (length of the feature).
        combiner: 池化方法（mean,sum,max），默认是mean.
        length_name：行为序列的实际长度，如果是None的话，表示特征中的0是用来填充的.
    """

    __slots__ = ()

    def __new__(cls, sparsefeat, dimension, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, dimension, combiner, length_name)

    @property
    def column_name(self):
        return self.sparsefeat.column_name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def padding_idx(self):
        return self.sparsefeat.padding_idx

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def kwargs(self):
        return self.sparsefeat.kwargs

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.column_name.__hash__()


class LinearLogits(nn.Module):
    """获取 linear_logit（线性变换）的结果

        Embedding 类特征 embedding_dim == 1；
        value 类型特征 加入一个权重 w * x；
        最后，特征相加求和输出标量。

    Args:
        nn ([type]): [description]
        feature_columns (list): 参与计算的 feature column
        feature_index (list): feature index
    """

    def __init__(self, feature_columns, feature_index):
        super().__init__()
        self.feature_index = feature_index
        self.sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        )
        self.varlen_sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
        )
        self.dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        )
        self.emb_dense_feature_columns = (
            list(filter(lambda x: isinstance(x, EmbDenseFeat), feature_columns)) if len(feature_columns) else []
        )

        self.embedding_dict = create_embedding_matrix(
            feature_columns, linear=True, include_sparse=True, include_dense=True
        )

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1))
            torch.nn.init.normal_(self.weight, mean=0, std=0.0001)

    def forward(self, X, sparse_feat_refine_weight=None):
        value_list = [
            X[:, self.feature_index[feat.column_name][0] : self.feature_index[feat.column_name][1]]
            for feat in self.dense_feature_columns
        ]
        embedding_list = [
            self.embedding_dict[feat.embedding_name](
                X[:, self.feature_index[feat.column_name][0] : self.feature_index[feat.column_name][1]].long()
            )
            for feat in self.sparse_feature_columns
        ]

        varlen_feature_columns = self.emb_dense_feature_columns + self.varlen_sparse_feature_columns
        varlen_embed_dict = embedding_lookup(X, self.embedding_dict, self.feature_index, varlen_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(
            varlen_embed_dict, X, self.feature_index, self.emb_dense_feature_columns
        )

        embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(X.device)
        if len(embedding_list) > 0:
            sparse_embedding_cat = torch.cat(embedding_list, dim=-1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(value_list) > 0:
            dense_embedding_logit = torch.cat(value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_embedding_logit

        return linear_logit


class CreateFeatureColumns(object):
    """特征解析类，根据特征描述文件(JSON)完成解析。

    方法只接受四种特征类型:
        "dense_feature_columns"
        "emb_dense_feature_columns"
        "sparse_feature_columns"
        "val_sparse_feature_columns"
    """

    def __init__(self, json_path):
        with open(json_path) as fn:
            self.features_file = json.load(fn)
        self.feats_types = [
            "dense_feature_columns",
            "emb_dense_feature_columns",
            "sparse_feature_columns",
            "val_sparse_feature_columns",
        ]
        if not set(self.features_file.keys()) <= set(self.feats_types):  # 判断是否为子集
            raise ValueError("[!] JSON 描述文件含有不能被解析的特征类型, ", self.feats_types)

        self.feature_columns = []
        self.dense_feature_columns = []
        self.emb_dense_feature_columns = []
        self.sparse_feature_columns = []
        self.val_sparse_feature_columns = []
        self.__bulid__()

    def __bulid__(self):
        """feature columns object 解析函数"""
        # column_name 唯一性检测
        self.__global_unique_check__()

        for feats_type, feats in self.features_file.items():
            if not feats:  # feats_type 空内容过滤
                continue

            if feats_type == "dense_feature_columns":
                args_list = self.__combine_args__(feats_type, feats)
                self.dense_feature_columns += [DenseFeat(**_args) for _args in args_list]

            elif feats_type == "sparse_feature_columns":
                args_list = self.__combine_args__(feats_type, feats)
                self.sparse_feature_columns += [SparseFeat(**_args) for _args in args_list]

            elif feats_type == "emb_dense_feature_columns":
                outer_list, inner_list = self.__combine_args__(feats_type, feats, clip=True)
                self.emb_dense_feature_columns += [
                    EmbDenseFeat(DenseFeat(**inner), **outer) for inner, outer in zip(inner_list, outer_list)
                ]

            elif feats_type == "val_sparse_feature_columns":
                outer_list, inner_list = self.__combine_args__(feats_type, feats, clip=True)
                self.val_sparse_feature_columns += [
                    VarLenSparseFeat(SparseFeat(**inner), **outer) for inner, outer in zip(inner_list, outer_list)
                ]

        self.feature_columns = (
            self.emb_dense_feature_columns
            + self.dense_feature_columns
            + self.sparse_feature_columns
            + self.val_sparse_feature_columns
        )

    def __global_unique_check__(self):
        all_columns_name = list(
            chain.from_iterable([feats["column_name"] if feats else [] for feats in self.features_file.values()])
        )
        if len(set(all_columns_name)) != len(all_columns_name):
            raise ValueError("[!] 特征列中，列名有重复值!")

    def __combine_args__(self, feats_type, feats, clip=False):
        """将参数进行重组合

        raw collection : {"K1":[1,2,3], "K2":[4,5,6]}

        ------> transform ------->

        new collection : [{"K1":1, "K2":4},
                          {"K1":2, "K2":5},
                          {"K1":3, "K2":6}]
        """
        _class_ = {
            "dense_feature_columns": DenseFeat,
            "emb_dense_feature_columns": EmbDenseFeat,
            "sparse_feature_columns": SparseFeat,
            "val_sparse_feature_columns": VarLenSparseFeat,
        }

        def __transform_fn__(d):
            args_list = []
            for values in zip(*(d.values())):
                _dict_ = {}
                for k, v in zip(d.keys(), values):
                    _dict_[k] = v
                args_list.append(_dict_)
            return args_list

        args_dict = OrderedDict()
        column_num = len(feats["column_name"])
        for attr, value in feats.items():
            args_dict.update({attr: self.__broadcast__(value, column_num, "{}: {}".format(feats_type, attr))})

        if clip:
            outer_args = inspect.getargspec(_class_[feats_type].__new__).args[2:]
            outer_dict, inner_dict = self.__clip_dict__(args_dict, lambda k, v: k in outer_args)
            return __transform_fn__(outer_dict), __transform_fn__(inner_dict)
        else:
            return __transform_fn__(args_dict)

    def __get_attribute__(self, feat, key):
        try:
            return feat[key]
        except:
            return None

    def __broadcast__(self, msg, length, name):
        if isinstance(msg, int) or isinstance(msg, str) or msg is None:
            msg = [msg] * length
        elif isinstance(msg, list) and len(msg) == 1:
            msg = msg * length
        elif isinstance(msg, list) and len(msg) == length:
            msg = msg
        else:
            raise ValueError("[!] MSG broadcast Error, check {}.".format(name))
        return msg

    def __clip_dict__(self, d, condition):
        """
        Partition a dictionary based on some condition function
        :param d: a dict
        :param condition: a function with parameters k, v returning a bool for k: v in d
        :return: two dictionaries, with the contents of d, split according to condition
        """
        return {k: v for k, v in d.items() if condition(k, v)}, {k: v for k, v in d.items() if not condition(k, v)}

    def __build_input_features_index__(self, feature_columns):
        """构建 feature column 对应的 batch data 中位置的索引
        Return OrderedDict: {feature_name:(start, start+dimension)}
                            feature_name order : [SparseFeat..., DenseFeat..., VarLenSparseFeat..., Other]
        """
        features = OrderedDict()

        start = 0
        for feat in feature_columns:
            feat_name = feat.column_name
            if feat_name in features:
                continue
            if isinstance(feat, SparseFeat):
                features[feat_name] = (start, start + 1)
                start += 1
            elif isinstance(feat, DenseFeat):
                features[feat_name] = (start, start + feat.dimension)
                start += feat.dimension
            elif isinstance(feat, (EmbDenseFeat, VarLenSparseFeat)):
                features[feat_name] = (start, start + feat.dimension)
                start += feat.dimension
                if feat.length_name is not None and feat.length_name not in features:
                    features[feat.length_name] = (start, start + 1)
                    start += 1
            else:
                raise TypeError("Invalid feature column type, got", type(feat))
        return features

    def __get_feature_names__(self, feature_columns):
        """获取特征列的列名"""
        features = self.__build_input_features_index__(feature_columns)
        return list(features.keys())

    def get_feature_columns(self):
        return self.feature_columns

    def get_dense_feature_columns(self):
        return self.dense_feature_columns

    def get_emb_dense_feature_columns(self):
        return self.emb_dense_feature_columns

    def get_sparse_feature_columns(self):
        return self.sparse_feature_columns

    def get_val_sparse_feature_columns(self):
        return self.val_sparse_feature_columns

    def get_feature_name(self):
        return self.__get_feature_names__(self.feature_columns)

    def get_dense_feature_names(self):
        return self.__get_feature_names__(self.dense_feature_columns)

    def get_emb_dense_feature_names(self):
        return self.__get_feature_names__(self.emb_dense_feature_columns)

    def get_sparse_feature_names(self):
        return self.__get_feature_names__(self.sparse_feature_columns)

    def get_val_sparse_feature_names(self):
        return self.__get_feature_names__(self.val_sparse_feature_columns)

    def get_feature_index(self):
        return self.__build_input_features_index__(self.feature_columns)

    def find_feature_columns(self, feature_name_list: list, negation=False):
        if isinstance(feature_name_list, str):
            feature_name_list = [feature_name_list]
        if negation:
            feature_columns = (
                list(filter(lambda x: x.column_name not in feature_name_list, self.feature_columns))
                if len(self.feature_columns)
                else []
            )
        else:
            feature_columns = (
                list(filter(lambda x: x.column_name in feature_name_list, self.feature_columns))
                if len(self.feature_columns)
                else []
            )
        return feature_columns


def compute_input_dim(feature_columns, include_sparse=True, include_dense=True, varlen_feature_pooling=True):
    """计算 DNN 的 input dim"""
    sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    )

    varlen_sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
    )
    dense_feature_columns = (
        list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
    )
    emb_dense_feature_columns = (
        list(filter(lambda x: isinstance(x, EmbDenseFeat), feature_columns)) if len(feature_columns) else []
    )

    dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))  # dense value
    sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)  # sparse emb

    # val len dense emb and val len sparse emb
    if varlen_feature_pooling:
        sparse_input_dim += sum(feat.embedding_dim for feat in varlen_sparse_feature_columns)
        dense_input_dim += sum(feat.embedding_dim for feat in emb_dense_feature_columns)
    else:
        sparse_input_dim += sum(feat.embedding_dim * feat.dimension for feat in varlen_sparse_feature_columns)
        dense_input_dim += sum(feat.embedding_dim * feat.dimension for feat in emb_dense_feature_columns)

    input_dim = 0
    if include_sparse:
        input_dim += sparse_input_dim
    if include_dense:
        input_dim += dense_input_dim
    return input_dim


def create_embedding_matrix(feature_columns, linear=False, include_sparse=True, include_dense=True):
    """Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}"""
    sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    )
    varlen_sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
    )
    emb_dense_feature_columns = (
        list(filter(lambda x: isinstance(x, EmbDenseFeat), feature_columns)) if len(feature_columns) else []
    )
    embedding_dict = nn.ModuleDict()

    if include_sparse:
        embedding_dict.update(
            {
                feat.embedding_name: nn.Embedding(
                    feat.vocabulary_size, feat.embedding_dim if not linear else 1, padding_idx=feat.padding_idx
                )
                for feat in sparse_feature_columns + varlen_sparse_feature_columns
            }
        )

        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.0001)
    if include_dense:
        embedding_dict.update(
            {
                feat.embedding_name: DenseFeatureEncoding(
                    feat.embedding_dim if not linear else 1, padding_value=feat.padding_value
                )
                for feat in emb_dense_feature_columns
            }
        )

    return embedding_dict


def input_from_sparse_feature_columns(X, sparse_feature_columns, feature_index, embedding_dict):
    sparse_embedding_list = [
        embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.column_name][0] : feature_index[feat.column_name][1]].long()
        )
        for feat in sparse_feature_columns
    ]  # sparse embed
    return sparse_embedding_list


def input_from_dense_feature_columns(X, dense_feature_columns, feature_index):
    dense_value_list = [
        X[:, feature_index[feat.column_name][0] : feature_index[feat.column_name][1]] for feat in dense_feature_columns
    ]  # dense value
    return dense_value_list


def input_from_varlen_sparse_feature_columns(X, varlen_sparse_feature_columns, feature_index, embedding_dict):
    varlen_embed_dict = embedding_lookup(X, embedding_dict, feature_index, varlen_sparse_feature_columns)
    varlen_embedding_list = get_varlen_pooling_list(varlen_embed_dict, X, feature_index, varlen_sparse_feature_columns)
    return varlen_embedding_list


def input_from_feature_columns(X, feature_columns, feature_index, embedding_dict, support_dense=True):
    """return model input"""
    sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    )
    varlen_sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
    )
    dense_feature_columns = (
        list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
    )
    emb_dense_feature_columns = (
        list(filter(lambda x: isinstance(x, EmbDenseFeat), feature_columns)) if len(feature_columns) else []
    )

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError("DenseFeat is not supported in this model !")

    dense_value_list = [
        X[:, feature_index[feat.column_name][0] : feature_index[feat.column_name][1]] for feat in dense_feature_columns
    ]  # dense value
    sparse_embedding_list = [
        embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.column_name][0] : feature_index[feat.column_name][1]].long()
        )
        for feat in sparse_feature_columns
    ]  # sparse embed

    varlen_feature_columns = emb_dense_feature_columns + varlen_sparse_feature_columns
    varlen_embed_dict = embedding_lookup(X, embedding_dict, feature_index, varlen_feature_columns)
    varlen_embedding_list = get_varlen_pooling_list(varlen_embed_dict, X, feature_index, varlen_feature_columns)

    return (sparse_embedding_list + varlen_embedding_list, dense_value_list)


def embedding_lookup(X, embedding_dict, feature_index, feature_columns):
    """取得 data Embedding tensor   {feature_name: embedding tensor}

    Args:
        X ([type]): [description]
        embedding_dict ([type]): [description]
        feature_index ([type]): [description]
        varlen_sparse_feature_columns ([type]): [description]

    Returns:
        (dict): {feature_name: embedding tensor}
    """
    embedding_vec_dict = {}
    for feat in feature_columns:
        feature_name = feat.column_name
        embedding_name = feat.embedding_name
        lookup_idx = feature_index[feature_name]
        x = X[:, lookup_idx[0] : lookup_idx[1]]
        if isinstance(feat, (VarLenSparseFeat, SparseFeat)):
            x = x.long()
        elif isinstance(feat, (DenseFeat, EmbDenseFeat)):
            x = x.float()
        else:
            raise ValueError
        embedding_vec_dict[feature_name] = embedding_dict[embedding_name](x)

    return embedding_vec_dict


def get_varlen_pooling_list(varlen_embed_tensor_dict, X, feature_index, varlen_feature_columns):
    """对 vallen data embedding tensor 进行 pooling"""
    varlen_sparse_embedding_list = []
    for feat in varlen_feature_columns:
        seq_emb = varlen_embed_tensor_dict[feat.column_name]

        if feat.dimension == 1:  # val len 的长度 == 1，embed tensor 不用 mask 处理.
            emb = seq_emb
        # 以下情况为 dimension > 1 ，根据给定 val len data 的真实长度，构建 mask 矩阵
        elif feat.length_name is None:
            # TODO: !=0 这个位置是否可以改为 不等于 padding_value or padding_idx
            seq_mask = X[:, feature_index[feat.column_name][0] : feature_index[feat.column_name][1]].long() != 0
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True)([seq_emb, seq_mask])
        else:
            # csv 专门一列特征列用来标定真实长度
            seq_length = X[
                :,
                feature_index[feat.length_name][0] : feature_index[feat.length_name][1],
            ].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False)([seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


def get_varlen_group_and_mask(
    varlen_embed_tensor_dict, X, feature_index, varlen_feature_columns, group_by_group_name=True
):
    """val len embedding tensor and its mask matrix"""
    group_embedding_dict = defaultdict(list)
    mask_dict = defaultdict(list)
    for feat in varlen_feature_columns:
        seq_emb = varlen_embed_tensor_dict[feat.column_name]
        group_embedding_dict[feat.group_name].append(seq_emb)

        # 构建 mask 矩阵
        if feat.length_name is None:
            # [B, max_len]
            mask = X[:, feature_index[feat.column_name][0] : feature_index[feat.column_name][1]].long() != 0
        else:
            seq_length = X[
                :,
                feature_index[feat.length_name][0] : feature_index[feat.length_name][1],
            ].long()
            max_len = seq_emb.shape[1]
            row_vector = torch.arange(0, max_len, 1)
            matrix = torch.unsqueeze(seq_length, dim=-1)
            mask = row_vector < matrix
            mask.type(torch.bool)  # [B, 1, dimension]
            mask = mask.squeeze(1)  # [B, dimension]
        # mask = torch.repeat_interleave(mask, feat.embedding_dim, dim=2)  # [B, dimension, E]
        mask_dict.update({feat.group_name: ~mask})  # transformer padding mask 1 指代 mask 0 指代真值

    if group_by_group_name:
        try:
            for k, v in group_embedding_dict.items():
                group_embedding_dict.update({k: torch.sum(torch.stack(v), 0)})
        except:
            raise ValueError("[!] 同一 group_name 的 sequence val feature [dimension] 长度不一致")

    return group_embedding_dict, mask_dict


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    """将 sparse_embedding_list 和 dense_value_list 两类数据， flatten 并 cat 成 dnn 输入"""
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)
