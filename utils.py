# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : 工程框架工具类

import json
import os
import datetime
import random
import inspect
import sys
import pkgutil
import pprint as pp
import numpy as np
import config

import torch
import torch.backends.cudnn as cudnn

from loguru import logger as loguru_logger
from datetime import date
from pathlib import Path
from importlib import import_module


def count_param(model):
    """计算模型参数量

    Args:
        model ([type]): 模型的实例化对象
    Returns:
        [type]: 返回参数量
    """
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    # print('totoal parameters: %.2fMb  (%d)'%(param/1e6,param))
    return param_count


def fix_random_seed_as(random_seed=0):
    """
    设置模型随机种子数

    Args:
        random_seed (int, optional): . Defaults to 0.
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)  # CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(random_seed)  # GPU设置种子用于生成随机数，以使得结果是确定的
    np.random.seed(random_seed)

    # https://zhuanlan.zhihu.com/p/73711222
    # 针对卷积的优化，benchmark-false使用cudnn默认的卷积优化算法
    cudnn.deterministic = True
    cudnn.benchmark = False


def setup_run(args):
    set_up_gpu(args)
    experiment_path = setup_experiment_folder(args)
    pp.pprint({k: v for k, v in args.items() if v is not None}, width=1)
    # pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return experiment_path


def setup_experiment_folder(args):
    """
    实验数据存储文件夹设置方法

    Args:
        args ([type]): [description]

    Returns:
        str: 返回实验数据文件夹路径
    """
    experiment_dir = os.path.join("experiments", args.experiment_dir)
    experiment_description = args.experiment_description
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    experiment_path = os.path.join(experiment_dir, experiment_description)
    if args.run_mode == "train":
        if args.resume_training:
            assert os.path.exists(experiment_path), "[!] RESUME 模型文件夹路径不存在，检查「args.experiment_description」！"
            export_experiments_config_as_json(args, experiment_path, json_name="resume_config.json")
        else:
            experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
            os.mkdir(experiment_path)
            export_experiments_config_as_json(args, experiment_path)
        write_run_msg(experiment_path)
    elif args.run_mode == "analyse":
        assert os.path.exists(experiment_path), "[!] 未发现模型文件夹，检查「args.experiment_description」！"
    else:
        raise ValueError("[!]「args.run_mode」 模式错误！  --可选：train、analyse")
    loguru_logger.info("文件夹位置为: " + os.path.abspath(experiment_path))
    return experiment_path


def write_run_msg(experiment_path):
    pid = str(os.getpid())
    f = open(os.path.join(experiment_path, "pid.txt"), "w")
    f.write("该训练脚本的启动时间为 : " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    f.write("该训练脚本的 PID 为 : " + str(pid))
    f.close()


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def export_experiments_config_as_json(args, experiment_path, json_name="config.json"):
    with open(os.path.join(experiment_path, json_name), "w") as outfile:
        json.dump(vars(args), outfile, indent=4)


def all_subclasses(cls):
    """
    返回所有的子类
    Returns:
        [class]: 父类类名
    """
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


def import_all_subclasses(_file, _name, _class):
    """
    导入所有的子类

    Args:
        _file ([type]): [description]
        _name ([type]): [description]
        _class ([type]): [description]
    """
    modules = get_all_submodules(_file, _name)
    for m in modules:
        for i in dir(m):
            attribute = getattr(m, i)
            if inspect.isclass(attribute) and issubclass(attribute, _class):
                setattr(sys.modules[_name], i, attribute)


def get_all_submodules(_file, _name):
    modules = []
    _dir = os.path.dirname(_file)
    for _, name, ispkg in pkgutil.iter_modules([_dir]):
        module = import_module("." + name, package=_name)
        modules.append(module)
        if ispkg:
            modules.extend(get_all_submodules(module.__file__, module.__name__))
    return modules


def set_up_gpu(args):
    device_idx = str(args.device_idx)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
    args.num_gpu = len(device_idx.split(","))
    args.device_idx = [int(i) for i in device_idx.split(",")]


def save_test_result(export_root, result):
    """
    保存测试结果到文件

    Args:
        export_root (str): 保存路径
        result (str): [description]
    """
    filepath = Path(export_root).joinpath("test_result.txt")
    with filepath.open("w") as f:
        json.dump(result, f, indent=2)


def load_pretrained_weights(model, path):
    """
    加载预训练模型权重

    Args:
        model (object): 模型类的实例化对象
        path (str): 模型参数地址
    """
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[config.STATE_DICT_KEY] if config.STATE_DICT_KEY in chk_dict else chk_dict["state_dict"]
    model.load_state_dict(model_state_dict)


def setup_to_resume_from_recent(args, path):
    """
    从最近的保存点继续训练

    Args:
        args ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
    """
    epoch_start = 0
    accum_iter_start = 0
    model = None
    optimizer = None
    lr_scheduler = None

    chk_dict = torch.load(os.path.join(path, "models", config.RECENT_STATE_DICT_FILENAME))
    if args.num_gpu > 1:
        model.module.load_state_dict(chk_dict[config.STATE_DICT_KEY])
    else:
        model.load_state_dict(chk_dict[config.STATE_DICT_KEY])
    if config.OPTIMIZER_STATE_DICT_KEY in chk_dict:
        optimizer.load_state_dict(chk_dict[config.OPTIMIZER_STATE_DICT_KEY])
    if config.SCHEDULER_STATE_DICT_KEY in chk_dict:
        lr_scheduler.load_state_dict(chk_dict[config.SCHEDULER_STATE_DICT_KEY])
    epoch_start, accum_iter_start = chk_dict[config.STEPS_DICT_KEY]
    return epoch_start, accum_iter_start, model, optimizer, lr_scheduler


class AverageMeterSet(object):
    def __init__(self, meters=None):
        """
        训练指标数据处理类，负责各类指标数据操作, dict

        Args:
            object ([type]): [description]
            meters (object, optional): [description]. Defaults to None.
        """
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter(key)
            meter.update(0)
            return meter
        return self.meters[key]

    def keys(self):
        return self.meters.keys()

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter(name)
        self.meters[name].update(value, n)

    def reset(self):
        if len(self.meters) != 0:
            for meter in self.meters.values():
                meter.reset()

    """
    values() averages() sums() counts() 均返回字典：{name：meter.xxx}
    """

    def values(self, format_string="{}"):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string="{}"):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string="{}"):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string="{}"):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}

    def get_meters(self, meter_type):
        if meter_type == "sum":
            return self.sums()
        elif meter_type == "avg":
            return self.averages()
        elif meter_type == "value":
            return self.values()
        elif meter_type == "count":
            return self.counts()
        else:
            raise ValueError('[!] meter_type error, choise ["sum", "avg", "val", "count"]')


class AverageMeter(object):
    """
    计算并存储平均值 average 和当前值 current value
    Args:
        name (str): 数据的名称，Loss？ Acc？ or other
        val (int): 一次 batch 计算的当前值
        avg (int): 总 batch 计算的平均值
        sum (int): 总 batch 计算的累加值
        count (int, optional): batch 次的累计值. Defaults 1 个 batch.
        fmt (str, optional): [description]. Defaults to ':f'.
    """

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def get_data(self, meter_type):
        if meter_type == "sum":
            return self.sum
        elif meter_type == "avg":
            return self.avg
        elif meter_type == "value":
            return self.val
        elif meter_type == "count":
            return self.count
        else:
            raise ValueError('[!] meter_type error, choise ["sum", "avg", "val", "count"]')
