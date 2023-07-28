# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : template 配置文件处理

import yaml

import os
from copy import deepcopy


def set_template(conf):
    """将 args 的参数和 json的参数进行合并 并更新
    Args:
        conf ([type]): 来自 args 的参数
    """
    given = deepcopy(conf)
    for template_name in conf['template']:
        set_single_template(conf, template_name)  # yaml --> conf
    # 将 non none 的 parser --> conf
    overwrite_with_nonnones(conf, given)  # apply given(non-nones) last (highest priority)


def set_single_template(conf, template_name):
    """
    用 template_name 文件中的值，更新conf中的值
    """
    template = load_template(template_name)
    overwrite(conf, template)


def load_template(template_name):
    """
    加载json template文件
    """
    return yaml.safe_load(open(os.path.join('templates', f'{template_name}.yaml')))


def overwrite(this_dict, other_dict):
    for k, v in other_dict.items():
        if isinstance(v, dict):
            overwrite(this_dict[k], v)
        else:
            this_dict[k] = v


def overwrite_with_nonnones(this_dict, other_dict):
    for k, v in other_dict.items():
        if isinstance(v, dict):
            overwrite_with_nonnones(this_dict[k], v)
        elif v is not None:
            this_dict[k] = v
            print(k, v)