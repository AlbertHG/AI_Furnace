# -*- coding: utf-8 -*-
# @Author        : HuangGang
# @Email         : hhhgggpps@gmail.com
# @Time          : 2021/11/10 12:00
# @Function      : 模型基类

import torch.nn as nn

from abc import *

class AbstractModel(nn.Module, metaclass=ABCMeta):
    """
    模型抽象类，所有的模型都得继承此类之后才能加入训练

    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass
