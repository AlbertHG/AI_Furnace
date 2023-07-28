from collections import Counter  # 引入Counter
from loguru import logger as loguru_logger
from ._base import AbstractModel
from utils import all_subclasses, import_all_subclasses, count_param

import_all_subclasses(__file__, __name__, AbstractModel)

# code 命名唯一性检测
code_name = [c.code() for c in all_subclasses(AbstractModel) if c.code() is not None]
if len(set(code_name)) != len(code_name):
    raise ValueError(
        "[!] Model code 发现命名重复, {}".format([key for key, value in dict(Counter(code_name)).items() if value > 1])
    )

MODELS = {c.code(): c for c in all_subclasses(AbstractModel) if c.code() is not None}


def model_factory(args):
    model = MODELS[args.model_code](args)
    param = count_param(model)
    loguru_logger.info("Model Totoal Parameters: %.2fMb  (%d)" % (param / 1e6, param))
    return model
