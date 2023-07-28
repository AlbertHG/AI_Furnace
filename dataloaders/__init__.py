from collections import Counter  # 引入Counter
from ._base import AbstractDataloader, SplitCSVDataLoader, CSVChunkDataLoader
from utils import all_subclasses, import_all_subclasses

import_all_subclasses(__file__, __name__, AbstractDataloader)


# code 命名唯一性检测
code_name = [c.code() for c in all_subclasses(AbstractDataloader) if c.code() is not None]
if len(set(code_name)) != len(code_name):
    raise ValueError(
        "[!] Dataloader code 发现命名重复, {}".format([key for key, value in dict(Counter(code_name)).items() if value > 1])
    )

DATALOADERS = {c.code(): c for c in all_subclasses(AbstractDataloader) if c.code() is not None}


def dataloader_factory(args):
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args)

    if args.run_mode == "train":
        train, val, test = dataloader.get_dataloaders()
        return train, val, test
    elif args.run_mode == "analyse":
        test = dataloader.get_test_loader()
        return test
    else:
        raise ValueError('[!] "args.mode" 训练模式错误！')
