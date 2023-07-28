from collections import Counter  # 引入Counter
from ._base import AbstractTrainer
from .ctr import CTRTrainer
from utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractTrainer)
# code 命名唯一性检测

code_name = [c.code() for c in all_subclasses(AbstractTrainer) if c.code() is not None]
if len(set(code_name)) != len(code_name):
    raise ValueError(
        "[!] Trainer code 发现命名重复, {}".format([key for key, value in dict(Counter(code_name)).items() if value > 1])
    )


TRAINERS = {c.code(): c for c in all_subclasses(AbstractTrainer) if c.code() is not None}

def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)

