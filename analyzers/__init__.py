from collections import Counter  # 引入Counter
from ._base import AbstractAnalyzer
from utils import all_subclasses, import_all_subclasses

import_all_subclasses(__file__, __name__, AbstractAnalyzer)

# code 命名唯一性检测
code_name = [c.code() for c in all_subclasses(AbstractAnalyzer) if c.code() is not None]
if len(set(code_name)) != len(code_name):
    raise ValueError(
        "[!] analyzer code 发现命名重复, {}".format([key for key, value in dict(Counter(code_name)).items() if value > 1])
    )

ANALYZERS = {c.code(): c for c in all_subclasses(AbstractAnalyzer) if c.code() is not None}

def analyzer_factory(args, model, test_loader, export_root):
    analyzer = ANALYZERS[args.trainer_code]
    return analyzer(args, model, test_loader, export_root)