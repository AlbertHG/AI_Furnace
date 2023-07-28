import torch
import torch.nn as nn
from tqdm import tqdm
import time
import json
from abc import *
from pathlib import Path

from utils import *
from config import *


class AbstractAnalyzer(metaclass=ABCMeta):
    """
    测试器的抽象类，封装了各个模型测试相关的通用操作.

    Args:
        args ([type]): 全局参数对象
        model (object): 模型的实例化对象
        analyzer_loader (object): 测试集加载器
        export_root (str): Log、model的存放路径
    """

    def __init__(self, args, model, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.test_loader = test_loader

        self.export_root = export_root
        # 计算指标，example：Top 1，2，3，4，...
        self.metrics_list = args.metrics_list
        # 基于哪个指标保存模型，example：Top 1?、2?、3?...
        self.best_metric = args.best_metric

        # 指标数据记录的间隔
        self.log_period_as_iter = args.log_period_as_iter
        self.analyzer_meter_set = AverageMeterSet()
    

    def analyse(self):
        try:
            model_file = torch.load(os.path.join(self.export_root, "models", BEST_STATE_DICT_FILENAME))
        except RuntimeError:
            model_file = torch.load(
                os.path.join(self.export_root, "models", BEST_STATE_DICT_FILENAME),
                map_location=lambda storage, loc: storage,
            )
        best_model = model_file.get(STATE_DICT_KEY)
        best_epoch = model_file.get("epoch") - 1

        loguru_logger.info("分析最佳模型, 模型来自于 Epoch = {}".format(model_file.get("epoch")))

        self.model.load_state_dict(best_model)
        self.model.eval()

        self.analyzer_meter_set.reset()
        fix_random_seed_as(best_epoch)

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):

                _, batch = self._batch_to_device(batch)
                scores = self.feed_forward(batch)
                metrics = self.calculate_metrics(batch, scores)
                self.batch_analyse(batch, scores, metrics)

                description = "Analyse: "
                for k, v in metrics.items():
                    self.analyzer_meter_set.update(k, v)
                    description = description + ":".join(
                        [k, "{:.5f} ".format(self.analyzer_meter_set[k].get_data(self.args.metrics_meter_type))]
                    )

                tqdm_dataloader.set_description(description)

            metrics_meter = self.analyzer_meter_set.get_meters(self.args.metrics_meter_type)
            self.full_analyse()

            with open(os.path.join(self.export_root, "analyse_result.json"), "a") as f:
                json.dump(metrics_meter, f, indent=4)
            print(metrics_meter)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_metrics(self, batch, scores):
        pass

    def batch_analyse(self, batch, scores, metrics):
        pass

    def full_analyse(self):
        pass

    def feed_forward(self, batch):
        return self.model(batch)

    def _batch_to_device(self, batch):
        """
        在此处将一个 batch 的数据 转换为模型需要的格式的数据，并加载到device中
        """
        batch_size = list(batch.values())[0].size(0)
        batch = {k: v.to(self.device) for (k, v) in batch.items()}
        return batch_size, batch

    def _needs_to_log(self, accum_iter):
        return accum_iter != 0 and (accum_iter % (self.log_period_as_iter * self.args.train_batch_size)) == 0
