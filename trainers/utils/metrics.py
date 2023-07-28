import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def accuracy_top1(labels, scores):
    """计算当前 Batch 准确率Top1

    Args:
        scores (Tensor): 预测值
        labels (Tensor): 样本标签

    """
    metrics = {}
    batch_size = scores.size(0)
    _, predicted = scores.max(1)
    correct = predicted.eq(labels).sum().item()
    metrics["Top@1"] = correct / batch_size
    return metrics


def accuracy_topk(labels, scores, ks=[1, 2, 3, 4], ignore_idxs=None):
    """计算当前 Batch 准确率TopK

    Args:
        scores (Tensor): 预测值
        labels (Tensor): 样本标签
        metric_ks (list): Topk中的 K example：K=[1,2,3,4,...]
        ignore_idxs (Tensor, optional): 掩码Tensor, shape=[num_classes]
                                        不加入排序的位置置 0, 置 1 的类别将参与排序

    """
    metrics = {}
    batch_size = scores.size(0)

    if ignore_idxs is not None:
        assert scores.size(-1) == ignore_idxs.size(
            -1
        ), "[!] ignore_idxs的类别数和scores的类别数不一致！"
        ignore_idxs = ignore_idxs.view(-1, ignore_idxs.size(-1))
        scores = scores * ignore_idxs

    _, max_k = torch.topk(scores, k=max(ks), dim=-1)
    labels = labels.view(-1, 1)  # [n] --> [n,1]

    for k in ks:
        top = (labels == max_k[:, 0:k]).sum().item()
        metrics["Top@{}".format(k)] = top / batch_size
    return metrics


def accuracy_topk_add_bias(labels, scores, ks=[1, 2, 3, 4], ignore_idxs=None, bias=0.0):
    """将当前 Batch 计算的topk + 微信占比

    Args:
        scores (Tensor): 预测值
        labels (Tensor): 样本标签
        ks (list): Topk中的 K example：K=[1,2,3,4,...]
        bias (float): 微信占比
        ignore_idxs (Tensor, optional): 掩码Tensor，shape=[num_classes]，不加入排序的位置置 0

    Returns:
        [dict]: [description]
    """
    metrics = accuracy_topk(labels, scores, ks, ignore_idxs)
    metrics = {k: v * (1 - bias) + bias for (k, v) in metrics.items()}
    return metrics


class AUC(object):
    def __init__(self, num_buckets):
        self._num_buckets = num_buckets
        self._table = np.zeros(shape=[2, self._num_buckets])

    def reset(self):
        self._table = np.zeros(shape=[2, self._num_buckets])

    def update(self, labels: np.ndarray, predicts: np.ndarray):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.astype(np.int)
        predicts = self._num_buckets * predicts

        buckets = np.round(predicts).astype(np.int)
        buckets = np.where(buckets < self._num_buckets, buckets, self._num_buckets - 1)

        for i in range(len(labels)):
            self._table[labels[i], buckets[i]] += 1

    def compute(self):
        tn = 0
        tp = 0
        area = 0
        for i in range(self._num_buckets):
            new_tn = tn + self._table[0, i]
            new_tp = tp + self._table[1, i]
            # self._table[1, i] * tn + self._table[1, i]*self._table[0, i] / 2
            area += (new_tp - tp) * (tn + new_tn) / 2
            tn = new_tn
            tp = new_tp
        if tp < 1e-3 or tn < 1e-3:
            return -0.5  # 样本全正例，或全负例
        return area / (tn * tp)


if __name__ == "__main__":
    label = np.random.randint(low=0, high=2, size=[100000])
    predict = np.random.uniform(0, 1, size=[100000])
    auc = AUC(num_buckets=102400)
    auc.update(label, predict)
    print(auc.compute())
    print(roc_auc_score(label, predict))
