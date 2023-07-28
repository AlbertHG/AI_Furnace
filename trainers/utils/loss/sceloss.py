import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    """
    Args:
        alpha ([type]): [description]
        beta ([type]): [description]
        num_classes (int, optional): [description]. Defaults to 10.

    References:
        - [Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, James Bailey; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 322-330]
    """

    def __init__(self, alpha, beta, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        device = pred.device
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class BSCELoss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.b_cross_entropy = torch.nn.BCELoss()

    def forward(self, pred, labels):
        # CBCE
        bce = self.b_cross_entropy(pred, labels)

        # RBCE
        pred = torch.cat((1 - pred, pred), dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels.long(), 2).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot.squeeze(1), min=1e-4, max=1.0)
        rbce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * bce + self.beta * rbce.mean()
        return loss
