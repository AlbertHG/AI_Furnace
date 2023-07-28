from models import AbstractModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from layers import VGG, ResNet, DNN, PredictionLayer


class MnistModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    @classmethod
    def code(cls):
        return "mnist_cnn"

    def forward(self, batch):
        _return = {}
        x = batch[0]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        _return["y_pred"] = x

        return _return


class MnistModel2(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.dnn_input_units = args.dnn_input_units
        self.dnn_hidden_units = args.dnn_hidden_units
        self.activation = args.activation
        self.task = args.task
        self.dnn = DNN(
            self.dnn_input_units,
            self.dnn_hidden_units,
            activation=self.activation,
            dropout_p=args.dropout_p,
        )
        self.out = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return "mnist_dnn"

    def forward(self, batch):
        _return = {}
        x = batch[0]
        x = x.view(x.size(0), -1)
        x = self.dnn(x)
        x = self.out(x)
        # x = F.log_softmax(x, dim=1)
        _return["y_pred"] = x
        return _return


class Cifar10Model(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    @classmethod
    def code(cls):
        return "cifar10"

    def forward(self, batch):
        _return = {}
        x = batch[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        _return["y_pred"] = x
        return _return


class Cifar10VGGModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        self.feature_layer = VGG(vgg_name=args.vgg_name, in_channels=3)
        self.classifier = nn.Linear(512, 10)

    @classmethod
    def code(cls):
        return "cifar10_vgg"

    def forward(self, batch):
        _return = {}
        x = batch[0]
        x = self.feature_layer(x)
        x = self.classifier(x)
        # x = F.log_softmax(x, dim=1) # 搭配 F.mll_loss
        _return["y_pred"] = x  # 搭配 F.cross_entropy
        return _return


class ResnetVGGModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        self.feature_layer = ResNet(res_name=args.res_name, in_channels=3)
        if args.res_name in ["ResNet18", "ResNet34"]:
            expansion = 1
        else:
            expansion = 4
        self.classifier = nn.Linear(512 * expansion, 10)

    @classmethod
    def code(cls):
        return "cifar10_resnet"

    def forward(self, batch):
        _return = {}
        x = batch[0]
        x = self.feature_layer(x)
        x = self.classifier(x)
        # x = F.log_softmax(x, dim=1) # 搭配 F.mll_loss
        _return["y_pred"] = x  # 搭配 F.cross_entropy
        return _return
