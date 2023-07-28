from dataloaders import AbstractDataloader
from torchvision import datasets, transforms


class MnistDataLoader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_path = "./datasets/"

    @classmethod
    def code(cls):
        return "mnist"

    def _get_dataset(self, mode):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
        if mode == "train":
            dataset = datasets.MNIST(root=self.dataset_path, train=True, transform=transform, download=True)
        elif mode == "val":
            dataset = datasets.MNIST(root=self.dataset_path, train=False, transform=transform, download=True)
        elif mode == "test":
            dataset = datasets.MNIST(root=self.dataset_path, train=False, transform=transform, download=True)
        else:
            raise ValueError

        return dataset


class Cifar10Dataloader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_path = "./datasets/"

    @classmethod
    def code(cls):
        return "cifar10"

    def _get_dataset(self, mode):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if mode == "train":
            dataset = datasets.CIFAR10(root=self.dataset_path, train=True, transform=transform_train, download=True)
        elif mode == "val":
            dataset = datasets.CIFAR10(root=self.dataset_path, train=False, transform=transform_test, download=True)
        elif mode == "test":
            dataset = datasets.CIFAR10(root=self.dataset_path, train=False, transform=transform_test, download=True)
        else:
            raise ValueError

        return dataset
