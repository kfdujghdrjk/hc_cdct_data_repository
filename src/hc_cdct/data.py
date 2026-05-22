from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision
import torchvision.transforms as transforms


@dataclass
class DataConfig:
    dataset: str = "CIFAR10"
    root: str = "data/raw"
    batch_size_train: int = 128
    batch_size_test: int = 100
    num_workers: int = 2
    download: bool = True


def get_cifar_loaders(config: DataConfig):
    """Create train/test dataloaders for CIFAR-10 or CIFAR-100."""

    dataset_name = config.dataset.upper()

    if dataset_name == "CIFAR10":
        dataset_cls = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == "CIFAR100":
        dataset_cls = torchvision.datasets.CIFAR100
        # Commonly used CIFAR-100 normalization statistics.
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = dataset_cls(
        root=config.root,
        train=True,
        download=config.download,
        transform=transform_train,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=config.num_workers,
    )

    testset = dataset_cls(
        root=config.root,
        train=False,
        download=config.download,
        transform=transform_test,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size_test,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return trainloader, testloader
