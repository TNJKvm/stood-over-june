import ddu_dirty_mnist
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.datasets.cinic10 import CINIC10
from src.datasets.clothing1m import Clothin1m
from src.datasets.qmnist import QMNIST
from src.utils.noise import mnist_structured_noise, mnist_label_noise
from src.utils import unbalance


def get_CIFAR10(datadir, batch_size):
    transformer = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_and_val = CIFAR10(datadir, train=True, transform=transformer, download=True)

    test = CIFAR10(datadir, train=False, transform=transformer, download=True)
    train = Subset(train_and_val, list(range(0, len(train_and_val), 2)))
    val = Subset(train_and_val, list(range(1, len(train_and_val), 2)))

    train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_CIFAR10_no_holdout(datadir, batch_size):
    transformer = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_and_val = CIFAR10(datadir, train=True, transform=transformer, download=True)

    test = CIFAR10(datadir, train=False, transform=transformer, download=True)
    val_1 = Subset(train_and_val, list(range(1, len(train_and_val), 4)))
    val_2 = Subset(train_and_val, list(range(0, len(train_and_val), 4)))

    val_1_dataloader = DataLoader(dataset=val_1, batch_size=batch_size, shuffle=True)
    val_2_dataloader = DataLoader(dataset=val_2, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    return val_1_dataloader, val_2_dataloader, test_dataloader


def get_CIFAR100(datadir, batch_size):
    transformer = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_and_val = CIFAR100(datadir, train=True, transform=transformer, download=True)

    test = CIFAR100(datadir, train=False, transform=transformer, download=True)
    train = Subset(train_and_val, list(range(0, len(train_and_val), 2)))
    val = Subset(train_and_val, list(range(1, len(train_and_val), 2)))

    train_dataloader = DataLoader(dataset=train, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def get_CIFAR100_no_holdout(datadir, batch_size):
    transformer = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_and_val = CIFAR100(datadir, train=True, transform=transformer, download=True)

    test = CIFAR100(datadir, train=False, transform=transformer, download=True)
    val_1 = Subset(train_and_val, list(range(1, len(train_and_val), 4)))
    val_2 = Subset(train_and_val, list(range(0, len(train_and_val), 4)))

    val_1_dataloader = DataLoader(dataset=val_1, batch_size=batch_size, shuffle=True)
    val_2_dataloader = DataLoader(dataset=val_2, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    return val_1_dataloader, val_2_dataloader, test_dataloader


def get_QMNIST(datadir, batch_size):
    transformer = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = DataLoader(dataset=MNIST(datadir, train=True, download=True, transform=transformer),
                               batch_size=batch_size)
    val_dataset = DataLoader(dataset=QMNIST(datadir, "test50k", download=True, transform=transformer),
                             batch_size=batch_size)
    test_dataset = DataLoader(dataset=MNIST(datadir, train=False, download=True, transform=transformer),
                              batch_size=batch_size)

    return train_dataset, val_dataset, test_dataset


def get_CINIC10(datadir, batchsize):
    cinic = CINIC10(root=datadir, download=True)
    train_dataset = DataLoader(cinic.get_dataset('train'), batch_size=batchsize, shuffle=True)
    val_dataset = DataLoader(cinic.get_dataset('validation'), batch_size=batchsize, shuffle=True)
    test_dataset = DataLoader(cinic.get_dataset('test'), batch_size=batchsize, shuffle=True)

    return train_dataset, val_dataset, test_dataset


def get_CINIC10_no_holdout(datadir, batchsize):
    cinic = CINIC10(root=datadir, download=True)
    val_1_dataset = DataLoader(
        Subset(cinic.get_dataset('validation'), list(range(1, len(cinic.get_dataset('validation')), 2))),
        batch_size=batchsize, shuffle=True)
    val_2_dataset = DataLoader(
        Subset(cinic.get_dataset('validation'), list(range(0, len(cinic.get_dataset('validation')), 2))),
        batch_size=batchsize, shuffle=True)
    test_dataset = DataLoader(cinic.get_dataset('test'), batch_size=batchsize, shuffle=True)

    return val_1_dataset, val_2_dataset, test_dataset


def get_MNIST_label_noise(datadir, batch_size, pc_corrupted):
    transformer = transforms.Compose(
        [transforms.ToTensor()]
    )
    dataset = QMNIST(datadir, "test50k", download=True, transform=transformer)
    dataset_struct_noise, _mask = mnist_label_noise(dataset, pc_corrupted)
    val_dataset = DataLoader(dataset=dataset_struct_noise, batch_size=batch_size)
    return val_dataset


def get_MNIST_struct_noise(datadir, batch_size, probability):
    transformer = transforms.Compose(
        [transforms.ToTensor()]
    )
    dataset = QMNIST(datadir, "test50k", download=True, transform=transformer)
    dataset_struct_noise, _mask = mnist_structured_noise(dataset, probability, digits=[3, 4, 9], replacements=[5, 5, 7])
    val_dataset = DataLoader(dataset=dataset_struct_noise, batch_size=batch_size)
    return val_dataset


def get_MNIST_ambiguous(datadir, batch_size):
    dataset = ddu_dirty_mnist.AmbiguousMNIST(datadir, train=True, download=True, device="cuda")
    dataset = torch.utils.data.Subset(dataset, np.random.choice(np.arange(0, len(dataset)), 50000, replace=False))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def get_Cloting1M(datadir, batch_size):
    dataset_test = DataLoader(Clothin1m(datadir, 'test'), batch_size=batch_size, num_workers=4, shuffle=True)
    dataset_val = DataLoader(Clothin1m(datadir, 'noisy_train'), batch_size=batch_size, num_workers=4, shuffle=True)
    dataset_train = DataLoader(Clothin1m(datadir, 'train'), batch_size=batch_size, num_workers=4, shuffle=True)

    return dataset_train, dataset_val, dataset_test


def get_CIFAR100_unbalanced(datadir, batch_size):
    cifar100_train = CIFAR100(root=datadir, train=True, download=True)
    cifar100_test = CIFAR100(root=datadir, train=False, download=True)

    u_i, u_l, selected_classes = unbalance.unbalance_dataset(cifar100_train)
    u_i = torch.stack([torch.reshape(torch.from_numpy(item).float(), (3, 32, 32)) for item in u_i])
    u_l = torch.from_numpy(np.array(u_l))

    dataset = TensorDataset(u_i, u_l)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    u_i, u_l, selected_classes = unbalance.unbalance_dataset(cifar100_test, classes=selected_classes)
    u_i = torch.stack([torch.reshape(torch.from_numpy(item).float(), (3, 32, 32)) for item in u_i])
    u_l = torch.from_numpy(np.array(u_l))

    dataset = TensorDataset(u_i, u_l)
    test_dataloader = DataLoader(dataset, batch_size=batch_size)

    return train_dataloader, None, test_dataloader
