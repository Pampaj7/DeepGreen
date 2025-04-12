# scripts/download_cifar100.py

from torchvision.datasets import CIFAR100
from torchvision import transforms
import numpy as np


def download_cifar100():
    transform = transforms.Compose([transforms.ToTensor()])

    CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
    CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    print("✅ CIFAR-100 downloaded in ./data/cifar100")


def convert_to_npz():
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)

    x_train = np.stack([np.array(img) for img, _ in train_data])
    y_train = np.array([label for _, label in train_data])

    x_test = np.stack([np.array(img) for img, _ in test_data])
    y_test = np.array([label for _, label in test_data])

    np.savez_compressed('../../data/cifar100.npz',
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    print("✅ CIFAR-100 salvato in formato .npz")


if __name__ == '__main__':
    #download_cifar100()
    convert_to_npz()
