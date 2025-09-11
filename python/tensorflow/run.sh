#!/bin/bash
set -e  # interrompe se un comando fallisce

export PYTHON_BIN=$(command -v python3)

echo "▶️ Avvio sequenza training TensorFlow (Python)..."

# CIFAR100
echo "===> ResNet18 CIFAR100"
#PYTHONPATH=. python3 python/tensorflow/train/resnet18/train_cifar100.py

echo "===> VGG16 CIFAR100"
#PYTHONPATH=. python3 python/tensorflow/train/vgg16/train_cifar100.py

# Fashion-MNIST
echo "===> ResNet18 Fashion-MNIST"
#PYTHONPATH=. python3 python/tensorflow/train/resnet18/train_fashion.py

echo "===> VGG16 Fashion-MNIST"
#PYTHONPATH=. python3 python/tensorflow/train/vgg16/train_fashion.py

# Tiny ImageNet
echo "===> ResNet18 TinyImageNet"
#PYTHONPATH=. python3 python/tensorflow/train/resnet18/train_tiny.py

echo "===> VGG16 TinyImageNet"
PYTHONPATH=. python3 python/tensorflow/train/vgg16/train_tiny.py

echo "🏁 Tutti i training TensorFlow completati con successo."
