#!/bin/bash
set -e  # stop immediato se un comando fallisce

export PYTHONPATH=.

echo "▶️ Avvio sequenza training JAX..."

# CIFAR100
echo "===> ResNet18 CIFAR100"
python3 python/jax/train/resnet18/train_cifar100.py

echo "===> VGG16 CIFAR100"
python3 python/jax/train/vgg16/train_cifar100.py

# Fashion-MNIST
echo "===> ResNet18 Fashion-MNIST"
python3 python/jax/train/resnet18/train_fashion.py

echo "===> VGG16 Fashion-MNIST"
python3 python/jax/train/vgg16/train_fashion.py

# Tiny ImageNet
echo "===> ResNet18 TinyImageNet"
python3 python/jax/train/resnet18/train_tiny.py

echo "===> VGG16 TinyImageNet"
python3 python/jax/train/vgg16/train_tiny.py

echo "🏁 Tutti i training JAX completati con successo."
