#!/bin/bash
set -e  # interrompe se un comando fallisce

export PYTHON_BIN=$(command -v python3)

echo "▶️ Avvio sequenza training R..."

# CIFAR100
echo "===> ResNet18 CIFAR100"
Rscript R/train/resnet18/train_cifar100.r

echo "===> VGG16 CIFAR100"
Rscript R/train/vgg/train_cifar100.r

# Fashion-MNIST
echo "===> ResNet18 Fashion-MNIST"
Rscript R/train/resnet18/train_fashion.r

echo "===> VGG16 Fashion-MNIST"
Rscript R/train/vgg/train_fashion.r

# Tiny ImageNet
echo "===> ResNet18 TinyImageNet"
Rscript R/train/resnet18/train_tiny.r

echo "===> VGG16 TinyImageNet"
Rscript R/train/vgg/train_tiny.r

echo "🏁 Tutti i training R completati con successo."

#./R/run.sh