#!/bin/bash
set -e  # interrompe se un comando fallisce

echo "▶️ Avvio sequenza training Rust..."

# CIFAR100
#echo "===> ResNet18 CIFAR100"
#cargo run --release --bin resnet_cifar100

echo "===> VGG16 CIFAR100"
cargo run --release --bin vgg_cifar100

# Fashion-MNIST
echo "===> ResNet18 Fashion-MNIST"
cargo run --release --bin resnet_fashion

echo "===> VGG16 Fashion-MNIST"
cargo run --release --bin vgg_fashion

# Tiny ImageNet
echo "===> ResNet18 TinyImageNet"
#cargo run --release --bin resnet_tiny

echo "===> VGG16 TinyImageNet"
cargo run --release --bin vgg_tiny

echo "🏁 Tutti i training Rust completati con successo."
