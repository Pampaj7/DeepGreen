#!/bin/bash
# === Sequential execution of each MATLAB training script ===

matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); resnet18.train_cifar100; exit"
sleep 30

matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); resnet18.train_fashion; exit"
sleep 30

matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); resnet18.train_tiny; exit"
sleep 30

matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); vgg16.train_cifar100; exit"
sleep 30

matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); vgg16.train_fashion; exit"
sleep 30

matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); vgg16.train_tiny; exit"

echo "All scripts have been executed."