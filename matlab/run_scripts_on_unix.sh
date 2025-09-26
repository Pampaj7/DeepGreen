#!/bin/bash
# === Sequential execution of each MATLAB training script ===

/usr/local/MATLAB/R2023a/bin/matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); resnet18.train_cifar100; exit" -sd '/home/marcopaglio/DeepGreen/matlab'
sleep 30

/usr/local/MATLAB/R2023a/bin/matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); resnet18.train_fashion; exit" -sd '/home/marcopaglio/DeepGreen/matlab'
sleep 30

/usr/local/MATLAB/R2023a/bin/matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); resnet18.train_tiny; exit" -sd '/home/marcopaglio/DeepGreen/matlab'
sleep 30

/usr/local/MATLAB/R2023a/bin/matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); vgg16.train_cifar100; exit" -sd '/home/marcopaglio/DeepGreen/matlab'
sleep 30

/usr/local/MATLAB/R2023a/bin/matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); vgg16.train_fashion; exit" -sd '/home/marcopaglio/DeepGreen/matlab'
sleep 30

/usr/local/MATLAB/R2023a/bin/matlab -batch "setup_env('/usr/bin/python3', '/home/marcopaglio/DeepGreen'); vgg16.train_tiny; exit" -sd '/home/marcopaglio/DeepGreen/matlab'

echo "All scripts have been executed."