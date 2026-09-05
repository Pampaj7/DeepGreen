**Training against inference ranking, per block**

| model    | dataset      |   n_ecosystems |   spearman_rho |          p | same_best   | identical_order   | best_training     | best_inference   |
|:---------|:-------------|---------------:|---------------:|-----------:|:------------|:------------------|:------------------|:-----------------|
| resnet18 | cifar100     |              7 |          0.714 | 0.0713436  | True        | False             | Cpp/LibTorch      | Cpp/LibTorch     |
| resnet18 | fashionmnist |              7 |          0.929 | 0.00251947 | True        | False             | Cpp/LibTorch      | Cpp/LibTorch     |
| resnet18 | tinyimagenet |              7 |          0.821 | 0.0234488  | True        | False             | Cpp/LibTorch      | Cpp/LibTorch     |
| vgg16    | cifar100     |              7 |          0.536 | 0.215217   | False       | False             | Python/TensorFlow | Cpp/LibTorch     |
| vgg16    | fashionmnist |              7 |          0.536 | 0.215217   | False       | False             | Python/TensorFlow | Cpp/LibTorch     |
| vgg16    | tinyimagenet |              7 |          0.679 | 0.0937503  | False       | False             | Python/JAX        | Cpp/LibTorch     |
