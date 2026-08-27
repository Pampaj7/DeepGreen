**Training against inference ranking, per block**

| model    | dataset      |   n_ecosystems |   spearman_rho |           p | same_best   | identical_order   | best_training   | best_inference   |
|:---------|:-------------|---------------:|---------------:|------------:|:------------|:------------------|:----------------|:-----------------|
| resnet18 | cifar100     |              7 |          0.964 | 0.000454149 | True        | False             | Python/JAX      | Python/JAX       |
| resnet18 | fashionmnist |              7 |          1     | 0           | True        | True              | Python/JAX      | Python/JAX       |
| resnet18 | tinyimagenet |              7 |          1     | 0           | True        | True              | Python/JAX      | Python/JAX       |
| vgg16    | cifar100     |              7 |          0.929 | 0.00251947  | True        | False             | Python/JAX      | Python/JAX       |
| vgg16    | fashionmnist |              7 |          0.964 | 0.000454149 | True        | False             | Python/JAX      | Python/JAX       |
| vgg16    | tinyimagenet |              7 |          0.964 | 0.000454149 | True        | False             | Python/JAX      | Python/JAX       |
