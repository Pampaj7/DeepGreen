**VGG-16 collapses per ecosystem; the effect is not stack-specific**

| ecosystem         | dataset      |   n_runs |   n_collapsed |
|:------------------|:-------------|---------:|--------------:|
| C++/LibTorch      | cifar100     |        5 |             0 |
| Java/DL4J         | cifar100     |        5 |             3 |
| Python/JAX        | cifar100     |        5 |             0 |
| Python/PyTorch    | cifar100     |        5 |             0 |
| Python/TensorFlow | cifar100     |        5 |             2 |
| R/torch           | cifar100     |        5 |             0 |
| C++/LibTorch      | fashionmnist |        5 |             0 |
| Java/DL4J         | fashionmnist |        5 |             0 |
| Python/JAX        | fashionmnist |        5 |             0 |
| Python/PyTorch    | fashionmnist |        5 |             0 |
| Python/TensorFlow | fashionmnist |        5 |             0 |
| R/torch           | fashionmnist |        5 |             0 |
| Rust/tch          | fashionmnist |        5 |             0 |
| C++/LibTorch      | tinyimagenet |        5 |             2 |
| Java/DL4J         | tinyimagenet |        5 |             2 |
| Python/JAX        | tinyimagenet |        5 |             0 |
| Python/PyTorch    | tinyimagenet |        5 |             1 |
| Python/TensorFlow | tinyimagenet |        5 |             2 |
| R/torch           | tinyimagenet |        5 |             0 |
| Rust/tch          | tinyimagenet |        5 |             0 |
