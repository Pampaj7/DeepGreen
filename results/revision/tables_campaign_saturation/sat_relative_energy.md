**Per-epoch energy of each ecosystem as a multiple of the cheapest of the seven, at 224 and on each 32x32 dataset, on sat_spread's definition; the largest value of a column is the seven-stack spread. Drawn as fig_saturation**

| model    | phase     | ecosystem         |   energy_224_J |   rel_224 |   rel_32_cifar100 |   rel_32_fashionmnist |   rel_32_tinyimagenet |   rel_32_mean |   rel_32_min |   rel_32_max |
|:---------|:----------|:------------------|---------------:|----------:|------------------:|----------------------:|----------------------:|--------------:|-------------:|-------------:|
| resnet18 | Training  | Rust/tch          |       4449.6   |    1.72   |            1.3176 |                1.161  |                1.604  |        1.3609 |       1.161  |       1.604  |
| resnet18 | Training  | C++/LibTorch      |       3267.97  |    1.2632 |            1      |                1      |                1      |        1      |       1      |       1      |
| resnet18 | Training  | Python/PyTorch    |       3502.14  |    1.3537 |            1.3012 |                1.2573 |                1.2835 |        1.2807 |       1.2573 |       1.3012 |
| resnet18 | Training  | Python/JAX        |       2587.03  |    1      |            1.1681 |                1.0486 |                1.0861 |        1.1009 |       1.0486 |       1.1681 |
| resnet18 | Training  | Python/TensorFlow |       2896.99  |    1.1198 |            1.4067 |                1.3649 |                1.3926 |        1.3881 |       1.3649 |       1.4067 |
| resnet18 | Training  | R/torch           |       9186.31  |    3.5509 |            9.7088 |                9.7829 |                9.2975 |        9.5964 |       9.2975 |       9.7829 |
| resnet18 | Training  | Java/DL4J         |      39570.8   |   15.2958 |            8.7278 |                8.7871 |                8.6233 |        8.7128 |       8.6233 |       8.7871 |
| resnet18 | Inference | Rust/tch          |        892.194 |    1.4937 |            1.3831 |                1.2407 |                2.3189 |        1.6476 |       1.2407 |       2.3189 |
| resnet18 | Inference | C++/LibTorch      |        730.615 |    1.2232 |            1      |                1      |                1      |        1      |       1      |       1      |
| resnet18 | Inference | Python/PyTorch    |        815.911 |    1.366  |            2.5177 |                2.9808 |                2.4915 |        2.6633 |       2.4915 |       2.9808 |
| resnet18 | Inference | Python/JAX        |        597.288 |    1      |            2.2964 |                2.6515 |                2.2518 |        2.3999 |       2.2518 |       2.6515 |
| resnet18 | Inference | Python/TensorFlow |        601.229 |    1.0066 |            2.2163 |                2.6503 |                2.2267 |        2.3644 |       2.2163 |       2.6503 |
| resnet18 | Inference | R/torch           |       3548.74  |    5.9414 |           30.6104 |               38.7305 |               31.7417 |       33.6942 |      30.6104 |      38.7305 |
| resnet18 | Inference | Java/DL4J         |      14662.9   |   24.5492 |           15.0152 |               18.8036 |               14.594  |       16.1376 |      14.594  |      18.8036 |
| vgg16    | Training  | Rust/tch          |      16209.7   |    1.6369 |            1.3081 |                1.2418 |                1.4633 |        1.3377 |       1.2418 |       1.4633 |
| vgg16    | Training  | C++/LibTorch      |      15034.5   |    1.5182 |            1.1584 |                1.1644 |                1.1789 |        1.1672 |       1.1584 |       1.1789 |
| vgg16    | Training  | Python/PyTorch    |      15243.7   |    1.5393 |            1.193  |                1.1993 |                1.211  |        1.2011 |       1.193  |       1.211  |
| vgg16    | Training  | Python/JAX        |      12012.5   |    1.213  |            1.03   |                1.0159 |                1      |        1.0153 |       1      |       1.03   |
| vgg16    | Training  | Python/TensorFlow |       9902.83  |    1      |            1      |                1      |                1.0296 |        1.0099 |       1      |       1.0296 |
| vgg16    | Training  | R/torch           |      19769.8   |    1.9964 |            4.0468 |                4.0497 |                4.1159 |        4.0708 |       4.0468 |       4.1159 |
| vgg16    | Training  | Java/DL4J         |     189258     |   19.1115 |            7.3917 |                7.3618 |                7.3821 |        7.3785 |       7.3618 |       7.3917 |
| vgg16    | Inference | Rust/tch          |       2716.08  |    1.6834 |            1.6584 |                1.387  |                2.299  |        1.7815 |       1.387  |       2.299  |
| vgg16    | Inference | C++/LibTorch      |       2158.14  |    1.3376 |            1      |                1      |                1      |        1      |       1      |       1      |
| vgg16    | Inference | Python/PyTorch    |       2337.81  |    1.4489 |            2.115  |                2.1544 |                2.4075 |        2.2256 |       2.115  |       2.4075 |
| vgg16    | Inference | Python/JAX        |       1634.99  |    1.0133 |            2.3296 |                2.3647 |                2.3362 |        2.3435 |       2.3296 |       2.3647 |
| vgg16    | Inference | Python/TensorFlow |       1613.49  |    1      |            1.8773 |                1.8956 |                2.0434 |        1.9388 |       1.8773 |       2.0434 |
| vgg16    | Inference | R/torch           |       5535.58  |    3.4308 |           18.8525 |               19.3314 |               19.8829 |       19.3556 |      18.8525 |      19.8829 |
| vgg16    | Inference | Java/DL4J         |      78795.1   |   48.8351 |           23.0935 |               23.3745 |               23.6716 |       23.3799 |      23.0935 |      23.6716 |
