**Cross-ecosystem accuracy spread before and after excluding collapsed runs**

| model    | dataset      |   n_ecosystems |   raw_spread_pp |   converged_spread_pp |   converged_min_pct |   converged_max_pct |
|:---------|:-------------|---------------:|----------------:|----------------------:|--------------------:|--------------------:|
| resnet18 | cifar100     |              7 |            7.46 |                  7.46 |               24.64 |               32.1  |
| resnet18 | fashionmnist |              6 |            1.25 |                  1.25 |               89.35 |               90.61 |
| resnet18 | tinyimagenet |              7 |            6.29 |                  6.29 |               10.85 |               17.14 |
| vgg16    | cifar100     |              6 |           20.19 |                  1.26 |               33.31 |               34.57 |
| vgg16    | fashionmnist |              7 |            0.45 |                  0.45 |               92.27 |               92.72 |
| vgg16    | tinyimagenet |              7 |            7.77 |                  2.94 |               14.89 |               17.83 |
