**Spread within the shared-backend control group against the full spread**

| model    | dataset      |   n_all |   n_libtorch |   spread_all |   spread_libtorch |   share_of_log_spread_pct |
|:---------|:-------------|--------:|-------------:|-------------:|------------------:|--------------------------:|
| resnet18 | cifar100     |       7 |            4 |        14.88 |              9.55 |                      83.6 |
| resnet18 | fashionmnist |       6 |            3 |        18.32 |             10.84 |                      82   |
| resnet18 | tinyimagenet |       6 |            3 |        10.25 |              8.52 |                      92   |
| vgg16    | cifar100     |       7 |            4 |         7.87 |              1.69 |                      25.4 |
| vgg16    | fashionmnist |       6 |            3 |         7.92 |              1.92 |                      31.5 |
| vgg16    | tinyimagenet |       6 |            3 |         7.82 |              1.81 |                      28.7 |
