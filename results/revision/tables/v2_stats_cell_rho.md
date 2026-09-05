**Energy-time rank agreement inside each block, where the workload is fixed and only the ecosystem varies**

| model    | dataset      | phase     |   n_ecosystems |   spearman_rho |           p |   discordant_pairs |   total_pairs |
|:---------|:-------------|:----------|---------------:|---------------:|------------:|-------------------:|--------------:|
| resnet18 | cifar100     | Inference |              7 |          1     | 0           |                  0 |            21 |
| resnet18 | cifar100     | Training  |              7 |          0.679 | 0.0937503   |                  5 |            21 |
| resnet18 | fashionmnist | Inference |              7 |          0.964 | 0.000454149 |                  1 |            21 |
| resnet18 | fashionmnist | Training  |              7 |          0.929 | 0.00251947  |                  2 |            21 |
| resnet18 | tinyimagenet | Inference |              7 |          0.893 | 0.00680719  |                  2 |            21 |
| resnet18 | tinyimagenet | Training  |              7 |          0.857 | 0.0136973   |                  3 |            21 |
| vgg16    | cifar100     | Inference |              7 |          0.964 | 0.000454149 |                  1 |            21 |
| vgg16    | cifar100     | Training  |              7 |          0.964 | 0.000454149 |                  1 |            21 |
| vgg16    | fashionmnist | Inference |              7 |          0.929 | 0.00251947  |                  2 |            21 |
| vgg16    | fashionmnist | Training  |              7 |          0.964 | 0.000454149 |                  1 |            21 |
| vgg16    | tinyimagenet | Inference |              7 |          0.929 | 0.00251947  |                  2 |            21 |
| vgg16    | tinyimagenet | Training  |              7 |          0.964 | 0.000454149 |                  1 |            21 |
