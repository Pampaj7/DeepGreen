**Chance-accuracy runs separated by their per-epoch traces**

| run                                       | ecosystem         | model   | dataset      |   final_test_acc_pct |   train_loss_drop_pct |   test_loss_first |   test_loss_last | diagnosis             |
|:------------------------------------------|:------------------|:--------|:-------------|---------------------:|----------------------:|------------------:|-----------------:|:----------------------|
| Cpp-LibTorch_vgg16_tinyimagenet_rep1      | C++/LibTorch      | vgg16   | tinyimagenet |                  0.5 |                     0 |              5.3  |             5.3  | optimisation collapse |
| Cpp-LibTorch_vgg16_tinyimagenet_rep3      | C++/LibTorch      | vgg16   | tinyimagenet |                  0.5 |                     0 |              5.3  |             5.3  | optimisation collapse |
| Java-DL4J_vgg16_cifar100_rep2             | Java/DL4J         | vgg16   | cifar100     |                  1   |                     0 |            nan    |           nan    | optimisation collapse |
| Java-DL4J_vgg16_cifar100_rep3             | Java/DL4J         | vgg16   | cifar100     |                  1   |                     0 |            nan    |           nan    | optimisation collapse |
| Java-DL4J_vgg16_cifar100_rep4             | Java/DL4J         | vgg16   | cifar100     |                  1   |                     0 |            nan    |           nan    | optimisation collapse |
| Java-DL4J_vgg16_tinyimagenet_rep2         | Java/DL4J         | vgg16   | tinyimagenet |                  0.5 |                     0 |            nan    |           nan    | optimisation collapse |
| Java-DL4J_vgg16_tinyimagenet_rep4         | Java/DL4J         | vgg16   | tinyimagenet |                  0.5 |                    -0 |            nan    |           nan    | optimisation collapse |
| Python-PyTorch_vgg16_tinyimagenet_rep1    | Python/PyTorch    | vgg16   | tinyimagenet |                  0.5 |                    -0 |              5.3  |             5.3  | optimisation collapse |
| Python-TensorFlow_vgg16_cifar100_rep0     | Python/TensorFlow | vgg16   | cifar100     |                  1   |                    -0 |              4.61 |             4.61 | optimisation collapse |
| Python-TensorFlow_vgg16_cifar100_rep3     | Python/TensorFlow | vgg16   | cifar100     |                  1   |                     0 |              4.61 |             4.61 | optimisation collapse |
| Python-TensorFlow_vgg16_tinyimagenet_rep2 | Python/TensorFlow | vgg16   | tinyimagenet |                  0.5 |                     0 |              5.3  |             5.3  | optimisation collapse |
| Python-TensorFlow_vgg16_tinyimagenet_rep4 | Python/TensorFlow | vgg16   | tinyimagenet |                  0.5 |                     0 |              5.3  |             5.3  | optimisation collapse |
