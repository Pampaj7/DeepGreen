**One configuration re-executed in a later window, against its original runs**

| ecosystem      | model    | dataset      | phase     |   n_original |   n_recheck |   original_J |   recheck_J |   difference_pct |   within_window_cv_pct |   difference_in_sd |
|:---------------|:---------|:-------------|:----------|-------------:|------------:|-------------:|------------:|-----------------:|-----------------------:|-------------------:|
| Python/PyTorch | resnet18 | fashionmnist | Inference |            5 |           5 |        231   |       206.6 |           -10.58 |                   9.56 |               1.55 |
| Python/PyTorch | resnet18 | fashionmnist | Training  |            5 |           5 |       4080.4 |      4072   |            -0.21 |                   0.38 |               0.61 |
