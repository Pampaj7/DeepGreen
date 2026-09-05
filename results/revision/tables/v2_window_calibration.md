**One configuration re-executed in a later window, against its original runs**

| ecosystem      | model    | dataset      | phase     |   n_original |   n_recheck |   original_J |   recheck_J |   difference_pct |   within_window_cv_pct |   difference_in_sd |
|:---------------|:---------|:-------------|:----------|-------------:|------------:|-------------:|------------:|-----------------:|-----------------------:|-------------------:|
| Python/PyTorch | resnet18 | fashionmnist | Inference |            5 |           5 |        173.4 |       167.8 |            -3.23 |                   0.95 |               3.95 |
| Python/PyTorch | resnet18 | fashionmnist | Training  |            5 |           5 |       1357.6 |      1340.6 |            -1.25 |                   0.59 |               1.62 |
