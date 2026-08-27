**Within-run (pseudo-replicate) dispersion**

| ecosystem         | phase     |   n_epochs |   mean_energy_j |   median_energy_j |   cv_pct |   epoch_ci95_lo |   epoch_ci95_hi |
|:------------------|:----------|-----------:|----------------:|------------------:|---------:|----------------:|----------------:|
| C++/LibTorch      | Inference |        180 |         246.183 |           238.203 |   21.751 |         238.49  |         253.942 |
| C++/LibTorch      | Training  |        180 |        4162.86  |          3706.36  |   56.291 |        3823.1   |        4511.45  |
| Java/DL4J         | Inference |        180 |         896.744 |           843.215 |   44.503 |         841.481 |         957.326 |
| Java/DL4J         | Training  |        180 |       14480.1   |         12596     |   30.837 |       13839.6   |       15119.6   |
| MATLAB/DLT        | Inference |        180 |         532.583 |           485.682 |   29.825 |         509.94  |         555.537 |
| MATLAB/DLT        | Training  |        180 |       12337.1   |         10702.8   |   81.028 |       10901.9   |       13828.1   |
| Python/JAX        | Inference |        180 |         806.097 |           725.73  |   36.234 |         762.795 |         848.5   |
| Python/JAX        | Training  |        180 |        5478.24  |          3738.11  |   51.445 |        5066.69  |        5892.76  |
| Python/PyTorch    | Inference |        180 |         628.451 |           583.269 |   22.581 |         608.096 |         649.391 |
| Python/PyTorch    | Training  |        180 |        5266.52  |          5029.03  |   43.16  |        4939.21  |        5614.89  |
| Python/TensorFlow | Inference |        180 |        1661.04  |          1500.09  |   19.028 |        1615.69  |        1707.01  |
| Python/TensorFlow | Training  |        180 |       13716.7   |         10289.3   |   45.87  |       12799.3   |       14655.3   |
| R/torch           | Inference |        180 |        1792.14  |          1830.73  |    8.647 |        1768.98  |        1814.74  |
| R/torch           | Training  |        180 |       14059.1   |         12869.2   |   34.218 |       13363.1   |       14776.8   |
| Rust/tch          | Inference |        180 |        1236.44  |          1169.21  |   20.924 |        1198.96  |        1275.64  |
| Rust/tch          | Training  |        180 |        3164.68  |          2236.9   |   72.728 |        2834.58  |        3513.21  |
