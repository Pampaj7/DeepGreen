**Inference: ecosystem ranking under three energy definitions**

| ecosystem         |   as_measured_J |   gpu_only_J |   harmonised_J |   duration_s |   rank_as_measured |   rank_gpu_only |   rank_harmonised |   rank_duration |
|:------------------|----------------:|-------------:|---------------:|-------------:|-------------------:|----------------:|------------------:|----------------:|
| C++/LibTorch      |           246.2 |        131.9 |          300.4 |          1.6 |                  1 |               1 |                 1 |               1 |
| MATLAB/DLT        |           532.6 |        255.9 |          586.9 |          3.1 |                  2 |               4 |                 4 |               4 |
| Python/PyTorch    |           628.5 |        202.8 |          400.3 |          1.8 |                  3 |               2 |                 2 |               2 |
| Python/JAX        |           806.1 |        243.5 |          504.9 |          2.4 |                  4 |               3 |                 3 |               3 |
| Java/DL4J         |           896.7 |        522.7 |          950.7 |          4   |                  5 |               7 |                 6 |               6 |
| Rust/tch          |          1236.4 |        462.3 |          821.1 |          3.4 |                  6 |               5 |                 5 |               5 |
| Python/TensorFlow |          1661   |        491.7 |         1033.1 |          5.1 |                  7 |               6 |                 7 |               7 |
| R/torch           |          1792.1 |       1226.7 |         2654   |         13.3 |                  8 |               8 |                 8 |               8 |
