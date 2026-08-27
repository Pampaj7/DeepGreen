**CPU / GPU / RAM energy per epoch**

| ecosystem         | phase     |   cpu_J_per_epoch |   gpu_J_per_epoch |   ram_J_per_epoch |   total_J_per_epoch |   gpu_pct |   host_pct |
|:------------------|:----------|------------------:|------------------:|------------------:|--------------------:|----------:|-----------:|
| Rust/tch          | Inference |            142.29 |            462.35 |            631.81 |             1236.44 |     37.39 |      62.61 |
| C++/LibTorch      | Inference |             39.55 |            131.89 |             74.75 |              246.18 |     53.57 |      46.43 |
| Python/PyTorch    | Inference |             78.24 |            202.84 |            347.37 |              628.45 |     32.28 |      67.72 |
| Python/JAX        | Inference |            103.43 |            243.51 |            459.16 |              806.1  |     30.21 |      69.79 |
| Python/TensorFlow | Inference |            214.87 |            491.66 |            954.51 |             1661.03 |     29.6  |      70.4  |
| MATLAB/DLT        | Inference |             98.08 |            255.94 |            178.56 |              532.58 |     48.06 |      51.94 |
| R/torch           | Inference |            564.94 |           1226.72 |              0.48 |             1792.14 |     68.45 |      31.55 |
| Java/DL4J         | Inference |            129.57 |            522.71 |            244.47 |              896.74 |     58.29 |      41.71 |
| Rust/tch          | Training  |            372.42 |           1138.58 |           1653.68 |             3164.68 |     35.98 |      64.02 |
| C++/LibTorch      | Training  |            474.15 |           2630.26 |           1058.46 |             4162.86 |     63.18 |      36.82 |
| Python/PyTorch    | Training  |            534.34 |           2358.2  |           2373.97 |             5266.52 |     44.78 |      55.22 |
| Python/JAX        | Training  |            579.96 |           2324.27 |           2574.01 |             5478.24 |     42.43 |      57.57 |
| Python/TensorFlow | Training  |           1692.42 |           4505.49 |           7518.83 |            13716.7  |     32.85 |      67.15 |
| MATLAB/DLT        | Training  |           1305.62 |           4345.05 |           6686.4  |            12337.1  |     35.22 |      64.78 |
| R/torch           | Training  |           4331.63 |           9723.83 |              3.68 |            14059.1  |     69.16 |      30.84 |
| Java/DL4J         | Training  |           1924.39 |           8923.08 |           3632.67 |            14480.1  |     61.62 |      38.38 |
