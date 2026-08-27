**Per-component energy and GPU load by ecosystem and phase**

| ecosystem         | phase     |   mean_gpu_power_w |   mean_total_power_w |   gpu_energy_pct |   cpu_energy_pct |   ram_energy_pct |   gpu_power_pct_of_tdp |
|:------------------|:----------|-------------------:|---------------------:|-----------------:|-----------------:|-----------------:|-----------------------:|
| C++/LibTorch      | Inference |              83.72 |               155.54 |            53.57 |            16.06 |            30.36 |                  23.92 |
| C++/LibTorch      | Training  |             197.06 |               310.83 |            63.18 |            11.39 |            25.43 |                  56.3  |
| Java/DL4J         | Inference |             137.75 |               229.64 |            58.29 |            14.45 |            27.26 |                  39.36 |
| Java/DL4J         | Training  |             188.18 |               291.81 |            61.62 |            13.29 |            25.09 |                  53.77 |
| MATLAB/DLT        | Inference |              82.61 |               170.64 |            48.06 |            18.42 |            33.53 |                  23.6  |
| MATLAB/DLT        | Training  |             126.51 |               316.01 |            35.22 |            10.58 |            54.2  |                  36.15 |
| Python/JAX        | Inference |             102.64 |               332.86 |            30.21 |            12.83 |            56.96 |                  29.32 |
| Python/JAX        | Training  |             181.58 |               411.87 |            42.43 |            10.59 |            46.99 |                  51.88 |
| Python/PyTorch    | Inference |             110.44 |               341.03 |            32.28 |            12.45 |            55.27 |                  31.55 |
| Python/PyTorch    | Training  |             179.37 |               410.58 |            44.78 |            10.15 |            45.08 |                  51.25 |
| Python/TensorFlow | Inference |              97.33 |               328.42 |            29.6  |            12.94 |            57.46 |                  27.81 |
| Python/TensorFlow | Training  |             113.24 |               344.47 |            32.85 |            12.34 |            54.82 |                  32.35 |
| R/torch           | Inference |              91.98 |               134.37 |            68.45 |            31.52 |             0.03 |                  26.28 |
| R/torch           | Training  |              95.32 |               137.68 |            69.16 |            30.81 |             0.03 |                  27.23 |
| Rust/tch          | Inference |             141.27 |               372.07 |            37.39 |            11.51 |            51.1  |                  40.36 |
| Rust/tch          | Training  |             131.85 |               362.34 |            35.98 |            11.77 |            52.25 |                  37.67 |
