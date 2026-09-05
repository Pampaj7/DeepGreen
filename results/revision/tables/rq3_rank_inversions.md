**Agreement between the energy ranking and the time ranking, FIRST SUBMISSION campaign (results/data/). Not read by the manuscript**

| phase     | energy_definition                                     |   spearman_rho |   kendall_tau |   discordant_pairs |   total_pairs | examples                                                                      |
|:----------|:------------------------------------------------------|---------------:|--------------:|-------------------:|--------------:|:------------------------------------------------------------------------------|
| Inference | as measured (CodeCarbon total, instrument-confounded) |       0.904762 |      0.785714 |                  3 |            28 | Java/DL4J vs Rust/tch; MATLAB/DLT vs Python/JAX; MATLAB/DLT vs Python/PyTorch |
| Inference | GPU only (NVML counter, identical instrument)         |       0.97619  |      0.928571 |                  1 |            28 | Java/DL4J vs Python/TensorFlow                                                |
| Inference | harmonised (GPU + 107 W uniform host model)           |       1        |      1        |                  0 |            28 |                                                                               |
| Training  | as measured (CodeCarbon total, instrument-confounded) |       0.952381 |      0.857143 |                  2 |            28 | C++/LibTorch vs Python/PyTorch; Java/DL4J vs R/torch                          |
| Training  | GPU only (NVML counter, identical instrument)         |       0.928571 |      0.857143 |                  2 |            28 | C++/LibTorch vs Python/JAX; Python/JAX vs Python/PyTorch                      |
| Training  | harmonised (GPU + 107 W uniform host model)           |       0.97619  |      0.928571 |                  1 |            28 | C++/LibTorch vs Python/JAX                                                    |
