**Best/worst ecosystem and spread by metric**

| phase     | energy_definition                                     | best         | worst     |   spread_x |
|:----------|:------------------------------------------------------|:-------------|:----------|-----------:|
| Training  | as measured (CodeCarbon total, instrument-confounded) | Rust/tch     | Java/DL4J |       4.58 |
| Training  | GPU only (NVML counter, identical instrument)         | Rust/tch     | R/torch   |       8.54 |
| Training  | harmonised (GPU + 107 W uniform host model)           | Rust/tch     | R/torch   |       9.94 |
| Training  | execution time (s)                                    | Rust/tch     | R/torch   |      11.63 |
| Inference | as measured (CodeCarbon total, instrument-confounded) | C++/LibTorch | R/torch   |       7.28 |
| Inference | GPU only (NVML counter, identical instrument)         | C++/LibTorch | R/torch   |       9.3  |
| Inference | harmonised (GPU + 107 W uniform host model)           | C++/LibTorch | R/torch   |       8.83 |
| Inference | execution time (s)                                    | C++/LibTorch | R/torch   |       8.47 |
