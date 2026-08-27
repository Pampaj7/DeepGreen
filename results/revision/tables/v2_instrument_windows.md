**CodeCarbon window vs counter window, per ecosystem and phase**

| ecosystem         | phase     |   n |     mean |      sd |    worst |
|:------------------|:----------|----:|---------:|--------:|---------:|
| Cpp/LibTorch      | Inference | 900 |  809.676 | 386.176 | 1873.58  |
| Cpp/LibTorch      | Training  | 900 |   58.335 |  45.717 |  157.941 |
| Java/DL4J         | Inference | 900 |  160.87  |  39.646 |  255.732 |
| Java/DL4J         | Training  | 900 |    0.029 |   0.009 |    0.05  |
| Python/JAX        | Inference | 900 | 1027.41  | 462.22  | 2210.47  |
| Python/JAX        | Training  | 900 |  113.94  |  95.879 |  313.507 |
| Python/PyTorch    | Inference | 900 |  362.142 |  97.164 |  493.948 |
| Python/PyTorch    | Training  | 900 |    8.263 |  14.561 |   53.701 |
| Python/TensorFlow | Inference | 900 |  923.558 | 417.323 | 2139.21  |
| Python/TensorFlow | Training  | 900 |   64.002 |  53.744 |  160.22  |
| R/torch           | Inference | 900 |    0.113 |   0.018 |    0.197 |
| R/torch           | Training  | 900 |    0.024 |   0.008 |    0.063 |
| Rust/tch          | Inference | 900 |  428.778 | 232.105 |  837.02  |
| Rust/tch          | Training  | 900 |   24.987 |  36.009 |  100.541 |
