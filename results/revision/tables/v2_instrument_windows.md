**CodeCarbon window vs counter window, per ecosystem and phase**

| ecosystem         | phase     |   n |     mean |      sd |    worst |
|:------------------|:----------|----:|---------:|--------:|---------:|
| Cpp/LibTorch      | Inference | 270 | 1145.5   | 212.663 | 1592.3   |
| Cpp/LibTorch      | Training  | 270 |   78.321 |  27.872 |  132.167 |
| Java/DL4J         | Inference | 270 |   76.802 |  24.479 |  115.551 |
| Java/DL4J         | Training  | 270 |    0.03  |   0.009 |    0.059 |
| Python/JAX        | Inference | 180 |  494.015 | 104.715 |  630.538 |
| Python/JAX        | Training  | 180 |   75.65  |  25.042 |  112.556 |
| Python/PyTorch    | Inference | 240 |  408.346 |  64.302 |  494.883 |
| Python/PyTorch    | Training  | 240 |   61.362 |  31.025 |  107.656 |
| Python/TensorFlow | Inference | 240 |  526.6   |  83.965 |  630.188 |
| Python/TensorFlow | Training  | 240 |   68.008 |  30.592 |  117.165 |
| R/torch           | Inference | 330 |    0.115 |   0.017 |    0.18  |
| R/torch           | Training  | 330 |    0.026 |   0.009 |    0.061 |
| Rust/tch          | Inference | 240 |  702.364 | 240.12  | 1236.83  |
| Rust/tch          | Training  | 240 |   53.352 |  36.968 |  102.828 |
