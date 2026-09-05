**Accelerator utilisation per ecosystem and architecture; the number the energy tables do not carry**

| ecosystem         | model    |   n_runs |   util_mean_pct |   util_min_pct |   util_max_pct |   mem_min_mib |   mem_max_mib |   power_min_w |   power_max_w |
|:------------------|:---------|---------:|----------------:|---------------:|---------------:|--------------:|--------------:|--------------:|--------------:|
| C++/LibTorch      | resnet18 |       11 |            32.2 |           23.7 |           43.7 |           792 |           797 |         165   |         196.9 |
| C++/LibTorch      | vgg16    |       12 |            50.2 |           39.5 |           63.2 |          1093 |          1101 |         221.6 |         275.8 |
| Java/DL4J         | resnet18 |       11 |            36.8 |           35.5 |           38.1 |          8622 |         11282 |         188.4 |         195.1 |
| Java/DL4J         | vgg16    |       11 |            79.9 |           77.2 |           83.2 |          3785 |          3995 |         317.3 |         327.9 |
| Python/JAX        | resnet18 |       11 |            13.8 |           10.1 |           18.1 |         18479 |         18580 |         134.8 |         148.4 |
| Python/JAX        | vgg16    |       15 |            43.5 |           33.2 |           55.6 |         18529 |         18576 |         203.7 |         256.9 |
| Python/PyTorch    | resnet18 |       10 |            23.5 |           16.8 |           30.7 |           790 |           794 |         155.5 |         181.5 |
| Python/PyTorch    | vgg16    |       12 |            50.3 |           37.1 |           69.5 |          1089 |          1096 |         217.6 |         294.5 |
| Python/TensorFlow | resnet18 |       11 |            20.4 |           14.9 |           28.5 |         22503 |         22567 |         160.9 |         194   |
| Python/TensorFlow | vgg16    |       12 |            42.3 |           32   |           55.4 |         22497 |         22607 |         203.2 |         265.6 |
| R/torch           | resnet18 |       10 |             4.7 |            4.5 |            5   |          1271 |          1305 |         125.8 |         127.9 |
| R/torch           | vgg16    |        9 |            12.7 |           12.1 |           13.4 |          2854 |          3001 |         138.4 |         140.6 |
| Rust/tch          | resnet18 |       11 |            27.7 |           19.9 |           35   |           797 |           800 |         156.8 |         188.2 |
| Rust/tch          | vgg16    |       11 |            44.6 |           33.8 |           54   |          1095 |          1104 |         211.9 |         262.3 |
