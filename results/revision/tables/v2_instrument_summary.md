**Campaign-wide agreement between the two instruments**

| quantity                                         |   mean_ratio_or_share |     sd |    p05 |     p95 |
|:-------------------------------------------------|----------------------:|-------:|-------:|--------:|
| GPU (NVML counter vs CodeCarbon pynvml sampling) |                1.0044 | 0.0362 | 1      |  1.0063 |
| CPU package (RAPL counter vs CodeCarbon)         |                1.0072 | 0.0102 | 1.0001 |  1.03   |
| GPU + CPU, the measured part                     |                1.0048 | 0.0259 | 1      |  1.0109 |
| CodeCarbon total incl. modelled RAM              |                1.0842 | 0.0386 | 1.0504 |  1.1305 |
| RAM share of the CodeCarbon total                |                7.2654 | 2.245  | 4.7856 | 10.6252 |
