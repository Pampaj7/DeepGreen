**Campaign-wide agreement between the two instruments**

| quantity                                         |   mean_ratio_or_share |     sd |    p05 |     p95 |
|:-------------------------------------------------|----------------------:|-------:|-------:|--------:|
| GPU (NVML counter vs CodeCarbon pynvml sampling) |                1.0043 | 0.0351 | 1      |  1.0063 |
| CPU package (RAPL counter vs CodeCarbon)         |                1.007  | 0.0099 | 1.0001 |  1.0293 |
| GPU + CPU, the measured part                     |                1.0048 | 0.0252 | 1      |  1.0109 |
| CodeCarbon total incl. modelled RAM              |                1.0843 | 0.0379 | 1.0504 |  1.1302 |
| RAM share of the CodeCarbon total                |                7.2799 | 2.2217 | 4.7904 | 10.5167 |
