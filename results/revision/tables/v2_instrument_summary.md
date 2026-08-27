**Campaign-wide agreement between the two instruments**

| quantity                                         |   mean_ratio_or_share |     sd |    p05 |     p95 |
|:-------------------------------------------------|----------------------:|-------:|-------:|--------:|
| GPU (NVML counter vs CodeCarbon pynvml sampling) |                1.0044 | 0.0353 | 1      |  1.0063 |
| CPU package (RAPL counter vs CodeCarbon)         |                1.007  | 0.0099 | 1.0001 |  1.0293 |
| GPU + CPU, the measured part                     |                1.0049 | 0.0254 | 1      |  1.011  |
| CodeCarbon total incl. modelled RAM              |                1.0844 | 0.038  | 1.0504 |  1.1303 |
| RAM share of the CodeCarbon total                |                7.2791 | 2.2214 | 4.7904 | 10.5143 |
