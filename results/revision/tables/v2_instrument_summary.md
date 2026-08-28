**Campaign-wide agreement between the two instruments**

| quantity                                                   |   mean_ratio_or_share |     sd |    p05 |     p95 |   campaign_weighted |
|:-----------------------------------------------------------|----------------------:|-------:|-------:|--------:|--------------------:|
| GPU, ratio (both read the NVML energy register)            |                1.0043 | 0.0351 | 1      |  1.0063 |             1.00032 |
| CPU package, ratio (both read RAPL energy_uj)              |                1.007  | 0.0099 | 1.0001 |  1.0293 |             1.00064 |
| GPU + CPU, ratio (the part both read)                      |                1.0048 | 0.0252 | 1      |  1.0109 |             1.00038 |
| CodeCarbon total over counters, ratio (incl. modelled RAM) |                1.0843 | 0.0379 | 1.0504 |  1.1302 |             1.07669 |
| RAM share of the CodeCarbon total (per cent, not a ratio)  |                7.2799 | 2.2217 | 4.7904 | 10.5167 |             7.08748 |
