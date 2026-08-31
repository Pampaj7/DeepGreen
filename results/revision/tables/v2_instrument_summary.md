**Campaign-wide agreement between the two instruments**

| quantity                                                   |   mean_ratio_or_share |     sd |    p05 |     p95 |   campaign_weighted |
|:-----------------------------------------------------------|----------------------:|-------:|-------:|--------:|--------------------:|
| GPU, ratio (both read the NVML energy register)            |                1.0029 | 0.0242 | 1      |  1.0038 |             1.00032 |
| CPU package, ratio (both read RAPL energy_uj)              |                1.0052 | 0.0073 | 1.0001 |  1.0218 |             1.00063 |
| GPU + CPU, ratio (the part both read)                      |                1.0034 | 0.0182 | 1      |  1.0073 |             1.00039 |
| CodeCarbon total over counters, ratio (incl. modelled RAM) |                1.0901 | 0.0307 | 1.0505 |  1.1281 |             1.08349 |
| RAM share of the CodeCarbon total (per cent, not a ratio)  |                7.9094 | 2.0104 | 4.784  | 10.5172 |             7.6696  |
