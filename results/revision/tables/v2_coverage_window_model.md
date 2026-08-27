**Competing models of CodeCarbon's reported duration**

| model                                      |   r_squared |   mean_abs_error_s |
|:-------------------------------------------|------------:|-------------------:|
| max(phase, 3.99 s)                         |      0.9926 |              1.128 |
| phase + 3.28 s                             |      0.9909 |              1.329 |
| phase + 3.28 s if phase < 11 s, else phase |      0.998  |              0.396 |
