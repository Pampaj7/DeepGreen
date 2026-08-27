**Training: pairwise Mann-Whitney with Holm correction and Cliff's delta on harmonised energy (epoch-level pseudo-replicates)**

| phase    | a                 | b                 |   median_a_J |   median_b_J |   ratio_b_over_a |     U |    p_raw |   cliffs_delta | magnitude   |   p_holm | significant_0.05   |
|:---------|:------------------|:------------------|-------------:|-------------:|-----------------:|------:|---------:|---------------:|:------------|---------:|:-------------------|
| Training | C++/LibTorch      | Java/DL4J         |      3760.56 |     12761.4  |         3.39349  |     0 | 0        |      -1        | large       | 0        | True               |
| Training | C++/LibTorch      | MATLAB/DLT        |      3760.56 |      7531.03 |         2.00263  |  6975 | 0        |      -0.569444 | large       | 0        | True               |
| Training | C++/LibTorch      | Python/JAX        |      3760.56 |      2690.54 |         0.715462 | 16829 | 0.524382 |       0.038827 | negligible  | 1        | False              |
| Training | C++/LibTorch      | Python/PyTorch    |      3760.56 |      3590.28 |         0.95472  | 18857 | 0.007129 |       0.164012 | small       | 0.028516 | True               |
| Training | C++/LibTorch      | Python/TensorFlow |      3760.56 |      6644.4  |         1.76686  |  4285 | 0        |      -0.735494 | large       | 0        | True               |
| Training | C++/LibTorch      | R/torch           |      3760.56 |     18937.8  |         5.03591  |     0 | 0        |      -1        | large       | 0        | True               |
| Training | C++/LibTorch      | Rust/tch          |      3760.56 |      1468.35 |         0.39046  | 26158 | 0        |       0.614691 | large       | 0        | True               |
| Training | Java/DL4J         | MATLAB/DLT        |     12761.4  |      7531.03 |         0.590141 | 28790 | 0        |       0.77716  | large       | 0        | True               |
| Training | Java/DL4J         | Python/JAX        |     12761.4  |      2690.54 |         0.210834 | 32328 | 0        |       0.995556 | large       | 0        | True               |
| Training | Java/DL4J         | Python/PyTorch    |     12761.4  |      3590.28 |         0.281339 | 32400 | 0        |       1        | large       | 0        | True               |
| Training | Java/DL4J         | Python/TensorFlow |     12761.4  |      6644.4  |         0.520663 | 25148 | 0        |       0.552346 | large       | 0        | True               |
| Training | Java/DL4J         | R/torch           |     12761.4  |     18937.8  |         1.48399  |  8201 | 0        |      -0.493765 | large       | 0        | True               |
| Training | Java/DL4J         | Rust/tch          |     12761.4  |      1468.35 |         0.115062 | 32400 | 0        |       1        | large       | 0        | True               |
| Training | MATLAB/DLT        | Python/JAX        |      7531.03 |      2690.54 |         0.357261 | 28380 | 0        |       0.751852 | large       | 0        | True               |
| Training | MATLAB/DLT        | Python/PyTorch    |      7531.03 |      3590.28 |         0.476732 | 27224 | 0        |       0.680494 | large       | 0        | True               |
| Training | MATLAB/DLT        | Python/TensorFlow |      7531.03 |      6644.4  |         0.88227  | 15890 | 0.753908 |      -0.019136 | negligible  | 1        | False              |
| Training | MATLAB/DLT        | R/torch           |      7531.03 |     18937.8  |         2.51464  |  1822 | 0        |      -0.887531 | large       | 0        | True               |
| Training | MATLAB/DLT        | Rust/tch          |      7531.03 |      1468.35 |         0.194973 | 29746 | 0        |       0.836173 | large       | 0        | True               |
| Training | Python/JAX        | Python/PyTorch    |      2690.54 |      3590.28 |         1.33441  | 16475 | 0.780982 |       0.016975 | negligible  | 1        | False              |
| Training | Python/JAX        | Python/TensorFlow |      2690.54 |      6644.4  |         2.46954  |  3477 | 0        |      -0.78537  | large       | 0        | True               |
| Training | Python/JAX        | R/torch           |      2690.54 |     18937.8  |         7.03868  |     0 | 0        |      -1        | large       | 0        | True               |
| Training | Python/JAX        | Rust/tch          |      2690.54 |      1468.35 |         0.545745 | 25437 | 0        |       0.570185 | large       | 0        | True               |
| Training | Python/PyTorch    | Python/TensorFlow |      3590.28 |      6644.4  |         1.85066  |  3574 | 0        |      -0.779383 | large       | 0        | True               |
| Training | Python/PyTorch    | R/torch           |      3590.28 |     18937.8  |         5.27475  |     0 | 0        |      -1        | large       | 0        | True               |
| Training | Python/PyTorch    | Rust/tch          |      3590.28 |      1468.35 |         0.408979 | 24621 | 0        |       0.519815 | large       | 0        | True               |
| Training | Python/TensorFlow | R/torch           |      6644.4  |     18937.8  |         2.8502   |  3093 | 0        |      -0.809074 | large       | 0        | True               |
| Training | Python/TensorFlow | Rust/tch          |      6644.4  |      1468.35 |         0.220991 | 31739 | 0        |       0.959198 | large       | 0        | True               |
| Training | R/torch           | Rust/tch          |     18937.8  |      1468.35 |         0.077535 | 32400 | 0        |       1        | large       | 0        | True               |
