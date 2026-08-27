**Inference: pairwise Mann-Whitney with Holm correction and Cliff's delta on harmonised energy (epoch-level pseudo-replicates)**

| phase     | a                 | b                 |   median_a_J |   median_b_J |   ratio_b_over_a |     U |    p_raw |   cliffs_delta | magnitude   |   p_holm | significant_0.05   |
|:----------|:------------------|:------------------|-------------:|-------------:|-----------------:|------:|---------:|---------------:|:------------|---------:|:-------------------|
| Inference | C++/LibTorch      | Java/DL4J         |      292.362 |      897.309 |         3.06917  |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | C++/LibTorch      | MATLAB/DLT        |      292.362 |      542.667 |         1.85614  |   267 | 0        |      -0.983519 | large       | 0        | True               |
| Inference | C++/LibTorch      | Python/JAX        |      292.362 |      452.43  |         1.5475   |  3192 | 0        |      -0.802963 | large       | 0        | True               |
| Inference | C++/LibTorch      | Python/PyTorch    |      292.362 |      362.301 |         1.23922  |  5757 | 0        |      -0.64463  | large       | 0        | True               |
| Inference | C++/LibTorch      | Python/TensorFlow |      292.362 |      930.399 |         3.18235  |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | C++/LibTorch      | R/torch           |      292.362 |     2708.35  |         9.26366  |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | C++/LibTorch      | Rust/tch          |      292.362 |      809     |         2.76711  |    23 | 0        |      -0.99858  | large       | 0        | True               |
| Inference | Java/DL4J         | MATLAB/DLT        |      897.309 |      542.667 |         0.604771 | 28169 | 0        |       0.738827 | large       | 0        | True               |
| Inference | Java/DL4J         | Python/JAX        |      897.309 |      452.43  |         0.504207 | 29016 | 0        |       0.791111 | large       | 0        | True               |
| Inference | Java/DL4J         | Python/PyTorch    |      897.309 |      362.301 |         0.403764 | 32369 | 0        |       0.998086 | large       | 0        | True               |
| Inference | Java/DL4J         | Python/TensorFlow |      897.309 |      930.399 |         1.03688  | 11158 | 0        |      -0.311235 | small       | 1e-06    | True               |
| Inference | Java/DL4J         | R/torch           |      897.309 |     2708.35  |         3.0183   |   347 | 0        |      -0.97858  | large       | 0        | True               |
| Inference | Java/DL4J         | Rust/tch          |      897.309 |      809     |         0.901584 | 18735 | 0.010253 |       0.156481 | small       | 0.010253 | True               |
| Inference | MATLAB/DLT        | Python/JAX        |      542.667 |      452.43  |         0.833716 | 21798 | 0        |       0.345556 | medium      | 0        | True               |
| Inference | MATLAB/DLT        | Python/PyTorch    |      542.667 |      362.301 |         0.667631 | 27617 | 0        |       0.704753 | large       | 0        | True               |
| Inference | MATLAB/DLT        | Python/TensorFlow |      542.667 |      930.399 |         1.71449  |   377 | 0        |      -0.976728 | large       | 0        | True               |
| Inference | MATLAB/DLT        | R/torch           |      542.667 |     2708.35  |         4.99081  |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | MATLAB/DLT        | Rust/tch          |      542.667 |      809     |         1.49079  |  5173 | 0        |      -0.680679 | large       | 0        | True               |
| Inference | Python/JAX        | Python/PyTorch    |      452.43  |      362.301 |         0.80079  | 21382 | 0        |       0.319877 | small       | 0        | True               |
| Inference | Python/JAX        | Python/TensorFlow |      452.43  |      930.399 |         2.05645  |   233 | 0        |      -0.985617 | large       | 0        | True               |
| Inference | Python/JAX        | R/torch           |      452.43  |     2708.35  |         5.98623  |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | Python/JAX        | Rust/tch          |      452.43  |      809     |         1.78812  |  3802 | 0        |      -0.765309 | large       | 0        | True               |
| Inference | Python/PyTorch    | Python/TensorFlow |      362.301 |      930.399 |         2.56803  |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | Python/PyTorch    | R/torch           |      362.301 |     2708.35  |         7.4754   |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | Python/PyTorch    | Rust/tch          |      362.301 |      809     |         2.23295  |   434 | 0        |      -0.97321  | large       | 0        | True               |
| Inference | Python/TensorFlow | R/torch           |      930.399 |     2708.35  |         2.91095  |     0 | 0        |      -1        | large       | 0        | True               |
| Inference | Python/TensorFlow | Rust/tch          |      930.399 |      809     |         0.869519 | 25752 | 0        |       0.58963  | large       | 0        | True               |
| Inference | R/torch           | Rust/tch          |     2708.35  |      809     |         0.298706 | 32400 | 0        |       1        | large       | 0        | True               |
