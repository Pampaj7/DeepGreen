**LibTorch build behind each of the four LibTorch-based ecosystems**

| ecosystem      | binding                                    | libtorch                      |
|:---------------|:-------------------------------------------|:------------------------------|
| Python/PyTorch | torch 2.6.0 (requirements/pytorch_raw.txt) | 2.6.0                         |
| C++/LibTorch   | direct, fetched by CMake                   | 2.7.0+cu128                   |
| Rust/tch       | tch 0.14.0 / torch-sys 0.14.0 (Cargo.lock) | 2.1.0                         |
| R/torch        | R torch 0.15.1 (requirements/r.txt)        | bundled, version not recorded |
