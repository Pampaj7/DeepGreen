**Training protocol per ecosystem, read from source (first campaign)**

| ecosystem         | optimiser    |   learning_rate |   batch_size | loader_mechanism           |   loader_threads | input_scaling                | source                                                      |
|:------------------|:-------------|----------------:|-------------:|:---------------------------|-----------------:|:-----------------------------|:------------------------------------------------------------|
| Python/PyTorch    | Adam         |          0.0001 |          128 | DataLoader workers         |                2 | [0,1] (ToTensor)             | python/pytorch/models/resnet18.py:42                        |
| Python/TensorFlow | Adam         |          0.001  |          128 | Keras generator            |                1 | [0,1] (rescale 1/255)        | python/tensorflow/models/resnet18.py (lr default)           |
| Python/JAX        | Adam (optax) |          0.0001 |          128 | Keras generator            |                1 | [0,1] (rescale 1/255)        | python/jax/models/resnet18.py:20                            |
| C++/LibTorch      | Adam         |          0.0001 |          128 | DataLoader workers         |                2 | [0,1]                        | cpp/src/train/native/train_model.h:64                       |
| Java/DL4J         | Adam         |          0.0001 |          128 | AsyncDataSetIterator       |                2 | [0,1]                        | …/dataloader/PNGDataloader.java:32                          |
| R/torch           | Adam         |          0.0001 |          128 | dataloader workers         |                0 | [0,1]                        | R/models/resnet18.r:75                                      |
| Rust/tch          | Adam         |          0.001  |          128 | rayon par_iter (all cores) |               96 | mean/std normalised          | rust/src/bin/resnet_cifar100.rs:33, datasets/cifar100.rs:62 |
| MATLAB/DLT        | adam         |          0.0001 |          128 | augmentedImageDatastore    |                1 | [0,1] (Normalization='none') | matlab/train/+resnet18/train_cifar100.m:13                  |
