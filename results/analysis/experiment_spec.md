# Experiment specification

One document that every ecosystem must conform to. It exists because the first
campaign did not have one: reading the eight stacks found different learning
rates, different input scaling, different loader parallelism, different model
implementations and different measurement instruments — while the manuscript
stated a common protocol.

`scripts/check_consistency.py` verifies the repository against this file.
Anything it cannot verify automatically is listed under **manual** below and
must be checked on the measurement machine.

Scope: seven ecosystems. MATLAB/DLT is out (proprietary toolbox, no license on
the replication machine, and it cannot be pinned to the shared LibTorch build).

## S1 — Model

| stack family | model source |
|---|---|
| Python/PyTorch | `torchvision.models.resnet18` / `vgg16`, final layer resized to `num_classes` |
| C++/LibTorch, Rust/tch | **the same module**, exported once by `scripts/export_torchscript_models.py` and loaded with `torch::jit::load` / `TrainableCModule::load` |
| R/torch | **cannot share it** — see below. Builds `torchvision::model_resnet18` / `model_vgg16`. |
| Python/TensorFlow | Model Garden ResNet-18 / Keras VGG16, no pretrained weights (TensorFlow **2.19.1**; see S5 on device placement) |
| Python/JAX | flaxmodels ResNet-18 / VGG16, no pretrained weights |
| Java/DL4J | `ResNet18GraphBuilder` / `Vgg16GraphBuilder` |

Verified in both non-Python LibTorch bindings:
`cargo run --example load_shared_module` loads all six modules into Rust/tch,
runs a forward pass and takes an optimizer step through the VarStore;
`Rscript R/scripts/load_shared_module_test.r` loads and forwards all six in
R/torch 0.17.0.
The parameter counts match the Python export exactly (ResNet-18/CIFAR-100:
11,227,812), so the graph is identical, not merely equivalent.

**R is a documented exception.** In R `torch` 0.17.0 a `script_module`'s
`$train()` and `$eval()` raise `unused argument (base::quote(TRUE))`, and the
underlying handle is not reachable through the object's documented fields, so a
loaded module cannot be switched between training and evaluation. The shared
modules are exported in training mode, so this stack would evaluate with batch
normalisation using batch statistics — precisely the defect found in the
TensorFlow and Rust stacks. It therefore builds its own model, and the
architecture parity that holds for Python, C++ and Rust does not hold for R.
That is a property of the binding and belongs in the paper's threats to
validity, not something to paper over.

Rationale: the four LibTorch stacks are the study's internal control group. If
they train *the same traced module*, any difference between them is binding,
runtime and data-pipeline overhead and nothing else — which is precisely the
claim the paper wants to make. The TensorFlow, JAX and DL4J stacks cannot share
that module and are compared at architecture level only, which must be stated.

No pretrained weights anywhere. Weight initialisation is each framework's
default and is a documented residual difference.

## Cross-stack validation

With S1-S5 applied, the three Python ecosystems were run for one epoch on
Fashion-MNIST under identical settings (ResNet-18, Adam 1e-4, batch 128,
32x32x3 in [0,1], 2 loader threads, fp32, seed 109x):

| ecosystem | train loss | test accuracy |
|---|---|---|
| Python/PyTorch | 0.4426 | 87.11% |
| Python/JAX | 0.4485 | 86.46% |
| Python/TensorFlow | 0.4872 | 86.06% |

Agreement to within one percentage point is the evidence that the stacks now
solve the same problem. The first campaign could not perform this check at all,
because no quality metric was written to disk -- which is how TensorFlow came to
evaluate with its backbone pinned to training mode without anyone noticing.

Any future change to a stack should be followed by re-running this comparison.

## S2 — Optimisation

| parameter | value |
|---|---|
| optimiser | Adam |
| learning rate | `1e-4` |
| batch size | 128 |
| epochs | 30 |
| loss | categorical cross-entropy over logits |
| LR schedule | none |
| weight decay | none (framework default of 0) |

First campaign: Python/TensorFlow and Rust/tch used `1e-3`. Fixed.

## S3 — Data pipeline

| parameter | value |
|---|---|
| input resolution | 32 x 32 |
| channels | 3 (grayscale replicated) |
| scaling | `[0, 1]` — divide by 255, **no** mean/std normalisation |
| train shuffling | on |
| eval shuffling | off |
| loader worker threads | **2** |
| dataset format | PNG folders produced by `dataloader/*.py` |

Loader parallelism is the single largest confound found in the first campaign:
it ranged from 0 (R) to 96 (Rust, rayon over all cores) and correlates with
epoch duration at Spearman -0.73 (p = 0.04). Since the GPU runs at 24-56% of its
power limit, the pipeline dominates the measurement. It is now a controlled
factor, not a framework default.

First campaign: only Rust normalised with per-channel mean/std. Fixed; set
`DEEPGREEN_NORMALIZE=1` to restore the old behaviour deliberately.

## S4 — Backend

| stack | requirement |
|---|---|
| all LibTorch stacks | one LibTorch build, one CUDA, one cuDNN |
| all stacks | precision pinned to FP32; TF32 matmuls disabled on Ampere and later |

First campaign: Python/PyTorch on LibTorch 2.6.0, C++ on 2.7.0, Rust/tch 0.14 on
2.1.0, R/torch 0.15.1 on its own bundle — six minor releases across the group
the paper treats as sharing a backend. CUDA ranged from 11.6 to 12.8.

## S5 — Measurement

**Primary instrument: hardware counters.** Energy is taken from
`nvmlDeviceGetTotalEnergyConsumption` for the accelerator and Intel RAPL for the
CPU package. These are counters integrated in hardware and read through software,
not estimates: there is no sampling interval to choose and no power model to be
wrong about. CodeCarbon runs over the same window as a second instrument, so the
two can be compared per block rather than one being trusted.

**The boundary is the chip, and we say so.** NVML plus RAPL covers the
accelerator and the CPU package. It excludes the power supply, fans, drives and
everything else a wall meter would see. That is a narrower boundary than "the
energy this computation cost", and reporting it as such is the point: the figure
the campaign under audit reports sits between two boundaries and corresponds to
neither, because roughly two thirds of it, for four ecosystems, was a model of
RAM power derived from installed memory.

We do not have a wall meter. Published validations put software estimators within
tens of percent of one (Fischer 2025 reports errors up to 40% against external
meters, and Khan et al. 2018 find RAPL highly correlated with plug power), so a
chip-boundary figure should not be presented as a whole-system one.

### Legacy note

| parameter | value |
|---|---|
| primary | NVML energy counter (GPU) + RAPL (CPU package) |
| cross-check | CodeCarbon `>= 3.0`, same window |
| `tracking_mode` | `machine` |
| `measure_power_secs` | 1 |
| tracked block | exactly one epoch of one phase |
| recorded per epoch | `train_loss`, `test_loss`, `test_acc` |
| device placement | asserted at startup and recorded in `manifest.json` |
| host state | machine otherwise idle; idle baseline recorded (see protocol section 7) |

**Every ecosystem must be proven to run on the accelerator.** A framework whose
build does not cover the card falls back to the CPU and says so only in a
warning; the run then completes normally and the CPU cost is attributed to the
ecosystem. `tools/deepgreen_bench.py` refuses to start in that state
(`DEEPGREEN_ALLOW_CPU=1` to override deliberately). This is not hypothetical: on this machine TensorFlow 2.21 resolves CUDA 12.9
wheels it cannot load against a binary built for 12.5.1, and runs the entire
workload on the CPU; TensorFlow 2.19.1 on the same card and driver uses the GPU.
The pinned version therefore belongs in the specification, and the device must be
asserted rather than assumed.

First campaign: two CodeCarbon major versions, `process` mode for R only, 15 s
sampling everywhere except JAX at 1 s, and no quality metric on disk. Fixed by
`tools/deepgreen_bench.py` and `tools/codecarbon_config.json`.

## S6 — Replication

* 5 independent repetitions per configuration, distinct seeds, fresh processes;
* repetition-major interleaving with a fixed shuffle, 60 s cooldown;
* the independent run is the unit of analysis.

## Environment requirements found the hard way

| stack | requirement |
|---|---|
| R/torch | `liblantern.so` links against `libcudart.so.12`, but the R bundle ships it under the hashed name `libcudart-c3a75b33.so.12`. A CUDA 12 runtime must be on `LD_LIBRARY_PATH` or the package loads as "Lantern is not loaded". |
| R/torch | the `png` package (a `torchvision` dependency) needs **libpng and zlib** development libraries; without zlib it fails at link with `cannot find -lz`. |
| Rust/tch | must be built with `-Wl,--no-as-needed -ltorch_cuda -lc10_cuda`, otherwise the linker drops the CUDA dependency and the binary runs on the CPU while reporting nothing. `scripts/build_rust_cuda.sh`. |
| C++/LibTorch | `CUDA::nvToolsExt` must be defined **before** `find_package(Torch)`; NVTX left the CUDA toolkit in CUDA 12. cuDNN must be present, and `CUDA_TOOLKIT_ROOT_DIR` must point at `targets/x86_64-linux`. |
| C++/LibTorch | the embedded interpreter resolves `site-packages` from whichever libpython it was linked against, not from the campaign environment; the bridge adds the contract interpreter's packages explicitly. |
| Java/DL4J | DL4J 1.0.0-M2.1 links against the **CUDA 11.6** runtime and is the last release of that API — there is no CUDA 12 backend. On a CUDA 12/13 host the backend fails with `libcudart.so.11.0: cannot open shared object file`. The pom now pulls `org.bytedeco:cuda-platform-redist` so the stack carries its own runtime. |
| all | the measurement machine must be otherwise idle (protocol section 7). |

## Manual checks

These cannot be verified by reading the repository and must be confirmed on the
measurement machine before the campaign:

1. resolved CUDA / cuDNN / LibTorch versions per stack (`manifest.json`);
2. effective concurrency of the Keras generators used by TensorFlow and JAX —
   they have no `num_workers` equivalent, so their pipeline parallelism must be
   measured, not assumed;
3. the daemon-based CodeCarbon controller used by the non-Python stacks,
   validated against in-process CodeCarbon and an external meter;
4. GPU utilisation per configuration, reported alongside energy.
