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
| Python/TensorFlow | ResNet-18 written against torchvision's definition layer for layer; VGG-16 backbone from `tf_keras.applications` with the shared head. No pretrained weights (TensorFlow **2.19.1**, `tf_keras` 2.19 throughout; see S5 on device placement) |
| Python/JAX | flaxmodels ResNet-18 / VGG16 backbone with the shared head, no pretrained weights |
| Java/DL4J | `ResNet18GraphBuilder` / `Vgg16GraphBuilder` |

Verified in both non-Python LibTorch bindings:
`cargo run --example load_shared_module` loads all six modules into Rust/tch,
runs a forward pass and takes an optimizer step through the VarStore;
`Rscript R/scripts/load_shared_module_test.r` loads and forwards all six in
R/torch 0.17.0.
The parameter counts match the Python export exactly (ResNet-18/CIFAR-100:
11,227,812), so the graph is identical, not merely equivalent.

**R cannot load the shared module.** In R `torch` 0.17.0 a `script_module`'s
`$train()` and `$eval()` raise `unused argument (base::quote(TRUE))`, and the
underlying handle is not reachable through the object's documented fields, so a
loaded module cannot be switched between training and evaluation. The shared
modules are exported in training mode, so this stack would evaluate with batch
normalisation using batch statistics — precisely the defect found in the
TensorFlow and Rust stacks. It therefore builds its own model from
torchvision-for-R, to the same definition. That is a property of the binding and
belongs in the paper's threats to validity, not something to paper over.

Rationale: the four LibTorch stacks are the study's internal control group. If
they train *the same traced module*, any difference between them is binding,
runtime and data-pipeline overhead and nothing else — which is precisely the
claim the paper wants to make. The TensorFlow, JAX and DL4J stacks cannot share
that module and are compared at architecture level only, which must be stated.

No pretrained weights anywhere. **Weight initialisation is no longer each
framework's default**, which was a residual difference large enough to decide
results: Deeplearning4j's was 4.6× wider than torchvision's on ResNet-18's stem
and flax's 3.3×, and in a controlled experiment the initialiser is what decides
whether VGG-16 collapses to chance at this learning rate — 0 of 6 runs under He,
2 of 6 under Glorot, 4 of 6 under Xavier. Every stack now uses torchvision's, per
S4.

**Architecture parity is proved rather than asserted.**
`scripts/verify_architecture_parity.py` builds each stack's model for every
(architecture, dataset) and compares the sorted multiset of parameter tensor
shapes, which is comparable across languages. All seven stacks agree shape for
shape on all six blocks: ResNet-18 is 62 tensors (20 convolution, 1 dense, 41
rank-1) and VGG-16 is 30 (13, 2, 15). Every stack also asserts its own parameter
count at startup against `models/MANIFEST.json`, carried into the run as
`DEEPGREEN_EXPECTED_PARAMS`, and refuses to train if it does not match.

R was the exception until the probe was run in the campaign's own environment.
It had nothing to do with the binding: R's `torch` loads `liblantern.so`, which
links `libcudart.so.12` under a name the R bundle hashes, so without a CUDA 12
runtime on `LD_LIBRARY_PATH` the package reports "Lantern is not loaded" and
every probe fails. With `tools/stack_environments.json`'s own environment R
fingerprints like the others.

## Cross-stack validation

With S1-S5 applied, the three Python ecosystems were run for one epoch on
Fashion-MNIST under identical settings (ResNet-18, Adam 1e-4, batch 128,
32x32x3 in [0,1], 2 loader threads, fp32, seed 109x):

| ecosystem | test loss | test accuracy | training J |
|---|---|---|---|
| C++/LibTorch | 0.3639 | 86.47% | 1,139 |
| Rust/tch | 0.3687 | 86.58% | 1,387 |
| Python/PyTorch | 0.3541 | 86.82% | 1,434 |
| Python/JAX | 0.3478 | 87.19% | 2,517 |
| Python/TensorFlow | 0.3609 | 86.57% | 2,764 |
| Java/DL4J | 0.3595 | 86.52% | 8,987 |
| R/torch | 0.3431 | 87.47% | 11,562 |

A spread of **1.00 percentage point** in accuracy and 0.026 in test loss, across
seven implementations reaching the same place while consuming energy over a
10.2× range. That is the shape the comparison has to have: the quality is held
constant and the cost is what varies.

Before this round's alignment the same table spanned 10.34 points, and the
difference was six defects, not six ecosystems -- see `REVISION_LOG.md`.

The first campaign could not perform this check at all, because no quality
metric was written to disk, which is how TensorFlow came to evaluate with its
backbone pinned to training mode without anyone noticing. Any future change to a
stack should be followed by re-running this comparison.

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
| input resolution | 32 x 32, **resized once offline** so that no stack resizes anything |
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

**Resizing happens once, on disk.** The seven ecosystems used four resamplers --
torchvision on PIL, `tf.image.resize`, `tch::vision::image::resize`, DataVec's
`ImageRecordReader`, R's `transform_resize` -- and over Tiny ImageNet's whole
test split their pixel standard deviations fell into three groups spanning 3.0%
while the means agreed to 0.1%, which is a filter and not a content difference.
`scripts/normalise_dataset_resolution.py` writes every image at 32x32, so each
stack's resize is a no-op and all seven decode the same pixels: measured, mean
0.443782 and standard deviation 0.256110 over 30,720,000 values, identical in
all seven. Every run records its own (`data_fingerprint.csv`), so the campaign
demonstrates this rather than the manuscript asserting it.

## S4 — Backend

| stack | requirement |
|---|---|
| all LibTorch stacks | one LibTorch build, one CUDA, one cuDNN |
| all stacks | storage precision fp32, no mixed precision; one TF32 policy set campaign-wide by `DEEPGREEN_TF32` and applied through `NVIDIA_TF32_OVERRIDE` |
| all stacks | one initialiser: `kaiming_normal_(fan_out, relu)` for convolutions, `nn.Linear`'s uniform ±1/√fan_in for dense layers |
| all stacks | one batch-normalisation policy: torch's momentum 0.1 and eps 1e-5, expressed in each framework's own convention |
| all stacks | cuDNN algorithm selection left to the framework — no stack pins determinism |
| all stacks | per-epoch record: train loss, test loss, test accuracy. **Train accuracy where the stack computes it** — two of seven do, it is read by no analysis, and the schema says so rather than implying seven |
| Java/DL4J | **does not use cuDNN at all**, and cannot at this version |

Deeplearning4j 1.0.0-M2.1 — the latest release — resolves its convolutions
through nd4j's own im2col and cuBLAS. Four independent confirmations:

  * `readelf -d` on `libnd4jcuda.so` shows a `NEEDED` entry for
    `libcublas.so.11` and **none for cuDNN**;
  * `org.deeplearning4j.cuda.convolution.CudnnConvolutionHelper` is absent from
    the classpath, as is any `org.bytedeco` cuDNN preset;
  * `deeplearning4j-cuda` and `deeplearning4j-cuda-11.6` have **no versions at
    all** in Maven Central's metadata, with `deeplearning4j-modelimport` as a
    control showing the endpoint answers;
  * the run log reads `Loaded [JCublasBackend]` with no convolution-helper line.

It is the only stack of the seven without cuDNN, and one of the two most
expensive. That is a property of the framework at this version and cannot be
corrected here, so it is stated — and **bounded**, because "Java/DL4J costs 6x"
otherwise reads as a claim about a language when part of it is a claim about a
missing convolution backend.

**The bound.** Disabling cuDNN in a stack that has it, holding hardware, model
and data constant — ResNet-18, batch 128, RTX 3090, median of three — costs
**13.3× the energy and 16.7× the time**: 100.3 J and 0.353 s against 1333.2 J
and 5.874 s. That is an upper bound on what losing cuDNN can cost, not an
estimate of Deeplearning4j's own penalty, since nd4j's im2col path is not
PyTorch's non-cuDNN fallback. It exceeds the entire Java-versus-cheapest gap the
campaign measures, which is the point: the missing backend is a first-order
explanation of this stack's cost, not a footnote to it.

`scripts/check_consistency.py` watches `libnd4jcuda.so` and fails if a future
version gains cuDNN, so the claim cannot quietly stop being true.

This clause used to read "precision pinned to FP32; TF32 matmuls disabled on
Ampere and later", and exactly one stack of seven obeyed it. The pin lived in
`tools/deepgreen_bench.py`, which only the three Python stacks import, and of
those only PyTorch reached cuDNN through it — TensorFlow was given a
`mixed_precision` policy, which is a different axis, and JAX was given nothing.
So Python/PyTorch ran true-fp32 convolutions and the other six took cuDNN's
Ampere default. On an RTX 3090, ResNet-18, batch 128, that setting alone is
**4.81× the GPU energy and 3.46× the step time**, against a measured
PyTorch-vs-C++ training gap of 3.24–3.79×. The clause was in the specification;
no check stood behind it; and the difference was reported as binding overhead.

Two changes. The policy is now one variable read by every stack, because a
requirement that only one code path can express is a requirement four stacks
cannot meet. And the default is TF32 *allowed* rather than disabled: it is what
every framework does on Ampere without being asked, so it is what a practitioner
measures, and pinning true fp32 across seven stacks costs roughly 138 h of
machine time against 57 h for the same 210 runs. The cost of the policy itself
is reported as a precision ablation rather than hidden inside an ecosystem
comparison. Set `DEEPGREEN_TF32=0` to take the other branch.

`cudnn.deterministic` was the other half of the same asymmetry — pinned for the
Python stacks alone, worth a further 1.35×, and not expressible in DL4J or R at
all. It is pinned nowhere now, so algorithm selection varies the same way for
every stack and run-to-run variation includes it.

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
