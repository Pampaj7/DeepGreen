#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation audit: is the comparison like-for-like?

Addresses:
  R1 c7  -- "the comparison is not strictly like-for-like at the implementation
            level". It is worse than the reviewer could see from the paper: the
            training protocol itself differs between ecosystems.
  R1 c12 -- the mechanism behind the ranking.
  R3 M3  -- the manuscript states that all ecosystems share "the same 30 epochs,
            learning rate, optimizer, and batch size". Two of the eight do not
            share the learning rate.

The table below was established by reading the source of all eight ecosystems in
the first-campaign state (git tag/commit of the submitted package). It is data,
not opinion: every cell cites the file and line it came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import load, save_table

# --------------------------------------------------------------------------
# Protocol as implemented in the FIRST campaign, read from source.
# --------------------------------------------------------------------------
PROTOCOL = pd.DataFrame(
    [
        # ecosystem, optimiser, lr, batch, loader parallelism, loader threads,
        # input scaling, source
        ("Python/PyTorch", "Adam", 1e-4, 128, "DataLoader workers", 2,
         "[0,1] (ToTensor)", "python/pytorch/models/resnet18.py:42"),
        ("Python/TensorFlow", "Adam", 1e-3, 128, "Keras generator", 1,
         "[0,1] (rescale 1/255)", "python/tensorflow/models/resnet18.py (lr default)"),
        ("Python/JAX", "Adam (optax)", 1e-4, 128, "Keras generator", 1,
         "[0,1] (rescale 1/255)", "python/jax/models/resnet18.py:20"),
        ("C++/LibTorch", "Adam", 1e-4, 128, "DataLoader workers", 2,
         "[0,1]", "cpp/src/train/native/train_model.h:64"),
        ("Java/DL4J", "Adam", 1e-4, 128, "AsyncDataSetIterator", 2,
         "[0,1]", "…/dataloader/PNGDataloader.java:32"),
        ("R/torch", "Adam", 1e-4, 128, "dataloader workers", 0,
         "[0,1]", "R/models/resnet18.r:75"),
        ("Rust/tch", "Adam", 1e-3, 128, "rayon par_iter (all cores)", 96,
         "mean/std normalised", "rust/src/bin/resnet_cifar100.rs:33, datasets/cifar100.rs:62"),
        ("MATLAB/DLT", "adam", 1e-4, 128, "augmentedImageDatastore", 1,
         "[0,1] (Normalization='none')", "matlab/train/+resnet18/train_cifar100.m:13"),
    ],
    columns=[
        "ecosystem", "optimiser", "learning_rate", "batch_size",
        "loader_mechanism", "loader_threads", "input_scaling", "source",
    ],
)

# --------------------------------------------------------------------------
# Protocol after the alignment applied in this revision.
# --------------------------------------------------------------------------
PROTOCOL_V2 = pd.DataFrame(
    [
        ("learning rate", "1e-4 for six stacks, 1e-3 for TensorFlow and Rust",
         "1e-4 everywhere",
         "python/tensorflow/models/*.py, rust/src/bin/*.rs"),
        ("loader threads", "0 (R), 1 (TF, JAX, MATLAB), 2 (PyTorch, C++, Java), 96 (Rust)",
         "2 by default; Rust pool capped via DEEPGREEN_LOADER_THREADS, R moved 0 -> 2",
         "R/models/*.r, rust/src/lib.rs::init_loader_pool"),
        ("input scaling", "Rust normalised with mean/std, the other seven used raw [0,1]",
         "raw [0,1] everywhere; set DEEPGREEN_NORMALIZE=1 to restore",
         "rust/src/datasets/*.rs, rust/src/lib.rs::normalize_inputs"),
        ("C++ dataset loader", "per-target choice between two implementations; the "
         "imported targets did not build at all",
         "single entry point cpp/src/dataset/ImageFolder.h, lazy by default; "
         "imported targets re-enabled",
         "cpp/src/dataset/ImageFolder.h, cpp/CMakeLists.txt"),
        ("Rust input shape", "Fashion-MNIST trained at 28x28x1 and evaluated at 32x32x3; "
         "resnet_tiny trained at 64x64 while vgg_tiny used 32x32",
         "32x32x3 everywhere, applied in the loader so both phases share it",
         "rust/src/datasets/fashion.rs, rust/src/bin/resnet_tiny.rs"),
        ("Rust eval batching", "iter_batches(1) in all six binaries, against 128 elsewhere",
         "batched at the training batch size, accuracy computed on the batch",
         "rust/src/bin/*.rs"),
        ("dataset paths", "hard-coded /home/pampaj/DeepGreen/... in every Rust binary",
         "resolved from DEEPGREEN_DATA with a 'data/' fallback",
         "rust/src/lib.rs::data_path"),
        ("CodeCarbon config", "two major versions, two tracking modes, two sampling intervals",
         "one pinned configuration, version-checked at startup",
         "tools/codecarbon_config.json, tools/deepgreen_bench.py"),
    ],
    columns=["aspect", "first_campaign", "after_alignment", "where"],
)

# --------------------------------------------------------------------------
# The "shared backend" is not a shared build.
# --------------------------------------------------------------------------
# Reviewer 1 comment 8 asks for the LibTorch build to be aligned across the four
# stacks that share it, so that binding overhead is isolated from backend
# version. The four stacks span six LibTorch minor releases, so the internal
# control group is not in fact controlled.
LIBTORCH_VERSIONS = pd.DataFrame(
    [
        ("Python/PyTorch", "torch 2.6.0 (requirements/pytorch_raw.txt)", "2.6.0"),
        ("C++/LibTorch", "direct, fetched by CMake", "2.7.0+cu128"),
        ("Rust/tch", "tch 0.14.0 / torch-sys 0.14.0 (Cargo.lock)", "2.1.0"),
        ("R/torch", "R torch 0.15.1 (requirements/r.txt)", "bundled, version not recorded"),
    ],
    columns=["ecosystem", "binding", "libtorch"],
)

#: Other implementation-level differences that are not per-ecosystem parameters.
STRUCTURAL_FINDINGS = [
    (
        "C++ off-the-shelf path does not build",
        "cpp/CMakeLists.txt comments out all six *_imported targets, and "
        "cpp/src/train/imported/train_model.h includes dataset/ImageFolder.h, which "
        "does not exist in cpp/src/dataset/ (only InMemoryImageFolder.h and "
        "LazyImageFolder.h do). Only the hand-ported native models build, so the "
        "manuscript's claim that models are 'natively provided off-the-shelf' does "
        "not hold for C++.",
    ),
    (
        "C++ loading strategy varied by build target",
        "Every native target used LazyImageFolder. Among the six imported targets, "
        "five listed LazyImageFolder.h and resnet18_fashion_imported listed "
        "InMemoryImageFolder.h, i.e. the in-memory preloading the paper mentions "
        "applied to exactly one configuration, which was not buildable anyway.",
    ),
    (
        "Rust preprocessing differs from every other stack",
        "rust/src/datasets/cifar100.rs applies per-channel mean/std normalisation "
        "(0.5071/0.4867/0.4408, 0.2675/0.2565/0.2761). The other seven ecosystems "
        "feed raw [0,1] inputs. This changes the optimisation problem, not just the "
        "arithmetic.",
    ),
    (
        "Rust trained and evaluated on different inputs",
        "rust/src/bin/resnet_fashion.rs and vgg_fashion.rs trained on Fashion-MNIST "
        "at its native 28x28x1 -- the loader applied no resize and no channel "
        "replication -- while their own evaluation loops resized each image to "
        "32x32 and replicated to three channels, as every other ecosystem did for "
        "both phases. The conv1 input volume during training was 0.26x everyone "
        "else's. Rust/ResNet18/FashionMNIST is also the single cheapest cell in the "
        "campaign, at 0.07x the median training energy of the other stacks.",
    ),
    (
        "Rust input resolution varied by binary",
        "resnet_tiny.rs passed resize_to = None, training Tiny ImageNet at its "
        "native 64x64, while vgg_tiny.rs passed Some(32) and every other ecosystem "
        "used 32x32 -- four times the spatial work for the same nominal "
        "configuration, inside the same ecosystem.",
    ),
    (
        "Rust evaluated one image at a time",
        "All six Rust binaries ran inference with test_data.iter_batches(1) while "
        "the seven other ecosystems evaluated at batch 128. Batch-1 GPU inference "
        "is launch-overhead bound. Rust's inference energy is 1.5-1.95x the median "
        "of the other stacks on CIFAR-100 and Fashion-MNIST while its training "
        "energy is 0.07-0.61x, which is the manuscript's train/inference ranking "
        "reversal reproduced as a pure artefact of batching.",
    ),
    (
        "The measurement instrument was configured per stack, in five places",
        "Each non-Python ecosystem drives CodeCarbon through its own bridge, and each "
        "bridge configures it differently. Read from source: Rust passes "
        "measure_power_secs=1 from its daemon, R passes measure_power_secs=1 AND "
        "tracking_mode='process', Python/JAX passes 1, while C++, Java, MATLAB, "
        "Python/PyTorch and Python/TensorFlow all take the 15 s default. Worse, the "
        "Rust daemon spawns 'python3' from PATH -- whichever interpreter happens to "
        "be first -- so the CodeCarbon version that measures the Rust stack is an "
        "ambient property of the shell, which is how one stack was measured with "
        "2.8.4 and another with 3.0.4 in the same campaign.",
    ),
    (
        "Rust builds to a CPU binary against a CUDA LibTorch",
        "cargo build links torch_cuda only if some Rust symbol references it, and none "
        "does, so the linker's default --as-needed drops the dependency. The binary "
        "then reports Device::Cpu and trains on the CPU while looking entirely normal; "
        "torch-sys's build.rs documents the problem in a comment without solving it. "
        "Building with -Wl,--no-as-needed and explicit -ltorch_cuda -lc10_cuda flips "
        "the same source from Cpu to Cuda(0). Nothing in the first campaign recorded "
        "which of the two it measured. scripts/build_rust_cuda.sh pins the flags and "
        "verifies the device before the campaign starts.",
    ),
    (
        "The Rust measurement bridge was not synchronised",
        "rust/src/emissions.rs wrote START into the tracker daemon's stdin and began "
        "computing immediately, without waiting for the tracker to exist. CodeCarbon "
        "takes seconds to initialise, so the tracked window did not cover the work it "
        "was meant to measure. Measured on this machine for one epoch of "
        "ResNet-18/CIFAR-100: the inference block recorded 0 J for real GPU work, and "
        "the training block recorded 861 J against 1127 J once START and STOP were made "
        "synchronous -- a 24% underestimate. Of the three daemon-style bridges only "
        "Rust had this defect: Java waits on a ready latch (readySignal.await()) and "
        "C++ embeds the interpreter with Py_Initialize, which is synchronous by "
        "construction. Reviewer 1 comment 13 asked for this controller to be "
        "validated; this is the answer.",
    ),
    (
        "A framework can silently run on the CPU",
        "TensorFlow 2.21's [and-cuda] extra resolves CUDA 12.9 wheels (cuBLAS "
        "12.9.2, cuDNN 9.24) while the binary is built against CUDA 12.5.1. It "
        "cannot dlopen them, reports one warning -- 'Cannot dlopen some GPU "
        "libraries' -- and runs the whole workload on the CPU. TensorFlow 2.19.1 on "
        "the identical card and driver resolves cuBLAS 12.5.3 and cuDNN 9.3.0 and "
        "uses the GPU normally; the compiled compute capabilities are the same in "
        "both (sm_60/70/80/89, compute_90), so this is a packaging accident, not a "
        "hardware-support limit. Nothing in the measurement reveals it: the run "
        "completes, the energy is recorded, and the CPU cost is attributed to the "
        "ecosystem. In the first campaign the TensorFlow stack had the LOWEST mean "
        "GPU power of all eight -- 113 W training and 97 W inference against a 350 W "
        "card -- with a total of 344 W, which is the signature of exactly this "
        "failure. The logs cannot settle whether it occurred, because device "
        "placement was never recorded. tools/deepgreen_bench.py now refuses to start "
        "when the framework cannot see the accelerator and records the resolved "
        "device in manifest.json.",
    ),
    (
        "TensorFlow evaluated with the backbone in training mode",
        "python/tensorflow/models/resnet18.py built the functional graph with "
        "backbone(inputs, training=True), pinning training-mode behaviour into the "
        "graph. Batch normalisation then used the statistics of the current batch "
        "during evaluation as well. Because the test split is served in class order, "
        "each evaluation batch held a single class and BN normalised against "
        "degenerate per-class statistics: measured test accuracy was 21% against 83% "
        "on train, and rose to 86% once the flag was removed. Two consequences. The "
        "accuracy this stack would have reported was meaningless -- but accuracy was "
        "never written to disk, so nothing surfaced it. And the inference phase was "
        "measuring a different computation from every other ecosystem, all of which "
        "switch to eval mode (model.eval(), set_train(false), $eval()).",
    ),
    (
        "A dataset class name contained a path separator",
        "Fashion-MNIST class 0 is 'T-shirt/top'. dataloader/download_convert_fashion.py "
        "used the label verbatim as a directory name, so <root>/train/T-shirt/ holds a "
        "top/ subdirectory rather than images. Loaders disagree about what the dataset "
        "contains: torchvision's ImageFolder and Keras' flow_from_directory walk "
        "recursively and recover the 6000 images, a loader listing only direct children "
        "sees an empty class, and Rust's fs::read_dir yields the subdirectory as an "
        "image path. Class names are now sanitised and the shared loader fails loudly "
        "on an empty class.",
    ),
    (
        "Learning rate is not common",
        "Python/TensorFlow and Rust/tch use 1e-3; the other six use 1e-4. The "
        "manuscript states a common learning rate. Energy per epoch is unaffected, "
        "but no accuracy-normalised comparison is valid across this difference.",
    ),
]


def parallelism_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Does data-loader parallelism explain the ranking?"""
    train = df[df["phase"] == "Training"]
    obs = train.groupby("ecosystem", observed=True).agg(
        mean_duration_s=("duration_s", "mean"),
        mean_energy_J=("energy_harm_j", "mean"),
        mean_gpu_power_W=("gpu_power_derived_w", "mean"),
    )
    tbl = PROTOCOL.set_index("ecosystem").join(obs).reset_index()
    tbl = tbl.sort_values("mean_duration_s")
    return tbl


def main() -> None:
    df = load()
    print("=" * 78)
    print("IMPLEMENTATION AUDIT  (R1 c7 and c12, R3 M3)")
    print("=" * 78)

    print("\n--- training protocol as implemented, first campaign ---")
    cols = ["ecosystem", "optimiser", "learning_rate", "batch_size",
            "loader_mechanism", "loader_threads", "input_scaling"]
    print(PROTOCOL[cols].to_string(index=False))
    save_table(PROTOCOL, "impl_protocol_divergence",
               "Training protocol per ecosystem, read from source (first campaign)")

    lrs = PROTOCOL["learning_rate"].unique()
    print(
        f"\n  The manuscript states a common learning rate. Two distinct values are "
        f"implemented: {sorted(lrs)}. Python/TensorFlow and Rust/tch use 1e-3, the "
        "other six use 1e-4."
    )

    print("\n--- the shared LibTorch backend is not a shared build (R1 c8) ---")
    print(LIBTORCH_VERSIONS.to_string(index=False))
    print(
        "\n  The four stacks that the manuscript treats as sharing a backend span\n"
        "  LibTorch 2.1.0 to 2.7.0. Any difference between them therefore mixes\n"
        "  binding overhead with six minor releases of kernel, allocator and cuDNN\n"
        "  behaviour. The LibTorch group is the study's natural control, and it was\n"
        "  not controlled."
    )
    save_table(LIBTORCH_VERSIONS, "impl_libtorch_versions",
               "LibTorch build behind each of the four LibTorch-based ecosystems")

    tbl = parallelism_analysis(df)
    print("\n--- ranking versus data-loader parallelism ---")
    print(
        tbl[["ecosystem", "loader_mechanism", "loader_threads",
             "mean_duration_s", "mean_energy_J", "mean_gpu_power_W"]]
        .round(2)
        .to_string(index=False)
    )

    rho, p = stats.spearmanr(tbl["loader_threads"], tbl["mean_duration_s"])
    rho_e, p_e = stats.spearmanr(tbl["loader_threads"], tbl["mean_energy_J"])
    print(
        f"\n  Spearman(loader threads, mean epoch duration) = {rho:.3f} (p = {p:.4f})"
        f"\n  Spearman(loader threads, mean epoch energy)   = {rho_e:.3f} (p = {p_e:.4f})"
    )
    fastest = tbl.iloc[0]
    slowest = tbl.iloc[-1]
    print(
        f"\n  The fastest ecosystem ({fastest['ecosystem']}) decodes images across "
        f"{int(fastest['loader_threads'])} cores; the slowest ({slowest['ecosystem']}) "
        f"uses {int(slowest['loader_threads'])} loader workers, i.e. it decodes on the "
        "main thread. They differ by "
        f"{slowest['mean_duration_s']/fastest['mean_duration_s']:.1f}x in duration."
    )
    save_table(tbl.round(4), "impl_parallelism_vs_ranking",
               "Measured ranking against data-loader parallelism")

    print("\n--- structural findings ---")
    for i, (title, detail) in enumerate(STRUCTURAL_FINDINGS, 1):
        print(f"\n  {i}. {title}")
        for line in _wrap(detail, 72):
            print(f"     {line}")
    save_table(
        pd.DataFrame(STRUCTURAL_FINDINGS, columns=["finding", "detail"]),
        "impl_structural_findings",
        "Implementation-level defects found by reading all eight ecosystems",
    )

    print("\n--- alignment applied in this revision ---")
    print(PROTOCOL_V2[["aspect", "first_campaign", "after_alignment"]].to_string(index=False))
    save_table(PROTOCOL_V2, "impl_alignment_applied",
               "Protocol divergences and the change applied to each")
    print(
        "\n  NOTE: the C++, Java, R, MATLAB and Rust edits could not be compiled or run\n"
        "  in the revision environment (no cargo, maven, Rscript or MATLAB available,\n"
        "  and LibTorch/OpenCV are not installed). They are source-level changes and\n"
        "  must be built and smoke-tested on the measurement server before the\n"
        "  replicated campaign starts."
    )

    print(
        "\n" + "=" * 78 + "\n"
        "CONCLUSION\n"
        "  The eight ecosystems do not implement the same experiment. They differ in\n"
        "  learning rate, in input scaling, and above all in how many CPU threads\n"
        "  decode and feed images. Combined with the audit finding that the GPU runs\n"
        "  at 24-56% of its power limit and that duration accounts for ~100% of the\n"
        "  log energy spread, the most parsimonious reading of the headline result is\n"
        "  that it ranks DATA-PIPELINE CONFIGURATIONS, not ecosystems.\n"
        "\n"
        "  This is a fixable defect, not a fatal one, but it must be fixed by\n"
        "  re-running with the pipeline held constant -- see repetition_protocol.md\n"
        "  section 7 -- before any ecosystem-level claim is made.\n"
        + "=" * 78
    )


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width)


if __name__ == "__main__":
    main()
