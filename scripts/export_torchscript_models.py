#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the reference models once, as TorchScript, for every stack that can load
them -- and publish the parameter counts every other stack must match.

Spec S1: the seven ecosystems must train the *same network*. They did not. Two
separate failures, found by two reviewers reading the same tree:

  * VGG-16 was four different networks. torchvision's ImageNet classifier in
    PyTorch, C++ and Rust (134,670,244 parameters); Flatten + 4096 + 4096 in
    TensorFlow and Deeplearning4j (34,006,948); 512 + 512 in R (15,028,644);
    and in JAX no classifier at all -- `include_head=False` followed by global
    average pooling and one Dense, a linear probe (14,765,988). A 9.1x range.
    Half the study's energy comparisons were comparing models.
  * ResNet-18 was parity for five stacks of seven. PyTorch, C++, Rust and JAX
    all land on 11,227,812 exactly. TensorFlow's Model Garden adds a 1x1
    projection shortcut on layer1 where torchvision uses the identity. And
    Deeplearning4j builds the graph by hand: WeightInit.RELU -- a *truncated*
    normal in fan_in mode -- against torchvision's untruncated kaiming_normal_
    in fan_out, ConvolutionMode.Truncate instead of explicit padding, and a bias
    on every convolution where torchvision omits it because a BatchNorm follows.

The initialiser is not a detail. In a controlled single-framework experiment,
holding everything else fixed, VGG-16 at this learning rate collapses to chance
in 0 of 6 runs under He, 2 of 6 under Glorot and 4 of 6 under Xavier -- and
Deeplearning4j, the one stack with a hand-rolled initialiser, carries 5 of the
campaign's 12 collapses. What the paper read as an ecosystem effect tracks the
initialiser.

Two decisions are recorded here rather than left implicit.

**The VGG-16 head is the CIFAR head, not torchvision's.** At 32x32 the feature
map reaching torchvision's classifier is 1x1x512, and AdaptiveAvgPool2d((7,7))
replicates it -- literally: allclose(pooled, feat.expand(-1,-1,7,7)) is True. So
Linear(25088, 4096) is fed 49 copies of 512 values, and 102.8M of the model's
134.7M parameters have rank at most 512. That is not a design at this
resolution, it is an artefact of borrowing an ImageNet architecture, and
measuring it measures Adam state bandwidth rather than VGG-16. Measured on an
RTX 3090 at batch 128: 29.67 ms and 10.11 J per step for the ImageNet head
against 14.14 ms and 4.22 J for the CIFAR head. Uniformity is what the
comparison needs; this is the more meaningful of the two uniform choices, and
the cheaper. Set HEAD = "imagenet" to take the other branch.

**The export is seeded.** It was not, and it mattered: C++ and Rust each loaded
a module from a separate unseeded export, so the two stacks the paper describes
as starting from identical parameters started from different draws, and
Python/PyTorch built torchvision fresh and loaded no module at all. One seed,
one export, and the claim becomes true instead of needing to be withdrawn.

    python3 scripts/export_torchscript_models.py            # -> models/*.pt
    DEEPGREEN_MODELS=/somewhere python3 scripts/export_torchscript_models.py

Writes models/MANIFEST.json, which carries the parameter count and SHA-256 of
every module. Stacks that cannot load TorchScript -- TensorFlow, JAX,
Deeplearning4j -- assert their own parameter count against it at startup, which
is the check that would have caught all of the above.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("DEEPGREEN_MODELS", REPO / "models"))

#: dataset -> number of classes, matching cpp/CMakeLists.txt
NUM_CLASSES = {"fashionmnist": 10, "cifar100": 100, "tinyimagenet200": 200}

#: One draw for the whole export, so every stack that loads a module and every
#: stack that rebuilds it from this definition starts from the same weights.
EXPORT_SEED = 20260830

#: "cifar" (512 -> 512 -> classes, on a global-average-pooled 1x1 map) or
#: "imagenet" (torchvision's 25088 -> 4096 -> 4096 -> classes). See the module
#: docstring: at 32x32 the second is fed 49 copies of the same 512 values.
HEAD = "cifar"


def build(arch: str, num_classes: int) -> nn.Module:
    """The canonical definition of each architecture, for every stack."""
    torch.manual_seed(EXPORT_SEED)
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "vgg16":
        m = models.vgg16(weights=None)
        if HEAD == "imagenet":
            m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
            return m
        m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        m.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        return m
    raise ValueError(arch)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_sha256(model: nn.Module) -> str:
    """Hash the parameters, in name order, rather than the file.

    Two exports of this script from the same seed produce identical weights and
    different bytes -- TorchScript's archive is not reproducible even with its
    timestamps fixed. So the file hash certifies the artefact that was shipped,
    and this one certifies the thing the study actually needs to be equal, and
    can be recomputed by anyone from the definition above. "Byte-identical
    module" was always a stronger claim than the argument required.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modules: dict[str, dict[str, object]] = {}
    for arch in ("resnet18", "vgg16"):
        for dataset, n in NUM_CLASSES.items():
            model = build(arch, n)
            model.train()
            out = OUT_DIR / f"{arch}_{dataset}.pt"
            torch.jit.script(model).save(str(out))
            params = sum(p.numel() for p in model.parameters())
            modules[out.name] = {
                "architecture": arch, "dataset": dataset, "num_classes": n,
                "parameters": params,
                "weights_sha256": _weights_sha256(model),
                "file_sha256": _sha256(out),
            }
            print(f"  {out.name:34} {n:>3} classes  {params:>12,} params")

    manifest = {
        "torch_version": torch.__version__,
        "export_seed": EXPORT_SEED,
        "vgg16_head": HEAD,
        "modules": modules,
        "note": (
            "Parameter counts are the contract every stack asserts against at "
            "startup, including the four that cannot load TorchScript. A module "
            "written by a newer torch will not load into an older LibTorch; "
            "scripts/check_consistency.py compares torch_version against the tch "
            "crate and the LibTorch the C++ build fetches (spec S4)."
        ),
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{len(modules)} modules and MANIFEST.json written to {OUT_DIR}")
    print(f"torch {torch.__version__}, seed {EXPORT_SEED}, VGG-16 head '{HEAD}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
