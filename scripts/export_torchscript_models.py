#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the reference models once, as TorchScript, for every LibTorch-based stack.

Spec S1: Python/PyTorch, C++/LibTorch, Rust/tch and R/torch must train the *same
module*. They are the study's internal control group -- four bindings over one
backend -- so any difference between them should be binding, runtime and
data-pipeline overhead and nothing else. In the first campaign each of them
built its own model: torchvision in Python, a hand-written port in C++, another
hand-written port in Rust, torchvision-for-R in R. That is four implementations,
not one.

    python3 scripts/export_torchscript_models.py            # -> models/*.pt
    DEEPGREEN_MODELS=/somewhere python3 scripts/export_torchscript_models.py

The C++ build produces the same artefacts through cmake_script/GenerateModel.cmake;
this script exists so Rust and R do not need a C++ build to get them, and so the
export is reproducible on its own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torchvision import models

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("DEEPGREEN_MODELS", REPO / "models"))

#: dataset -> number of classes, matching cpp/CMakeLists.txt
NUM_CLASSES = {"fashionmnist": 10, "cifar100": 100, "tinyimagenet200": 200}


def build(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif arch == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier[6] = torch.nn.Linear(m.classifier[6].in_features, num_classes)
    else:
        raise ValueError(arch)
    return m


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for arch in ("resnet18", "vgg16"):
        for dataset, n in NUM_CLASSES.items():
            model = build(arch, n)
            model.train()
            scripted = torch.jit.script(model)
            out = OUT_DIR / f"{arch}_{dataset}.pt"
            scripted.save(str(out))
            written.append(out)
            print(f"  {out.relative_to(REPO) if out.is_relative_to(REPO) else out}  "
                  f"({n} classes, {sum(p.numel() for p in model.parameters()):,} params)")

    manifest = OUT_DIR / "MANIFEST.txt"
    manifest.write_text(
        "# Shared TorchScript modules for the LibTorch-based ecosystems.\n"
        f"torch_version={torch.__version__}\n"
        f"modules={' '.join(sorted(p.name for p in written))}\n"
        "# A module written by a newer torch will not load into an older LibTorch.\n"
        "# scripts/check_consistency.py compares this against the tch crate version\n"
        "# and the LibTorch the C++ build fetches (spec S4).\n"
    )
    print(f"\n{len(written)} modules written to {OUT_DIR}")
    print("torch", torch.__version__, "- record this version: the four LibTorch stacks")
    print("must load a module produced by ONE torch build (spec S4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
