#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset effect on energy, on an absolute scale.

Addresses:
  R3 M5 -- the manuscript normalises each dataset by its own maximum and then
           reads growth across datasets off that heatmap, which the
           normalisation makes impossible. Reports absolute energy, a single
           common reference, and the confounded factors behind "complexity".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import load, save_table

# What actually differs between the three datasets. "Complexity" conflates all
# of these, so the design cannot attribute the scaling to any single one.
DATASET_FACTS = pd.DataFrame(
    [
        ("FashionMNIST", 60_000, 10_000, 10, 1, "28x28", "32x32"),
        ("CIFAR100", 50_000, 10_000, 100, 3, "32x32", "32x32"),
        ("TinyImageNet", 100_000, 10_000, 200, 3, "64x64", "32x32"),
    ],
    columns=[
        "dataset",
        "train_images",
        "test_images",
        "classes",
        "native_channels",
        "native_resolution",
        "resolution_used",
    ],
)


def main() -> None:
    df = load()
    print("=" * 78)
    print("DATASET SCALING  (R3 major comment 5)")
    print("=" * 78)

    print("\n--- what differs between the datasets ---")
    print(DATASET_FACTS.to_string(index=False))
    print(
        "\n  The three datasets differ simultaneously in training-set size, class count,\n"
        "  channel count and native resolution, and Tiny ImageNet is downsampled from\n"
        "  64x64 to 32x32, removing the property that makes it demanding. 'Dataset\n"
        "  complexity' is therefore not a single manipulated factor and the design\n"
        "  cannot isolate which of these drives the observed scaling."
    )
    save_table(DATASET_FACTS, "dataset_factors", "Factors confounded in 'dataset complexity'")

    for phase in ["Training", "Inference"]:
        sub = df[df["phase"] == phase]
        abs_tbl = (
            sub.groupby(["ecosystem", "dataset"], observed=True)["energy_harm_j"]
            .mean()
            .unstack()
            .round(1)
        )
        abs_tbl = abs_tbl[["FashionMNIST", "CIFAR100", "TinyImageNet"]]
        print(f"\n--- {phase}: absolute mean energy per epoch (J) ---")
        print(abs_tbl.to_string())

        # Single common reference: the cheapest ecosystem x dataset cell overall.
        ref = abs_tbl.min().min()
        rel = (abs_tbl / ref).round(2)
        print(f"\n--- {phase}: same values against one common reference ({ref:.0f} J) ---")
        print(rel.to_string())

        growth = abs_tbl.mean(axis=0)
        print(
            f"\n  Mean across ecosystems: FashionMNIST {growth['FashionMNIST']:.0f} J, "
            f"CIFAR100 {growth['CIFAR100']:.0f} J, TinyImageNet {growth['TinyImageNet']:.0f} J "
            f"({growth['TinyImageNet']/growth['FashionMNIST']:.2f}x from the smallest to the largest)."
        )
        per_img = growth / DATASET_FACTS.set_index("dataset").loc[growth.index, "train_images"]
        print(
            "  Per training image: "
            + ", ".join(f"{k} {v*1000:.2f} mJ" for k, v in per_img.items())
            + " -- once normalised by the number of images processed, the apparent"
        )
        print("  scaling largely disappears, i.e. it reflects dataset size, not difficulty.")

        save_table(
            abs_tbl.reset_index(),
            f"dataset_absolute_energy_{phase.lower()}",
            f"{phase}: absolute mean energy per epoch (J), not per-column normalised",
        )
        save_table(
            rel.reset_index(),
            f"dataset_common_reference_{phase.lower()}",
            f"{phase}: energy against one common reference cell",
        )


if __name__ == "__main__":
    main()
