#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The precision ablation the two campaigns already contain.

The first replicated campaign ran Python/PyTorch with TF32 denied -- the harness
of that time set ``matmul.allow_tf32 = False``, ``cudnn.allow_tf32 = False`` and
``cudnn.deterministic = True`` -- while the other six stacks, which never import
that harness, took cuDNN's Ampere default and ran with TF32 on. The second
campaign sets DEEPGREEN_TF32=1 and exports NVIDIA_TF32_OVERRIDE=1, so all seven
run with TF32 on. One stack changed precision policy between the campaigns and
six did not, on the same 210 configurations, 30 epochs and 5 repetitions each.

That is an ablation with a six-stack control group at 150 training epochs per
cell, and it was sitting on disk unread while the manuscript quoted a four-cell
micro-benchmark with no provenance. This script reports it.

results/campaign_v2_first_campaign/ is an input here and nowhere else in the
pipeline. Everything else in the analysis reads the current campaign only, for
the reason REVISION_LOG section 23 gives; this one comparison needs both, and
says so. It is read through 09_campaign_v2.collect() with the root
parameterised, so both sides pass the same completeness gate and neither side
gets its own copy of the parsing.

Comparability is the hard part
------------------------------
The campaigns differ in more than the precision flag, because the audit between
them fixed several things at once. A cell is a clean precision contrast only if
nothing else about it changed:

  * VGG-16 is out entirely. It ran as four different networks spanning 9.1x in
    parameters (log 2), so a v1/v2 VGG-16 ratio is mostly the architecture.

  * ResNet-18 was parity for five stacks of seven (log 2). Python/TensorFlow's
    Model Garden adds a projection shortcut on the first stage, and Java/DL4J
    built the graph by hand with ConvolutionMode.Truncate instead of explicit
    padding and a bias on every convolution -- both change the arithmetic, so
    both are out.

  * Tiny ImageNet and Fashion-MNIST were re-encoded between the campaigns, on
    the TRAINING split as well as the test one.
    scripts/normalise_dataset_resolution.py (commit 760a4c4d, 2026-08-30 14:44,
    after v1 finished on 08-29 and before v2 started on 08-30) rewrote all three
    datasets to 32x32 on disk. Log 13 reports a test-split fingerprint and log
    17 says 240,000 images; this is the direct check, against the trees the two
    campaigns actually read (data/*_png now, data/*_png_original then, dated
    08-30 14:2x and 08-18/19 respectively):

        cifar100      train 50,000 files, 300 sampled: 300 pixel-identical
        fashionmnist  train 60,000 files, 300 sampled: 300 changed, 28x28->32x32
        tinyimagenet  train 100,000 files, 300 sampled: 300 changed, 64x64->32x32

    and the test splits the same way. So on two of three datasets every stack
    decodes different pixels and does different loader work in the two
    campaigns; Tiny ImageNet also stopped being downsampled at all, and its
    decode is a quarter of the pixels it was. CIFAR-100 was already 32x32, was
    rewritten byte-wise and came back pixel-identical, and is the only dataset
    that did not move.

  * Python/TensorFlow and Python/JAX batched with drop_remainder=True in v1
    (log 5), so they ran one fewer gradient step per epoch and evaluated on
    9,984 of 10,000 images. On CIFAR-100 that is 390 steps against 391, 0.26%,
    two orders of magnitude below the effect being measured -- recorded as a
    caveat on JAX rather than an exclusion, because a control group of one is
    not a control group. TensorFlow is already excluded on architecture.

Two things are *not* excluded, deliberately, because they cannot move per-epoch
energy: the initialiser divergences (log 3, 14) and R's double rescaling
(log 8) change what a network learns, not how much arithmetic it does. They
invalidate a v1/v2 comparison of accuracy, and this table reports none.

One confound remains inside the comparable cells and is stated rather than
removed: in v1 Python/PyTorch built torchvision eagerly, and in v2 it loads the
exported TorchScript module (tools/deepgreen_bench.py load_shared_module). The
control group cannot separate that from TF32, because the other stacks loaded a
module in both campaigns. The PyTorch ratio is therefore an upper bound on the
precision effect.

Writes results/revision/tables/v2_tf32_campaign_contrast.{md,csv}.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, announce_scope, save_table)  # noqa: E402

V1 = REPO_ROOT / "results" / "campaign_v2_first_campaign"
V2 = REPO_ROOT / "results" / "campaign_v2"

#: The stack whose precision policy changed between the campaigns.
TREATMENT = "Python/PyTorch"
#: The stack the manuscript's headline gap is measured against.
REFERENCE = "C++/LibTorch"

#: Why a cell is not a clean precision contrast. Checked in this order, so a
#: cell excluded for its architecture is not also blamed on its dataset.
NOT_COMPARABLE = [
    (lambda e, m, d: m == "vgg16",
     "VGG-16 ran as four different networks spanning 9.1x in parameters (log 2)"),
    (lambda e, m, d: e == "Python/TensorFlow",
     "Model Garden adds a projection shortcut on the first stage (log 2), and "
     "drop_remainder=True gave one fewer gradient step per epoch (log 5)"),
    (lambda e, m, d: e == "Java/DL4J",
     "graph built by hand: ConvolutionMode.Truncate instead of explicit padding "
     "and a bias on every convolution, so the arithmetic differs (log 2)"),
    (lambda e, m, d: e == "Python/JAX",
     "the measurement window itself changed: block_until_ready was absent in "
     "v1, so 35.9% of the work finished after the block closed and the energy "
     "attributed to it was 0.70x the true figure (log 4)"),
    (lambda e, m, d: d == "tinyimagenet",
     "training images re-encoded 64x64 -> 32x32 between the campaigns "
     "(log 13, 17): all 100,000 differ pixel for pixel, so the loader stopped "
     "downsampling and every stack decodes a quarter of the pixels"),
    (lambda e, m, d: d == "fashionmnist",
     "training images re-encoded 28x28 -> 32x32 between the campaigns "
     "(log 17): all 60,000 differ pixel for pixel, so the loader stopped "
     "upsampling"),
]

#: Stated but not excluding: too small to matter at the scale being measured.
CAVEATS = [
    (lambda e, m, d: e == TREATMENT,
     "caveat: v1 built torchvision eagerly and v2 loads the exported "
     "TorchScript module, so the ratio bounds the precision effect from above"),
]


def comparability(ecosystem: str, model: str, dataset: str) -> tuple[str, str]:
    """``(comparable, note)`` for one cell -- "yes"/"no" and always a reason."""
    for predicate, reason in NOT_COMPARABLE:
        if predicate(ecosystem, model, dataset):
            return "no", reason
    for predicate, note in CAVEATS:
        if predicate(ecosystem, model, dataset):
            return "yes", note
    return "yes", "no known between-campaign difference for this cell"


def collector():
    """09's collector, imported rather than copied. See its docstring."""
    path = Path(__file__).resolve().parent / "09_campaign_v2.py"
    spec = importlib.util.spec_from_file_location("09_campaign_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_cell(blocks: pd.DataFrame) -> pd.DataFrame:
    """Per-epoch GPU energy and duration, per (ecosystem, model, dataset, phase).

    GPU energy rather than the counter total: it is the one quantity read by the
    same instrument in both campaigns and attributable to the computation, and
    the CPU package term carries whatever the host was doing around it.

    The mean is the reported statistic and the median is carried beside it,
    because a ratio of means and a ratio of medians disagreeing is itself worth
    seeing on a table this small.
    """
    keys = ["ecosystem", "model", "dataset", "phase"]
    g = blocks.groupby(keys, observed=True)
    out = g.agg(
        n_runs=("run_dir", "nunique"),
        n_epochs=("gpu_energy_j", "size"),
        gpu_j_mean=("gpu_energy_j", "mean"),
        gpu_j_median=("gpu_energy_j", "median"),
        duration_s_mean=("duration_s", "mean"),
    ).reset_index()
    return out


def contrast(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    keys = ["ecosystem", "model", "dataset", "phase"]
    m = v1.merge(v2, on=keys, how="inner", suffixes=("_v1", "_v2"))
    rows = []
    for _, r in m.iterrows():
        comparable, note = comparability(r.ecosystem, r.model, r.dataset)
        rows.append({
            "row_kind": "stack contrast",
            "quantity": "mean GPU J per epoch",
            "ecosystem": r.ecosystem, "model": r.model, "dataset": r.dataset,
            "phase": r.phase,
            "n_runs_v1": int(r.n_runs_v1), "n_runs_v2": int(r.n_runs_v2),
            "n_epochs_v1": int(r.n_epochs_v1), "n_epochs_v2": int(r.n_epochs_v2),
            "value_v1": round(float(r.gpu_j_mean_v1), 1),
            "value_v2": round(float(r.gpu_j_mean_v2), 1),
            "ratio_v1_v2": round(float(r.gpu_j_mean_v1 / r.gpu_j_mean_v2), 3),
            "ratio_v1_v2_median": round(float(r.gpu_j_median_v1 / r.gpu_j_median_v2), 3),
            "duration_v1_s": round(float(r.duration_s_mean_v1), 3),
            "duration_v2_s": round(float(r.duration_s_mean_v2), 3),
            "ratio_duration": round(float(r.duration_s_mean_v1 / r.duration_s_mean_v2), 3),
            "comparable": comparable,
            "comparability_note": note,
        })
    return pd.DataFrame(rows)


def gap_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The PyTorch-vs-C++ energy gap within each campaign, and how it moved.

    The manuscript's headline is this gap, and it was read as a binding cost.
    Same columns as the rows above under the reading "the quantity named in
    ``quantity``, in each campaign, and its v1/v2 ratio" -- here the quantity is
    itself a within-campaign ratio, so ``ratio_v1_v2`` is how far the gap
    collapsed.
    """
    rows = []
    stacks = table[table.row_kind == "stack contrast"]
    for (model, dataset, phase), g in stacks.groupby(["model", "dataset", "phase"]):
        a = g[g.ecosystem == TREATMENT]
        b = g[g.ecosystem == REFERENCE]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        # The gap is only a precision contrast where both of its sides are.
        if "no" in (a.comparable, b.comparable):
            comparable, note = "no", "one side of the gap is not comparable"
        else:
            comparable, note = "yes", f"{a.comparability_note}"
        rows.append({
            "row_kind": "PyTorch/C++ gap",
            "quantity": f"{TREATMENT} over {REFERENCE}, mean GPU J per epoch",
            "ecosystem": f"{TREATMENT} : {REFERENCE}", "model": model,
            "dataset": dataset, "phase": phase,
            "n_runs_v1": int(a.n_runs_v1 + b.n_runs_v1),
            "n_runs_v2": int(a.n_runs_v2 + b.n_runs_v2),
            "n_epochs_v1": int(a.n_epochs_v1 + b.n_epochs_v1),
            "n_epochs_v2": int(a.n_epochs_v2 + b.n_epochs_v2),
            "value_v1": round(float(a.value_v1 / b.value_v1), 3),
            "value_v2": round(float(a.value_v2 / b.value_v2), 3),
            "ratio_v1_v2": round(float((a.value_v1 / b.value_v1)
                                       / (a.value_v2 / b.value_v2)), 3),
            "ratio_v1_v2_median": np.nan,
            "duration_v1_s": round(float(a.duration_v1_s / b.duration_v1_s), 3),
            "duration_v2_s": round(float(a.duration_v2_s / b.duration_v2_s), 3),
            "ratio_duration": round(float((a.duration_v1_s / b.duration_v1_s)
                                          / (a.duration_v2_s / b.duration_v2_s)), 3),
            "comparable": comparable,
            "comparability_note": note,
        })
    return pd.DataFrame(rows)


def main() -> int:
    announce_scope("18_precision_ablation")
    if not V1.is_dir():
        print(f"no superseded campaign under {V1.relative_to(REPO_ROOT)}; "
              "this comparison needs both campaigns")
        return 0

    collect = collector().collect
    print(f"reading {V1.relative_to(REPO_ROOT)} ...")
    v1 = per_cell(collect(V1))
    print(f"reading {V2.relative_to(REPO_ROOT)} ...")
    v2 = per_cell(collect(V2))

    table = contrast(v1, v2)
    table = pd.concat([table, gap_rows(table)], ignore_index=True)
    table = table.sort_values(["row_kind", "phase", "model", "dataset", "ecosystem"])

    train = table[(table.phase == "Training") & (table.row_kind == "stack contrast")]
    clean = train[train.comparable == "yes"]
    print("\n--- comparable cells, training (the precision contrast) ---")
    print(clean[["ecosystem", "dataset", "value_v1", "value_v2", "ratio_v1_v2",
                 "ratio_duration"]].to_string(index=False))
    others = clean[clean.ecosystem != TREATMENT]
    if len(others):
        # The control group is the whole argument: six stacks whose precision
        # policy did not change must not move, or the ratio is measuring the
        # audit rather than the flag.
        print(f"\ncontrol group moves at most "
              f"{100 * (others.ratio_v1_v2 - 1).abs().max():.2f}% "
              f"({len(others)} stacks); {TREATMENT} moves "
              f"{clean[clean.ecosystem == TREATMENT].ratio_v1_v2.min():.2f}-"
              f"{clean[clean.ecosystem == TREATMENT].ratio_v1_v2.max():.2f}x")
    # The grading above is a priori, from the log. This checks it against the
    # data: on a dataset that did not change, the stacks that did not change
    # their precision policy must not move. It is what licenses the exclusions
    # rather than merely asserting them -- and it is why fashionmnist stays out
    # despite being 28x28 -> 32x32 rather than 64x64 -> 32x32.
    print("\n--- control check: how far the unchanged stacks move, by dataset ---")
    # ResNet-18 only: VGG-16 is a different network in the two campaigns, so its
    # movement measures the architecture and says nothing about the dataset.
    control = train[train.ecosystem.isin([REFERENCE, "Rust/tch", "R/torch"])
                    & (train.model == "resnet18")]
    for dataset, g in control.groupby("dataset"):
        worst = g.loc[(g.ratio_v1_v2 - 1).abs().idxmax()]
        print(f"  {dataset:<13} worst {100 * abs(worst.ratio_v1_v2 - 1):5.1f}%  "
              f"({worst.ecosystem})   [{g.comparable.iat[0]}]")

    excluded = train[train.comparable == "no"]
    print(f"\nnot a precision contrast: {len(excluded)} of {len(train)} training "
          f"cells, by reason:")
    for reason, g in excluded.groupby("comparability_note"):
        print(f"  {len(g):>3}  {reason}")

    save_table(table, "v2_tf32_campaign_contrast",
               "Per-epoch GPU energy and duration in the superseded campaign "
               "(TF32 denied for Python/PyTorch only) against the current one "
               "(TF32 on for all seven), with the comparability of each cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
