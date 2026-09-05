#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferential statistics for the replicated campaign, at the run level.

Reviewer #3 (M2) and Reviewer #1 (c5) both objected that the submitted analysis
treated the 30 epochs of a single run as repeated measurements. Epochs within a
run share an initialisation, a JIT outcome, an allocator state and a thermal
trajectory; the effective sample size per configuration was one, which makes any
p-value computed over them meaningless and any confidence interval far too
narrow.

This script therefore works on run totals. Each configuration contributes five
independent numbers, one per repetition, and every test below is computed on
those.

  * Kruskal-Wallis across ecosystems, within each (model, dataset, phase) block
  * pairwise Mann-Whitney with Holm correction, and Cliff's delta as the effect
    size, which is what the reviewers asked for and what a rank test can support
  * the LibTorch control group: how much of the total spread survives when the
    backend is held fixed
  * the energy--time relationship, on counter-bracketed durations rather than on
    the estimator's own duration field

Writes results/revision/tables/v2_stats_*.{md,csv}.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (write_table_path, REPO_ROOT, cliffs_delta, save_table, TABLES_RESOLVER, tables_dir)  # noqa: E402

TABLES = TABLES_RESOLVER  # writes divert on a live campaign, reads fall back
# Two nested groups, and the difference between them is the whole point.
#
# SHARED_MODULE are the stacks that under S1 load the byte-identical exported
# TorchScript module on the pinned LibTorch 2.7.0 build. That is the real
# control: same kernels, same weights, same graph.
#
# LIBTORCH_FAMILY adds R, which links its own bundled LibTorch and, because the
# R binding cannot switch a script module between train and eval mode, builds an
# equivalent architecture instead of loading the shared one. R was inside the
# control group and carried its spread: reporting 8.5--10.8x as "shared-backend
# overhead" was reporting the one member that shares neither the backend build
# nor the module.
SHARED_MODULE = {"Python/PyTorch", "Cpp/LibTorch", "C++/LibTorch", "Rust/tch"}
LIBTORCH_FAMILY = SHARED_MODULE | {"R/torch"}


def run_totals() -> pd.DataFrame:
    """One row per (run, phase): the quantity every test below is computed on."""
    e = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    g = (
        e.groupby(["ecosystem", "model", "dataset", "repetition", "phase"])
        .agg(energy_j=("hw_meas_j", "sum"),
             duration_s=("duration_hw_s", "sum"),
             gpu_j=("hw_gpu_j", "sum"),
             cpu_j=("hw_cpu_j", "sum"))
        .reset_index()
    )
    g["power_w"] = g.energy_j / g.duration_s
    return g


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni, returned in the input order."""
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def omnibus(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset, phase), g in runs.groupby(["model", "dataset", "phase"]):
        groups = [x.energy_j.values for _, x in g.groupby("ecosystem")]
        if len(groups) < 3:
            continue
        h, p = stats.kruskal(*groups)
        # epsilon-squared, the rank analogue of eta-squared
        n = sum(len(x) for x in groups)
        rows.append({
            "model": model, "dataset": dataset, "phase": phase,
            "k_ecosystems": len(groups), "n_runs": n,
            "H": round(h, 2), "p": p,
            "epsilon_sq": round((h - len(groups) + 1) / (n - len(groups)), 3),
        })
    return pd.DataFrame(rows)


def pairwise(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset, phase), g in runs.groupby(["model", "dataset", "phase"]):
        by_eco = {k: v.energy_j.values for k, v in g.groupby("ecosystem")}
        pairs = list(itertools.combinations(sorted(by_eco), 2))
        raw = []
        for a, b in pairs:
            try:
                _, p = stats.mannwhitneyu(by_eco[a], by_eco[b], alternative="two-sided")
            except ValueError:
                p = 1.0
            raw.append(p)
        for (a, b), p, padj in zip(pairs, raw, holm(raw)):
            d, mag = cliffs_delta(by_eco[a], by_eco[b])
            rows.append({
                "model": model, "dataset": dataset, "phase": phase,
                "a": a, "b": b,
                "median_a_J": round(float(np.median(by_eco[a])), 1),
                "median_b_J": round(float(np.median(by_eco[b])), 1),
                "p_raw": p, "p_holm": padj,
                "significant": padj < 0.05,
                "cliffs_delta": round(d, 3), "magnitude": mag,
            })
    return pd.DataFrame(rows)


def libtorch_control(runs: pd.DataFrame) -> pd.DataFrame:
    """How much spread survives when the backend is held fixed?

    Three of the seven ecosystems -- Python, C++ and Rust -- train the identical
    exported TorchScript module on the identical pinned LibTorch build. Whatever
    separates *those three* is binding and host overhead, not kernels, and that
    is the number this table exists to report.

    R links its own bundled LibTorch and builds an equivalent architecture
    rather than loading the shared module, so it belongs to the LibTorch family
    but not to the control. Both are reported, because the difference between
    them is itself a result: it says how much of the "shared-backend" spread was
    the one member that shared neither the build nor the module.
    """
    rows = []
    train = runs[runs.phase == "Training"]
    for (model, dataset), g in train.groupby(["model", "dataset"]):
        # Mean of run means, matching the estimator every other energy table in
        # this study uses. It was the median here, so the same quantity -- the
        # spread across all seven -- came out differently in two tables of the
        # same paper.
        med = g.groupby("ecosystem").energy_j.mean()
        ctrl = med[[e for e in med.index if e in SHARED_MODULE]]
        family = med[[e for e in med.index if e in LIBTORCH_FAMILY]]
        if len(ctrl) < 2:
            continue
        full_spread = med.max() / med.min()
        ctrl_spread = ctrl.max() / ctrl.min()
        family_spread = family.max() / family.min()
        rows.append({
            "model": model, "dataset": dataset,
            # Both group sizes are carried so that no consumer has to restate
            # them: 12's control table used to print the family size as
            # "shared module + 1", which is only true while R is the only
            # member LIBTORCH_FAMILY adds.
            "n_all": len(med), "n_shared_module": len(ctrl),
            "n_libtorch_family": len(family),
            "spread_all": round(full_spread, 2),
            "spread_shared_module": round(ctrl_spread, 2),
            "spread_libtorch_family": round(family_spread, 2),
            "share_of_log_spread_pct": round(
                100 * np.log(ctrl_spread) / np.log(full_spread), 1),
            "family_share_of_log_spread_pct": round(
                100 * np.log(family_spread) / np.log(full_spread), 1),
        })
    return pd.DataFrame(rows)


def energy_time(runs: pd.DataFrame) -> pd.DataFrame:
    """RQ3, on measured durations rather than on the estimator's duration field."""
    rows = []
    for phase, g in runs.groupby("phase"):
        med = g.groupby(["ecosystem", "model", "dataset"])[["energy_j", "duration_s"]].median()
        rho, p = stats.spearmanr(med.energy_j, med.duration_s)
        # rank inversions: pairs ordered one way by energy and the other by time
        vals = med.reset_index()
        inversions = sum(
            1 for (i, a), (j, b) in itertools.combinations(vals.iterrows(), 2)
            if (a.energy_j - b.energy_j) * (a.duration_s - b.duration_s) < 0
        )
        total = len(vals) * (len(vals) - 1) // 2
        rows.append({
            "phase": phase, "n_configurations": len(vals),
            "spearman_rho": round(rho, 3), "p": p,
            "discordant_pairs": inversions, "total_pairs": total,
            "discordant_pct": round(100 * inversions / total, 1),
        })
    return pd.DataFrame(rows)


def cell_energy_time(runs: pd.DataFrame) -> pd.DataFrame:
    """The energy-time relationship inside a single cell, not pooled over cells.

    energy_time() answers "does the ranking of configurations by energy agree
    with the ranking by time", pooling every architecture and dataset into one
    correlation per phase. That pooling is what makes it strong: a VGG-16 run on
    Tiny ImageNet is slower and costlier than a ResNet-18 run on Fashion-MNIST
    on any stack, so most of the agreement is the workload rather than the
    ecosystem.

    The question the manuscript actually needs is narrower: within one
    architecture on one dataset in one phase, where the workload is fixed and
    only the ecosystem varies, does time still order the stacks the way energy
    does? Same statistic, same estimator, one cell at a time. A cell that comes
    out below 1.0 is a cell where a practitioner timing their code would rank
    two stacks the wrong way round.
    """
    rows = []
    for (model, dataset, phase), g in runs.groupby(["model", "dataset", "phase"]):
        med = g.groupby("ecosystem")[["energy_j", "duration_s"]].median()
        if len(med) < 3:
            continue
        rho, p = stats.spearmanr(med.energy_j, med.duration_s)
        inversions = sum(
            1 for (i, a), (j, b) in itertools.combinations(med.reset_index().iterrows(), 2)
            if (a.energy_j - b.energy_j) * (a.duration_s - b.duration_s) < 0
        )
        total = len(med) * (len(med) - 1) // 2
        rows.append({
            "model": model, "dataset": dataset, "phase": phase,
            "n_ecosystems": len(med), "spearman_rho": round(float(rho), 3),
            "p": p, "discordant_pairs": inversions, "total_pairs": total,
        })
    return pd.DataFrame(rows, columns=["model", "dataset", "phase",
                                       "n_ecosystems", "spearman_rho", "p",
                                       "discordant_pairs", "total_pairs"])


def phase_consistency(runs: pd.DataFrame) -> pd.DataFrame:
    """Do training and inference rank the ecosystems the same way?

    The submitted analysis reported that they do not, and drew a
    "phase-aware decision making" recommendation from it. That claim is
    sensitive to the instrument: inference blocks are short enough to sit under
    the reported-window floor, so an estimator-derived inference ranking is
    partly a ranking of floors. Recomputed on counter-bracketed energy, the
    question is worth asking again.
    """
    rows = []
    for (model, dataset), g in runs.groupby(["model", "dataset"]):
        piv = g.pivot_table(index="ecosystem", columns="phase", values="energy_j",
                            aggfunc="median")
        if not {"Training", "Inference"}.issubset(piv.columns):
            continue
        piv = piv.dropna()
        if len(piv) < 3:
            continue
        rho, p = stats.spearmanr(piv["Training"], piv["Inference"])
        train_rank = list(piv["Training"].sort_values().index)
        infer_rank = list(piv["Inference"].sort_values().index)
        rows.append({
            "model": model, "dataset": dataset, "n_ecosystems": len(piv),
            "spearman_rho": round(rho, 3), "p": p,
            "same_best": train_rank[0] == infer_rank[0],
            "identical_order": train_rank == infer_rank,
            "best_training": train_rank[0], "best_inference": infer_rank[0],
        })
    return pd.DataFrame(rows)


def main() -> None:
    runs = run_totals()
    print(f"run-level observations: {len(runs)}")

    om = omnibus(runs)
    print("\n--- Kruskal-Wallis across ecosystems, per block ---")
    print(om.to_string(index=False))
    save_table(om, "v2_stats_omnibus",
               "Kruskal-Wallis on run totals, within each block")

    pw = pairwise(runs)
    sig = pw[pw.significant]
    print(f"\n--- pairwise: {len(sig)} of {len(pw)} comparisons significant after Holm ---")
    print(pw.magnitude.value_counts().to_string())
    save_table(pw, "v2_stats_pairwise",
               "Pairwise Mann-Whitney with Holm correction and Cliff's delta")

    ct = libtorch_control(runs)
    print("\n--- LibTorch control group ---")
    print(ct.to_string(index=False))
    save_table(ct, "v2_stats_libtorch_control",
               "Spread within the shared-backend control group against the full spread")

    et = energy_time(runs)
    print("\n--- energy against measured time ---")
    print(et.to_string(index=False))
    save_table(et, "v2_stats_energy_time",
               "Energy-time rank agreement on counter-bracketed durations")

    cr = cell_energy_time(runs)
    print("\n--- energy against time within each cell, ecosystems only ---")
    print(cr.to_string(index=False))
    save_table(cr, "v2_stats_cell_rho",
               "Energy-time rank agreement inside each block, where the "
               "workload is fixed and only the ecosystem varies")

    pc = phase_consistency(runs)
    print("\n--- do training and inference rank ecosystems alike? ---")
    print(pc.to_string(index=False))
    save_table(pc, "v2_stats_phase_consistency",
               "Training against inference ranking, per block")

    runs.to_csv(write_table_path("v2_run_totals.csv"), index=False)


if __name__ == "__main__":
    main()
