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
from common import REPO_ROOT, cliffs_delta, save_table  # noqa: E402

TABLES = REPO_ROOT / "results" / "revision" / "tables"
LIBTORCH = {"Python/PyTorch", "Cpp/LibTorch", "C++/LibTorch", "R/torch", "Rust/tch"}


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

    Four of the seven ecosystems run on LibTorch, and under specification S1 the
    Python, C++ and Rust members of that group train the identical TorchScript
    module. Whatever separates them is therefore binding and host overhead, not
    kernels. Comparing that spread with the full spread bounds how much of the
    headline effect is attributable to the backend rather than the ecosystem.
    """
    rows = []
    train = runs[runs.phase == "Training"]
    for (model, dataset), g in train.groupby(["model", "dataset"]):
        med = g.groupby("ecosystem").energy_j.median()
        ctrl = med[[e for e in med.index if e in LIBTORCH]]
        if len(ctrl) < 2:
            continue
        full_spread = med.max() / med.min()
        ctrl_spread = ctrl.max() / ctrl.min()
        rows.append({
            "model": model, "dataset": dataset,
            "n_all": len(med), "n_libtorch": len(ctrl),
            "spread_all": round(full_spread, 2),
            "spread_libtorch": round(ctrl_spread, 2),
            "share_of_log_spread_pct": round(
                100 * np.log(ctrl_spread) / np.log(full_spread), 1),
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

    pc = phase_consistency(runs)
    print("\n--- do training and inference rank ecosystems alike? ---")
    print(pc.to_string(index=False))
    save_table(pc, "v2_stats_phase_consistency",
               "Training against inference ranking, per block")

    runs.to_csv(TABLES / "v2_run_totals.csv", index=False)


if __name__ == "__main__":
    main()
