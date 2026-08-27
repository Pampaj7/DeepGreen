#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregation and analysis for the replicated campaign (results/campaign_v2/).

Addresses, once the campaign has been executed:
  R1 c5  / R3 M2 -- the independent run becomes the unit of analysis, so
                    confidence intervals and tests describe between-run
                    uncertainty rather than epoch autocorrelation
  R1 c6  / R3 M3 -- energy normalised by achieved model quality: accuracy per
                    kilojoule, and energy to reach a target accuracy

Run ``python3 scripts/run_campaign.py --repetitions 5`` first. If no campaign
data is present this script explains what is missing and exits cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import J_PER_KWH, REPO_ROOT, bootstrap_ci, save_table  # noqa: E402

CAMPAIGN_DIR = REPO_ROOT / "results" / "campaign_v2"
TARGET_ACCURACY = {  # per dataset, chosen well below the achievable ceiling
    "fashionmnist": 85.0,
    "cifar100": 30.0,
    "tinyimagenet": 20.0,
}


# A run that died part-way leaves its directory behind holding the few blocks it
# managed. Those are fragments, not small measurements, and averaging them in
# with complete runs produced ecosystem spreads three orders of magnitude too
# large. See the same gate in 11_instrument_comparison.py.
EXPECTED_EPOCHS = int(os.environ.get("DEEPGREEN_EPOCHS", "30"))


def _is_complete(run_dir: Path) -> bool:
    counters = run_dir / "counters.csv"
    if not counters.exists():
        return False
    try:
        hw = pd.read_csv(counters)
    except pd.errors.EmptyDataError:
        return False
    if hw.empty:
        return False
    counts = hw.groupby("phase").epoch.nunique()
    return all(int(counts.get(ph, 0)) >= EXPECTED_EPOCHS for ph in ("train", "eval"))


def collect() -> pd.DataFrame:
    """Join per-epoch CodeCarbon output with per-epoch quality metrics."""
    rows = []
    skipped = []
    for run_dir in sorted(p for p in CAMPAIGN_DIR.glob("*") if p.is_dir()):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        if not _is_complete(run_dir):
            skipped.append(run_dir.name)
            continue
        metrics = pd.read_csv(metrics_path)
        for emissions_path in sorted(run_dir.glob("emissions_*.csv")):
            stem = emissions_path.stem  # emissions_<phase>_epoch<N>
            _, phase, epoch_tag = stem.split("_", 2)
            epoch = int(epoch_tag.replace("epoch", ""))
            e = pd.read_csv(emissions_path)
            if e.empty:
                continue
            rec = e.iloc[-1].to_dict()
            rec.update(
                {
                    "run_dir": run_dir.name,
                    "phase": {"train": "Training", "eval": "Inference"}[phase],
                    "epoch": epoch,
                }
            )
            rows.append(rec)
    if skipped:
        print(f"excluded {len(skipped)} incomplete run(s): {', '.join(skipped)}")
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["energy_j"] = df["energy_consumed"].astype(float) * J_PER_KWH
    df["gpu_energy_j"] = df["gpu_energy"].astype(float) * J_PER_KWH
    df["duration_s"] = df["duration"].astype(float)

    meta = []
    for run_dir in sorted(p for p in CAMPAIGN_DIR.glob("*") if p.is_dir()):
        mp = run_dir / "metrics.csv"
        if mp.exists():
            m = pd.read_csv(mp)
            m["run_dir"] = run_dir.name
            meta.append(m)
    quality = pd.concat(meta, ignore_index=True) if meta else pd.DataFrame()
    if not quality.empty:
        df = df.merge(quality, on=["run_dir", "epoch"], how="left", suffixes=("", "_q"))
    return df


def between_run_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one value per RUN, then summarise across runs.

    This is the step the first campaign could not take: with a single run per
    configuration there was nothing to aggregate over.
    """
    per_run = (
        df.groupby(["ecosystem", "model", "dataset", "repetition", "phase"], observed=True)
        .agg(run_energy_j=("energy_j", "mean"), run_duration_s=("duration_s", "mean"))
        .reset_index()
    )
    rows = []
    for (eco, model, ds, phase), s in per_run.groupby(
        ["ecosystem", "model", "dataset", "phase"], observed=True
    ):
        e = s["run_energy_j"].to_numpy()
        mean, lo, hi = bootstrap_ci(e, np.mean)
        rows.append(
            {
                "ecosystem": eco,
                "model": model,
                "dataset": ds,
                "phase": phase,
                "n_runs": len(e),
                "mean_energy_J": mean,
                "ci95_lo_J": lo,
                "ci95_hi_J": hi,
                "sd_J": float(np.std(e, ddof=1)) if len(e) > 1 else np.nan,
                "cv_pct": 100 * float(np.std(e, ddof=1) / np.mean(e)) if len(e) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def quality_normalised(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per kilojoule and energy to reach a target accuracy."""
    rows = []
    for (eco, model, ds, rep), s in df.groupby(
        ["ecosystem", "model", "dataset", "repetition"], observed=True
    ):
        s = s.sort_values("epoch")
        train = s[s["phase"] == "Training"]
        if "test_acc" not in s.columns or train.empty:
            continue
        acc_by_epoch = s.groupby("epoch")["test_acc"].max()
        final_acc = float(acc_by_epoch.dropna().iloc[-1]) if acc_by_epoch.notna().any() else np.nan
        cum_energy = train.sort_values("epoch")["energy_j"].cumsum()
        total_energy = float(cum_energy.iloc[-1])

        target = TARGET_ACCURACY.get(str(ds).lower())
        e_to_target = np.nan
        if target is not None and acc_by_epoch.notna().any():
            reached = acc_by_epoch[acc_by_epoch >= target]
            if not reached.empty:
                ep = int(reached.index[0])
                e_to_target = float(
                    train[train["epoch"] <= ep]["energy_j"].sum()
                )
        rows.append(
            {
                "ecosystem": eco,
                "model": model,
                "dataset": ds,
                "repetition": rep,
                "final_test_acc_pct": final_acc,
                "train_energy_total_J": total_energy,
                "acc_per_kJ": final_acc / (total_energy / 1000) if total_energy else np.nan,
                "target_acc_pct": target,
                "energy_to_target_J": e_to_target,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 78)
    print("REPLICATED CAMPAIGN ANALYSIS  (R1 c5/c6, R3 M2/M3)")
    print("=" * 78)

    if not CAMPAIGN_DIR.exists() or not any(CAMPAIGN_DIR.glob("*/metrics.csv")):
        print(
            "\nNo replicated campaign data found under results/campaign_v2/.\n"
            "\nThis script is the analysis half of the fix for the two criticisms that\n"
            "cannot be answered from the existing logs: independent repetitions and\n"
            "model quality. Neither exists in the first campaign's data, so both\n"
            "require re-execution on the measurement server:\n"
            "\n    python3 scripts/run_campaign.py --repetitions 5 --print-plan\n"
            "    python3 scripts/run_campaign.py --repetitions 5\n"
            "\nThe Python stacks run directly; the C++, Java, R, MATLAB and Rust stacks\n"
            "must be driven from plan.json through their own build systems, honouring\n"
            "the repetition index and seed. See results/analysis/repetition_protocol.md.\n"
        )
        return

    df = collect()
    print(f"\ncollected {len(df)} measurement blocks from {df['run_dir'].nunique()} runs")

    brs = between_run_stats(df).round(3)
    print("\n--- between-run statistics (the independent run is the unit) ---")
    print(brs.to_string(index=False))
    save_table(brs, "v2_between_run_statistics",
               "Between-run energy statistics; CI95 is a genuine run-level interval")

    qn = quality_normalised(df).round(3)
    if not qn.empty:
        print("\n--- energy normalised by achieved quality ---")
        print(qn.to_string(index=False))
        save_table(qn, "v2_quality_normalised",
                   "Final accuracy, accuracy per kJ, and energy to reach a target accuracy")
        summary = (
            qn.groupby(["ecosystem", "dataset"], observed=True)[
                ["final_test_acc_pct", "acc_per_kJ", "energy_to_target_J"]
            ]
            .agg(["mean", "std"])
            .round(3)
        )
        print("\n--- summary across repetitions ---")
        print(summary.to_string())
        save_table(summary.reset_index(), "v2_quality_summary",
                   "Quality-normalised efficiency, mean and sd across repetitions")


if __name__ == "__main__":
    main()
