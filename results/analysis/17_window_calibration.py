#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How much does the machine drift between two measurement windows?

The campaign is interleaved, so drift in machine state is spread across
conditions rather than aliasing onto one ecosystem -- for 185 of the 210 runs.
The remaining 25 were executed in a later window, after a defect in one stack's
image loaders was corrected, and for those the design's central protection does
not hold. The manuscript can either wave at that with "we expect it to be
small", which is a non-sequitur (within-window variability bounds within-window
noise, not between-window drift), or measure it.

This measures it. One already-completed configuration is re-executed in a third
window, on the same machine, under the same idle conditions, with the same
seeds, writing to ``results/calibration/`` so the campaign it calibrates against
is untouched:

    source scripts/campaign_env.sh
    DEEPGREEN_CAMPAIGN_DIR=$PWD/results/calibration \\
        python3 scripts/run_campaign.py --repetitions 5 \\
            --ecosystems Python/PyTorch --models resnet18 --datasets fashionmnist

Then this script compares the two, run for run, and reports the between-window
difference against the within-window spread. If the first is small relative to
the second, the later-window runs are usable and the threat is bounded rather
than merely acknowledged.

Writes results/revision/tables/v2_window_calibration.{md,csv}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, save_table  # noqa: E402

TABLES = REPO_ROOT / "results" / "revision" / "tables"
CALIBRATION = REPO_ROOT / "results" / "calibration"
EXPECTED_EPOCHS = 30


def read_counters(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "counters.csv"
    if not path.exists():
        return None
    hw = pd.read_csv(path)
    if hw.empty:
        return None
    counts = hw.groupby("phase").epoch.nunique()
    if not all(int(counts.get(ph, 0)) >= EXPECTED_EPOCHS for ph in ("train", "eval")):
        return None  # a fragment is not a small measurement
    return hw


def collect_calibration() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in CALIBRATION.glob("*") if p.is_dir()):
        hw = read_counters(run_dir)
        if hw is None:
            continue
        stem, rep = run_dir.name.rsplit("_rep", 1)
        eco, model, dataset = stem.split("_", 2)
        for phase, g in hw.groupby("phase"):
            rows.append({
                "ecosystem": eco.replace("-", "/"), "model": model,
                "dataset": dataset, "repetition": int(rep),
                "phase": {"train": "Training", "eval": "Inference"}[phase],
                "energy_j": float(g.hw_total_j.mean()),
                "duration_s": float(g.duration_s.mean()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    if not CALIBRATION.exists():
        print(f"no calibration runs under {CALIBRATION.relative_to(REPO_ROOT)}; "
              "see this script's docstring for how to produce them")
        return 0
    later = collect_calibration()
    if later.empty:
        print("no complete calibration runs yet")
        return 0

    epochs = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    epochs["ecosystem"] = epochs.ecosystem.replace({"Cpp/LibTorch": "C++/LibTorch"})
    later["ecosystem"] = later.ecosystem.replace({"Python/PyTorch": "Python/PyTorch"})
    original = (epochs.groupby(["ecosystem", "model", "dataset", "repetition", "phase"])
                .agg(energy_j=("hw_total_j", "mean"),
                     duration_s=("duration_hw_s", "mean"))
                .reset_index())

    rows = []
    keys = ["ecosystem", "model", "dataset", "phase"]
    for key, g in later.groupby(keys):
        o = original
        for k, v in zip(keys, key):
            o = o[o[k] == v]
        if o.empty:
            continue
        a, b = o.energy_j.to_numpy(), g.energy_j.to_numpy()
        # Within-window spread, as the yardstick the difference is measured in.
        pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        rows.append({
            "ecosystem": key[0], "model": key[1], "dataset": key[2],
            "phase": key[3],
            "n_original": len(a), "n_recheck": len(b),
            "original_J": round(float(a.mean()), 1),
            "recheck_J": round(float(b.mean()), 1),
            "difference_pct": round(100 * (b.mean() - a.mean()) / a.mean(), 2),
            "within_window_cv_pct": round(100 * float(a.std(ddof=1) / a.mean()), 2),
            "difference_in_sd": round(abs(b.mean() - a.mean()) / pooled_sd, 2)
            if pooled_sd else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        print("no configuration matched between the two windows")
        return 0
    print(out.to_string(index=False))
    save_table(out, "v2_window_calibration",
               "One configuration re-executed in a later window, against its "
               "original runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
