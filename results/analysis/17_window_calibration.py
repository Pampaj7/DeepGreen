#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How much does the machine drift between two measurement windows?

The campaign is interleaved, so drift in machine state is spread across
conditions rather than aliasing onto one ecosystem -- for most of the runs. Some
were executed in a later window, after a defect found mid-campaign was
corrected, and for those the design's central protection does not hold. How many
and which stack are derived, not pinned here: 12_paper_numbers.reexecution_facts
reads the timestamps and emits \\vLateRuns and \\vLateEcosystems. (The counts
this docstring used to give -- 185 and 25, "one stack's image loaders" -- were
the first campaign's, and were wrong for the second within a day of it starting,
which is the whole argument for deriving them.) The manuscript can either wave
at the later window with "we expect it to be small", which is a non-sequitur
(within-window variability bounds within-window noise, not between-window
drift), or measure it.

This measures it. One already-completed configuration is re-executed in a third
window, on the same machine, under the same idle conditions, with the same
seeds, writing to ``results/calibration/`` so the campaign it calibrates against
is untouched. A previous calibration must be moved aside first -- a rerun
produces exactly the same run-directory names, and run_campaign refuses a
directory whose counters.csv already has data (``--force`` would rmtree it, and
then the old calibration is gone rather than kept for comparison):

    mv results/calibration results/calibration_$(date +%Y%m%d)   # if one exists
    source scripts/campaign_env.sh
    DEEPGREEN_CAMPAIGN_DIR=$PWD/results/calibration \\
        python3 scripts/run_campaign.py --repetitions 5 \\
            --ecosystems Python/PyTorch --models resnet18 --datasets fashionmnist

Then this script compares the two, run for run, and reports the between-window
difference against the within-window spread. If the first is small relative to
the second, the later-window runs are usable and the threat is bounded rather
than merely acknowledged. It refuses the comparison outright when the two
windows were not produced under the same precision policy, or when the
calibration predates the campaign -- see comparability_objection.

Writes results/revision/tables/v2_window_calibration.{md,csv}.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, read_complete_counters, save_table,  # noqa: E402
                    TABLES_RESOLVER)

TABLES = TABLES_RESOLVER  # writes divert on a live campaign, reads fall back
CALIBRATION = REPO_ROOT / "results" / "calibration"
CAMPAIGN = REPO_ROOT / "results" / "campaign_v2"

# The columns this script owns, named so that a refusal still writes a header.
CALIBRATION_COLUMNS = ["ecosystem", "model", "dataset", "phase",
                       "n_original", "n_recheck", "original_J", "recheck_J",
                       "difference_pct", "within_window_cv_pct",
                       "difference_in_sd"]

# The precision policy both sides must agree on before a difference between
# them can be called drift. torch_cudnn_allow_tf32 is deliberately not in this
# list: the two manifest writers disagree about it (null against true) inside a
# single campaign, because only one of them can import torch to ask.
PRECISION_KEYS = ("DEEPGREEN_TF32", "NVIDIA_TF32_OVERRIDE")


def read_counters(run_dir: Path) -> pd.DataFrame | None:
    """A fragment is not a small measurement; see common.EXPECTED_EPOCHS."""
    hw, _ = read_complete_counters(run_dir)
    return hw


def complete_runs(root: Path) -> list[Path]:
    """The run directories under ``root`` that passed the completeness gate.

    Every question this script asks about a window has to be asked of the same
    runs. The guard walked every subdirectory while the collector below applied
    the gate, so one aborted run, one leftover, one ``.ipynb_checkpoints``
    under either root read as a run recording no precision policy -- and
    disabled the comparison permanently while blaming a key that every real run
    does record. A failed run directory was moved out of this campaign by hand
    during its execution, so this is not a hypothetical shape for the tree.
    """
    return [p for p in sorted(root.glob("*"))
            if p.is_dir() and read_complete_counters(p)[0] is not None]


def collect_calibration() -> pd.DataFrame:
    rows = []
    for run_dir in complete_runs(CALIBRATION):
        hw = read_counters(run_dir)
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


def precision_policy(run_dir: Path) -> dict[str, str | None]:
    """The TF32 policy a run recorded, from its manifest.

    Both manifest writers put it under ``machine_state.precision_policy``. The
    older harness wrote neither that block nor the variables, so a missing key
    is an answer -- "this run does not say" -- and not an error to swallow.
    """
    try:
        m = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        return {k: None for k in PRECISION_KEYS}
    policy = (m.get("machine_state") or {}).get("precision_policy") or {}
    env = m.get("env") or {}
    return {k: policy.get(k, env.get(k)) for k in PRECISION_KEYS}


def _policies(runs: list[Path]) -> dict[str, set]:
    """Every value each precision key takes across ``runs``."""
    seen: dict[str, set] = {k: set() for k in PRECISION_KEYS}
    for run_dir in runs:
        for k, v in precision_policy(run_dir).items():
            seen[k].add(None if v is None else str(v))
    return seen


def _run_start(run_dir: Path) -> float | None:
    """When a run started, as a POSIX timestamp.

    ``machine_state.utc`` is written when the harness opens the run, which is
    exactly the quantity this comparison wants. counters.csv's mtime stood in
    for it and is the wrong end of the run -- the harness appends to that file
    for its whole life, so the mtime is forty minutes late -- and worse, an
    mtime does not survive a copy, an rsync, or
    scripts/restore_from_replication.py, any of which would make the campaign
    look newer than a genuinely later calibration and refuse the comparison
    forever. The mtime is kept only for runs written before manifests carried
    machine_state.
    """
    try:
        m = json.loads((run_dir / "manifest.json").read_text())
        utc = (m.get("machine_state") or {}).get("utc")
        if utc:
            return datetime.fromisoformat(str(utc)).timestamp()
    except (OSError, ValueError):
        pass
    counters = run_dir / "counters.csv"
    return counters.stat().st_mtime if counters.exists() else None


def _run_starts(runs: list[Path]) -> list[float]:
    return [t for t in (_run_start(p) for p in runs) if t is not None]


def comparability_objection() -> str | None:
    """Why these two windows may not be compared, or None if they may.

    This script measures how far the machine drifts between two measurement
    windows, which requires that drift is the only thing that differs between
    them. It stopped being so, and nothing noticed. The calibration then on
    disk (now results/calibration_first_harness/) was produced on 2026-08-29 by
    the harness that pinned TF32 off for PyTorch, while every run of the second
    campaign records DEEPGREEN_TF32=1. The join still matched and the table was
    still written: training energy 201% apart at 293 standard deviations of the
    within-window spread -- the precision policy wearing a between-window-drift
    label, under a manuscript sentence calling it "below the noise the design
    already carries".

    A calibration that predates the campaign is a comparison across harness
    versions, not across windows: it cannot be re-executed to match a harness
    that did not exist when it ran, so the campaign is the fixed side and the
    calibration is the one that has to be produced again.
    """
    calib_runs, camp_runs = complete_runs(CALIBRATION), complete_runs(CAMPAIGN)
    if not calib_runs or not camp_runs:
        return (f"one window has no complete run to compare (calibration "
                f"{len(calib_runs)}, campaign {len(camp_runs)})")
    calib, camp = _policies(calib_runs), _policies(camp_runs)
    for key in PRECISION_KEYS:
        a, b = calib[key], camp[key]
        if None in a or None in b:
            side = "calibration" if None in a else "campaign"
            return (f"a complete {side} run does not record {key}, so the two "
                    f"windows cannot be shown to share a precision policy "
                    f"(calibration {sorted(a, key=str)}, campaign {sorted(b, key=str)})")
        if a != b:
            return (f"{key} differs between the windows: calibration "
                    f"{sorted(a)}, campaign {sorted(b)}")
    calib_t, camp_t = _run_starts(calib_runs), _run_starts(camp_runs)
    if calib_t and camp_t and max(calib_t) < min(camp_t):
        return ("every calibration run predates the campaign's earliest run, so "
                "they were produced by a different harness and the difference "
                "would be that change rather than drift")
    return None


def refuse(reason: str) -> int:
    """Say why, and write the empty table anyway.

    12_paper_numbers reads v2_window_calibration behind an ``.exists()`` check.
    Writing nothing does not read as "no result" there: the name resolves to the
    committed copy from the first campaign and the manuscript quotes a drift
    figure this run has just declined to compute. An empty table with its header
    is the only thing that says "asked, and there is no answer".
    """
    print(reason)
    save_table(pd.DataFrame(columns=CALIBRATION_COLUMNS), "v2_window_calibration",
               "One configuration re-executed in a later window, against its "
               "original runs")
    return 0


def main() -> int:
    if not CALIBRATION.exists():
        return refuse(f"no calibration runs under {CALIBRATION.relative_to(REPO_ROOT)}; "
                      "see this script's docstring for how to produce them")
    later = collect_calibration()
    if later.empty:
        return refuse("no complete calibration runs yet")

    objection = comparability_objection()
    if objection:
        return refuse(f"refusing to compare the two windows: {objection}. "
                      "Re-run the calibration against the current harness.")

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
    out = pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)
    if out.empty:
        return refuse("no configuration matched between the two windows")
    print(out.to_string(index=False))
    save_table(out, "v2_window_calibration",
               "One configuration re-executed in a later window, against its "
               "original runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
