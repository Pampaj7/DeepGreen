#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How busy the accelerator actually was, run by run.

The campaign reports energy and duration; it does not report whether the device
was working. That matters most for the stack the study calls expensive: R/torch
draws roughly half the power of the others and takes far longer, and those two
facts together are either a slow kernel or an idle card waiting for a host, and
energy alone cannot tell them apart. REVISION_LOG's open list asks for
per-stack utilisation next to energy for exactly this reason.

The input is results/gpu_utilisation.csv, a 1 Hz nvidia-smi record started in
commit 98e2774 -- utilisation, power, SM clock and memory in use. It began
2026-08-31 13:43 UTC, which is after the campaign's first run, so it does not
cover the whole campaign. This script says which runs it covers rather than
averaging over whatever happens to be there: a per-stack utilisation computed
from an unstated subset is the same defect as an energy figure computed from
one.

A run's window is [manifest machine_state.utc, counters.csv mtime] -- the
harness writes the manifest as it opens the run and appends to counters.csv for
the run's life -- and a run counts only if that whole window falls inside the
record.

Writes results/revision/tables/v2_gpu_utilisation_{by_run,by_ecosystem}.{md,csv}.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAMPAIGN_DIR, REPO_ROOT, announce_scope,  # noqa: E402
                    read_campaign_metrics, read_complete_counters, save_table)

RECORD = REPO_ROOT / "results" / "gpu_utilisation.csv"

BY_RUN_COLUMNS = ["run", "ecosystem", "model", "dataset", "repetition",
                  "n_samples", "util_mean_pct", "mem_mean_mib", "power_mean_w",
                  "power_min_w", "power_max_w"]


def run_windows() -> pd.DataFrame:
    """``(run, start, end)`` for every complete run, in UTC."""
    rows = []
    for run_dir in sorted(p for p in CAMPAIGN_DIR.glob("*") if p.is_dir()):
        if read_complete_counters(run_dir)[0] is None:
            continue
        counters = run_dir / "counters.csv"
        try:
            utc = (json.loads((run_dir / "manifest.json").read_text())
                   .get("machine_state") or {}).get("utc")
        except (OSError, ValueError):
            utc = None
        if not utc or not counters.exists():
            continue
        rows.append({
            "run": run_dir.name,
            "start": pd.Timestamp(utc).tz_convert("UTC"),
            # The harness appends to counters.csv until the run ends, so its
            # mtime is when the run finished.
            "end": pd.Timestamp(datetime.fromtimestamp(
                counters.stat().st_mtime, tz=timezone.utc)),
        })
    return pd.DataFrame(rows)


def per_run(record: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """One row per covered run, and the names of the runs the record misses."""
    identity = (read_campaign_metrics()
                .groupby("run")[["ecosystem", "model", "dataset", "repetition"]]
                .first())
    lo, hi = record.t.min(), record.t.max()
    rows, uncovered = [], []
    for _, w in windows.iterrows():
        if w.start < lo or w.end > hi:
            uncovered.append(w.run)
            continue
        s = record[(record.t >= w.start) & (record.t <= w.end)]
        if s.empty:
            uncovered.append(w.run)
            continue
        meta = identity.loc[w.run] if w.run in identity.index else {}
        rows.append({
            "run": w.run,
            "ecosystem": meta.get("ecosystem"), "model": meta.get("model"),
            "dataset": meta.get("dataset"), "repetition": meta.get("repetition"),
            "n_samples": len(s),
            "util_mean_pct": round(float(s.utilisation_pct.mean()), 1),
            "mem_mean_mib": round(float(s.memory_used_mib.mean()), 0),
            "power_mean_w": round(float(s.power_w.mean()), 1),
            "power_min_w": round(float(s.power_w.min()), 1),
            "power_max_w": round(float(s.power_w.max()), 1),
        })
    return pd.DataFrame(rows, columns=BY_RUN_COLUMNS), uncovered


def by_ecosystem(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(columns=["ecosystem", "model", "n_runs",
                                     "util_mean_pct", "util_min_pct",
                                     "util_max_pct", "mem_min_mib",
                                     "mem_max_mib", "power_min_w",
                                     "power_max_w"])
    g = runs.groupby(["ecosystem", "model"], observed=True)
    return g.agg(
        n_runs=("run", "size"),
        util_mean_pct=("util_mean_pct", "mean"),
        util_min_pct=("util_mean_pct", "min"),
        util_max_pct=("util_mean_pct", "max"),
        mem_min_mib=("mem_mean_mib", "min"),
        mem_max_mib=("mem_mean_mib", "max"),
        power_min_w=("power_mean_w", "min"),
        power_max_w=("power_mean_w", "max"),
    ).round(1).reset_index()


def main() -> int:
    announce_scope("19_gpu_utilisation")
    if not RECORD.exists():
        print(f"no utilisation record at {RECORD.relative_to(REPO_ROOT)}")
        return 0
    record = pd.read_csv(RECORD)
    record["t"] = pd.to_datetime(record.unix_s, unit="s", utc=True)
    print(f"record: {len(record):,} samples, {record.t.min()} -> {record.t.max()}")

    windows = run_windows()
    runs, uncovered = per_run(record, windows)
    print(f"covers {len(runs)} of {len(windows)} complete runs")
    if uncovered:
        # Named, not counted. The record began after the campaign did, and an
        # unstated subset is how a per-stack average becomes untraceable.
        print(f"  not covered ({len(uncovered)}), the record starting after "
              f"they ran: {', '.join(uncovered[:6])}"
              + (" ..." if len(uncovered) > 6 else ""))

    save_table(runs, "v2_gpu_utilisation_by_run",
               "Accelerator utilisation, memory and power per run, from the "
               "1 Hz record, over the runs it covers")
    summary = by_ecosystem(runs)
    print("\n--- utilisation by ecosystem and model ---")
    print(summary.to_string(index=False))
    save_table(summary, "v2_gpu_utilisation_by_ecosystem",
               "Accelerator utilisation per ecosystem and architecture; the "
               "number the energy tables do not carry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
