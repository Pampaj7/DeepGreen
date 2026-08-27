#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the ecosystem ranking survive the energy that nobody is charged for?

Energy is attributed only to the phases each stack brackets. Whatever a stack
does between them -- rebuilding a loader, compiling a graph, synchronising,
reshuffling on the host -- consumes real energy and appears in no measurement.
That would not matter if the fraction were similar across stacks. It is not:
measured coverage ranges from about 40% to almost 100% of a run's wall time.

The stacks that leave the most time untracked are therefore flattered by a
per-phase comparison, and by a large factor. This script asks whether the
ranking survives charging that time back.

We cannot measure the gaps retrospectively, so we bound them instead. For every
run, job-level energy is

    E_job = E_tracked + t_untracked * P_gap

and the ranking is computed across the whole admissible range of P_gap:

  * P_gap = 0        -- the per-phase comparison as reported. A lower bound: the
                        gaps cannot consume negative energy.
  * P_gap = P_min    -- each stack's own lowest observed block power. An upper
                        bound: a gap does strictly less work than the least busy
                        measured block of the same stack, so it cannot draw more.

If the ranking is the same at both ends it is the same everywhere between them,
and the conclusion does not depend on an unmeasured quantity. Where it is not,
the affected pairs are named.

Writes results/revision/tables/v2_coverage_sensitivity_*.{md,csv}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, save_table  # noqa: E402

TABLES = REPO_ROOT / "results" / "revision" / "tables"


def per_run() -> pd.DataFrame:
    """Tracked energy, untracked time and the gap-power bound, per run."""
    d = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    d["_ts"] = pd.to_datetime(d.timestamp, format="mixed", utc=True)
    d["block_power_w"] = d.hw_meas_j / d.duration_hw_s

    # The least busy measured block of a stack bounds the power its idle gaps
    # can draw: a gap does strictly less work than any block.
    p_min = d.groupby("ecosystem").block_power_w.min()

    rows = []
    for keys, g in d.groupby(["ecosystem", "model", "dataset", "repetition"]):
        g = g.sort_values("_ts")
        span = ((g._ts.max() - g._ts.min()).total_seconds()
                + float(g.duration_hw_s.iloc[0]))
        tracked_s = float(g.duration_hw_s.sum())
        rows.append({
            "ecosystem": keys[0], "model": keys[1], "dataset": keys[2],
            "repetition": keys[3],
            "tracked_j": float(g.hw_meas_j.sum()),
            "tracked_s": tracked_s,
            "untracked_s": max(0.0, span - tracked_s),
            "gap_power_bound_w": float(p_min[keys[0]]),
        })
    r = pd.DataFrame(rows)
    r["job_j_lower"] = r.tracked_j
    r["job_j_upper"] = r.tracked_j + r.untracked_s * r.gap_power_bound_w
    r["inflation"] = r.job_j_upper / r.job_j_lower
    return r


def ranking_stability(r: pd.DataFrame) -> pd.DataFrame:
    """Per block, the ecosystem order at each end of the admissible range."""
    rows = []
    for (model, dataset), g in r.groupby(["model", "dataset"]):
        lo = g.groupby("ecosystem").job_j_lower.median().sort_values()
        hi = g.groupby("ecosystem").job_j_upper.median().sort_values()
        common = [e for e in lo.index if e in hi.index]
        rho, _ = stats.spearmanr(lo[common].rank(), hi[common].rank())
        rows.append({
            "model": model, "dataset": dataset, "n_ecosystems": len(common),
            "order_per_phase": " < ".join(lo.index),
            "order_job_upper": " < ".join(hi.index),
            "identical": list(lo.index) == list(hi.index),
            "spearman_rho": round(float(rho), 3),
            "spread_per_phase": round(lo.max() / lo.min(), 1),
            "spread_job_upper": round(hi.max() / hi.min(), 1),
        })
    return pd.DataFrame(rows)


def by_ecosystem(r: pd.DataFrame) -> pd.DataFrame:
    t = (r.groupby("ecosystem")
         .agg(n_runs=("inflation", "size"),
              untracked_s_per_run=("untracked_s", "mean"),
              gap_power_bound_w=("gap_power_bound_w", "first"),
              max_inflation=("inflation", "mean"))
         .reset_index())
    t["coverage_pct"] = (100 * r.groupby("ecosystem")
                         .apply(lambda g: g.tracked_s.sum()
                                / (g.tracked_s.sum() + g.untracked_s.sum()),
                                include_groups=False)
                         .values)
    return t.round(2)


def main() -> None:
    r = per_run()
    print(f"runs: {len(r)}")

    be = by_ecosystem(r)
    print("\n--- untracked time and the most it could be worth ---")
    print(be.to_string(index=False))
    save_table(be, "v2_coverage_sensitivity_by_ecosystem",
               "Untracked time per run and the upper bound on its energy")

    rs = ranking_stability(r)
    print("\n--- does the ranking survive charging the gaps? ---")
    print(rs[["model", "dataset", "identical", "spearman_rho",
              "spread_per_phase", "spread_job_upper"]].to_string(index=False))
    save_table(rs, "v2_coverage_sensitivity_ranking",
               "Ecosystem order under the per-phase and job-level bounds")

    stable = int(rs.identical.sum())
    print(f"\nranking identical at both bounds in {stable} of {len(rs)} blocks")
    print(f"median rank correlation between the bounds: {rs.spearman_rho.median():.3f}")

    r.to_csv(TABLES / "v2_coverage_per_run.csv", index=False)


if __name__ == "__main__":
    main()
