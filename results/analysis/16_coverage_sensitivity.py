#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whose energy is the time between measured phases?

Energy is attributed only to the phases each stack brackets, and the stacks
differ enormously in how much of a run that is: tracked time ranges from about
40% of wall time to almost 100%. The obvious reading is that the low-coverage
stacks do a lot of work between phases and are flattered by a per-phase
comparison.

That reading is wrong, and this script is what establishes it. The gap before
each block is almost exactly the amount by which CodeCarbon's reported window
exceeds the counter-bracketed phase: the two agree to a couple of hundred
milliseconds and correlate at r > 0.97 over every block in the campaign. The
untracked time is the *instrument* holding its window open, not the ecosystem
computing anything. It would not exist in an uninstrumented run, and charging it
to the ecosystems would be charging them for the cost of being measured.

Coverage is therefore reported here as a property of the apparatus rather than
of the stacks -- and it is a substantial one. Over a campaign of this size the
tracker's own overhead accounts for a large share of wall-clock time, which
matters for anyone planning machine-mode measurement: whole-machine tracking
attributes that time's energy to nobody, and the shorter your blocks the larger
the share.

Writes results/revision/tables/v2_coverage_*.{md,csv}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, save_table  # noqa: E402

TABLES = REPO_ROOT / "results" / "revision" / "tables"


def per_block() -> pd.DataFrame:
    """Every block, with the gap that precedes it and the instrument's excess."""
    d = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    d["_ts"] = pd.to_datetime(d.timestamp, format="mixed", utc=True)
    out = []
    for _, g in d.groupby(["ecosystem", "model", "dataset", "repetition"]):
        g = g.sort_values("_ts")
        gap = (g._ts.diff().dt.total_seconds() - g.duration_hw_s).clip(lower=0)
        out.append(g.assign(gap_s=gap))
    a = pd.concat(out)
    a["window_excess_s"] = a.duration_cc_s - a.duration_hw_s
    return a


def gap_is_instrument(a: pd.DataFrame) -> pd.DataFrame:
    """Is the gap between blocks the tracker's window overhead?"""
    g = a.dropna(subset=["gap_s"])
    rows = [{
        "n_blocks": len(g),
        "pearson_r": round(float(g.gap_s.corr(g.window_excess_s)), 4),
        "median_gap_s": round(float(g.gap_s.median()), 3),
        "median_window_excess_s": round(float(g.window_excess_s.median()), 3),
        "median_abs_difference_s": round(
            float((g.gap_s - g.window_excess_s).abs().median()), 3),
        "share_of_gap_explained_pct": round(
            100 * float(g.window_excess_s.sum() / g.gap_s.sum()), 1),
    }]
    return pd.DataFrame(rows)


def window_model(a: pd.DataFrame) -> pd.DataFrame:
    """What the reported window actually is, fitted rather than assumed.

    A first pass fitted ``max(phase, floor)`` and got R^2 = 0.993, which looked
    conclusive. It is not: that model is carried by the very short and the very
    long blocks, and it is wrong in the middle, where the excess is a constant
    rather than a floor. The excess is in fact bimodal -- about 3.27 s or about
    13 ms, with the transition near eleven seconds -- and a two-regime model
    fits substantially better on the same data.
    """
    y = a.duration_cc_s.to_numpy()
    phase = a.duration_hw_s.to_numpy()

    def score(pred, name):
        ss_res = float(((pred - y) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return {"model": name,
                "r_squared": round(1 - ss_res / ss_tot, 4),
                "mean_abs_error_s": round(float(np.abs(pred - y).mean()), 3)}

    short = a[a.duration_hw_s < 8]
    const = float(short.window_excess_s.median())
    # transition: the shortest phase length above which the excess collapses
    by_len = (a.assign(bin=pd.cut(a.duration_hw_s, np.arange(0, 30, 1)))
              .groupby("bin", observed=True).window_excess_s.median())
    collapsed = by_len[by_len < 1.0]
    threshold = float(collapsed.index[0].left) if len(collapsed) else np.nan

    rows = [
        score(np.maximum(phase, 3.99), "max(phase, 3.99 s)"),
        score(phase + const, f"phase + {const:.2f} s"),
        score(np.where(phase < threshold, phase + const, phase),
              f"phase + {const:.2f} s if phase < {threshold:.0f} s, else phase"),
    ]
    out = pd.DataFrame(rows)
    out.attrs["const"] = const
    out.attrs["threshold"] = threshold
    return out


def coverage(a: pd.DataFrame) -> pd.DataFrame:
    """Tracked share of wall time, and how much of the shortfall is the tracker."""
    rows = []
    for keys, g in a.groupby(["ecosystem", "model", "dataset", "repetition"]):
        g = g.sort_values("_ts")
        span = ((g._ts.max() - g._ts.min()).total_seconds()
                + float(g.duration_hw_s.iloc[0]))
        tracked = float(g.duration_hw_s.sum())
        rows.append({
            "ecosystem": keys[0],
            "span_s": span, "tracked_s": tracked,
            "untracked_s": max(0.0, span - tracked),
            "instrument_s": float(g.window_excess_s.sum()),
            "median_block_s": float(g.duration_hw_s.median()),
        })
    r = pd.DataFrame(rows)
    t = (r.groupby("ecosystem")
         .agg(n_runs=("span_s", "size"), tracked_s=("tracked_s", "sum"),
              untracked_s=("untracked_s", "sum"),
              instrument_s=("instrument_s", "sum"),
              median_block_s=("median_block_s", "median"))
         .reset_index())
    t["coverage_pct"] = (100 * t.tracked_s / (t.tracked_s + t.untracked_s)).round(1)
    t["untracked_that_is_instrument_pct"] = (
        100 * t.instrument_s / t.untracked_s).round(1)
    return t[["ecosystem", "n_runs", "median_block_s", "coverage_pct",
              "untracked_that_is_instrument_pct"]].round(2)


def main() -> None:
    a = per_block()

    gi = gap_is_instrument(a)
    print("--- is the gap between blocks the tracker's own window? ---")
    print(gi.to_string(index=False))
    save_table(gi, "v2_coverage_gap_attribution",
               "The gap before each block against the instrument's window excess")

    wm = window_model(a)
    print("\n--- what the reported window actually is ---")
    print(wm.to_string(index=False))
    save_table(wm, "v2_coverage_window_model",
               "Competing models of CodeCarbon's reported duration")

    cov = coverage(a)
    print("\n--- coverage, and how much of the shortfall is the instrument ---")
    print(cov.to_string(index=False))
    save_table(cov, "v2_coverage_by_ecosystem",
               "Tracked share of wall time, and the instrument's share of the rest")

    a.to_csv(TABLES / "v2_coverage_per_block.csv", index=False)


if __name__ == "__main__":
    main()
