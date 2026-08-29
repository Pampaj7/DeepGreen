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
        raw_gap = g._ts.diff().dt.total_seconds() - g.duration_hw_s
        gap = raw_gap.clip(lower=0)
        out.append(g.assign(gap_s=gap, gap_raw_s=raw_gap))
    a = pd.concat(out)
    a["window_excess_s"] = a.duration_cc_s - a.duration_hw_s
    return a


def gap_is_instrument(a: pd.DataFrame) -> pd.DataFrame:
    """Is the gap between blocks the tracker's window overhead?

    The pooled correlation is high, and on its own it flatters the claim: it is
    largely a two-group contrast between padded and unpadded blocks. Within the
    padded blocks alone it is much weaker, and within the unpadded blocks it is
    nothing at all. Both are reported. So is the resolution of the timestamps
    the gaps are computed from -- one second, against a median gap of about
    three -- because that is the precision this attribution actually has.
    """
    g = a.dropna(subset=["gap_s"])
    padded = g.window_excess_s > 0.5
    ts = pd.to_datetime(a.timestamp, format="mixed", utc=True)
    rows = [{
        "n_blocks": len(g),
        "pearson_r": round(float(g.gap_s.corr(g.window_excess_s)), 4),
        "pearson_r_padded_only": round(
            float(g[padded].gap_s.corr(g[padded].window_excess_s)), 4),
        "pearson_r_unpadded_only": round(
            float(g[~padded].gap_s.corr(g[~padded].window_excess_s)), 4),
        "timestamp_resolution_s": 1 if ts.dt.microsecond.nunique() == 1 else 0,
        # Negative before clipping: one-second timestamps against three-second
        # gaps. This is what the attribution can actually resolve.
        "negative_gap_pct": round(100 * float((g.gap_raw_s < 0).mean()), 1),
        "median_gap_s": round(float(g.gap_s.median()), 3),
        "median_window_excess_s": round(float(g.window_excess_s.median()), 3),
        "median_abs_difference_s": round(
            float((g.gap_s - g.window_excess_s).abs().median()), 3),
        "share_of_gap_explained_pct": round(
            100 * float(g.window_excess_s.sum() / g.gap_s.sum()), 1),
    }]
    return pd.DataFrame(rows)


def window_model(a: pd.DataFrame) -> pd.DataFrame:
    """What the reported window actually is, and why.

    ``EmissionsTracker.stop()`` resolves cloud metadata and then the host's
    geolocation over the network -- two blocking requests, made after the final
    energy reading, which is why only the duration is affected. Both are cached
    on the tracker, and a background thread makes the same call once
    ``api_call_interval`` measurements have accumulated (8, at one second). So a
    block pays for however many lookups remain outstanding when it ends: three
    modes, and a threshold near eleven seconds.
    ``scripts/probe_reported_window.py`` measures that directly.

    Two models were fitted here before the mechanism was known, and then both
    were rejected -- the second one wrongly. A floor, max(phase, 3.99 s),
    R^2 = 0.993, is wrong in the middle of the range. A threshold, phase plus a
    constant below 11 s, R^2 = 0.998, is essentially right, and the mechanism
    says why. The rejection rested on exceptions we could not explain, of which
    the load-bearing one was void: R/torch is unpadded in all 1800 of its blocks
    and its shortest block is 11.8 s, so it sits above the threshold and is
    consistent with the model rather than against it. The real exception is C++
    training, padded at durations where every other stack is not, and it is
    still unexplained.

    This function reports the modes and their occupancy, and keeps both rejected
    fits beside them, because the route matters: a high R^2 on a skewed
    predictor is not evidence of the right functional form, and a scatter of
    exceptions is not evidence against one. Only a mechanism settled it.
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
    by_len = (a.assign(bin=pd.cut(a.duration_hw_s, np.arange(0, 30, 1)))
              .groupby("bin", observed=True).window_excess_s.median())
    collapsed = by_len[by_len < 1.0]
    threshold = float(collapsed.index[0].left) if len(collapsed) else np.nan

    # The modes, found as gaps in the sorted excess rather than assumed.
    ex = np.sort(a.window_excess_s.to_numpy())
    cuts = [0.0]
    for lo, hi in zip(ex[:-1], ex[1:]):
        if hi - lo > 0.5:                     # a gap wider than any mode's width
            cuts.append((lo + hi) / 2)
    cuts.append(float(ex[-1]) + 1.0)
    modes = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        m = a[(a.window_excess_s > lo) & (a.window_excess_s <= hi)]
        if m.empty:
            continue
        modes.append({
            "mode_excess_s": round(float(m.window_excess_s.median()), 3),
            "n_blocks": len(m),
            "share_pct": round(100 * len(m) / len(a), 1),
            "spread_s": round(float(m.window_excess_s.max()
                                    - m.window_excess_s.min()), 3),
            "median_phase_s": round(float(m.duration_hw_s.median()), 2),
        })
    out = pd.DataFrame(modes).sort_values("mode_excess_s").reset_index(drop=True)

    padded = a.window_excess_s > 0.5
    out.attrs["const"] = const
    out.attrs["threshold"] = threshold
    out.attrs["rejected_fits"] = pd.DataFrame([
        score(np.maximum(phase, 3.99), "max(phase, 3.99 s)"),
        score(np.where(phase < threshold, phase + const, phase),
              f"phase + {const:.2f} s if phase < {threshold:.0f} s, else phase"),
    ])
    # The exceptions that rule the threshold out, counted.
    out.attrs["padded_above_threshold"] = int((padded & (a.duration_hw_s > threshold)).sum())
    out.attrs["unpadded_below_threshold"] = int((~padded & (a.duration_hw_s < threshold)).sum())
    out.attrs["never_padded_ecosystems"] = sorted(
        a.groupby("ecosystem").apply(lambda g: (g.window_excess_s > 0.5).mean(),
                                     include_groups=False)
        .pipe(lambda s: s[s == 0].index.tolist()))
    return out


def power_distortion(a: pd.DataFrame) -> pd.DataFrame:
    """The consequence, stated without a model.

    Whatever governs the padding, its effect on a derived quantity is directly
    measurable: divide each instrument's own energy by its own duration and
    compare. No fit, no threshold, no functional form to get wrong.
    """
    g = a[(a.duration_cc_s > 0) & (a.duration_hw_s > 0)].copy()
    # Like for like. Using CodeCarbon's *total* here puts its modelled RAM term
    # in the numerator, which is why the long-block rows came out at 0.93 --
    # the estimator overstating power because of the RAM model, not because of
    # the duration. This table is about the duration, so compare the terms both
    # instruments measure. The RAM term is criticised on its own two pages on.
    g["power_reported_w"] = g.cc_meas_j / g.duration_cc_s
    g["power_measured_w"] = g.hw_meas_j / g.duration_hw_s
    g["bin"] = pd.cut(g.duration_hw_s, [0, 0.5, 1, 2, 5, 10, 30, np.inf],
                      labels=["<0.5 s", "0.5-1 s", "1-2 s", "2-5 s", "5-10 s",
                              "10-30 s", ">30 s"])
    out = (g.groupby("bin", observed=True)
           .apply(lambda d: pd.Series({
               "n_blocks": len(d),
               "median_phase_s": round(float(d.duration_hw_s.median()), 2),
               "reported_power_w": round(float(d.power_reported_w.median()), 1),
               "measured_power_w": round(float(d.power_measured_w.median()), 1),
               "understated_by": round(float((d.power_measured_w
                                              / d.power_reported_w).median()), 2),
           }), include_groups=False)
           .reset_index())
    return out


def coverage(a: pd.DataFrame) -> pd.DataFrame:
    """Tracked share of the measured interval, and how much of the rest is the tracker.

    The denominator is the span from the first block's start to the last
    block's end, NOT the process's wall time: start-up, dataset construction,
    module load, JIT or XLA warm-up outside epoch 1, and teardown all fall
    outside it and are excluded from both numerator and denominator. Calling
    this "coverage of wall time" would overstate what is accounted for, and
    those excluded phases are exactly the ecosystem-side work an alternative
    reading of this table would want to invoke.

    The instrument share is reported unclamped. Where it exceeds 100% -- it does
    for one ecosystem -- the accounting has overshot, because CodeCarbon's
    timestamps have one-second resolution and the gaps here are of order three
    seconds. That is the honest precision of this attribution and the table says
    so rather than rounding it away.
    """
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
    print("\n--- what the reported window actually is: three modes ---")
    print(wm.to_string(index=False))
    print(f"  never padded: {wm.attrs['never_padded_ecosystems']}")
    print(f"  padded above the would-be threshold:   "
          f"{wm.attrs['padded_above_threshold']} blocks")
    print(f"  unpadded below the would-be threshold: "
          f"{wm.attrs['unpadded_below_threshold']} blocks")
    save_table(wm, "v2_coverage_window_model",
               "The three modes of CodeCarbon's reported-duration excess")

    rejected = wm.attrs["rejected_fits"]
    rejected["padded_above_threshold"] = wm.attrs["padded_above_threshold"]
    rejected["unpadded_below_threshold"] = wm.attrs["unpadded_below_threshold"]
    print("\n--- the two fits we rejected, and what their R^2 concealed ---")
    print(rejected.to_string(index=False))
    save_table(rejected, "v2_coverage_window_rejected_fits",
               "Length-based models of the reported duration, and why they fail")

    dist = power_distortion(a)
    print("\n--- power from the reported fields, by phase length ---")
    print(dist.to_string(index=False))
    save_table(dist, "v2_coverage_power_distortion",
               "Derived power from the estimator's own fields against the counters")

    cov = coverage(a)
    print("\n--- coverage, and how much of the shortfall is the instrument ---")
    print(cov.to_string(index=False))
    save_table(cov, "v2_coverage_by_ecosystem",
               "Tracked share of wall time, and the instrument's share of the rest")

    a.to_csv(TABLES / "v2_coverage_per_block.csv", index=False)


if __name__ == "__main__":
    main()
