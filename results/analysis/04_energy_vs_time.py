#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3: is execution time a reliable proxy for energy?

Addresses:
  R3 M4  -- replaces the qualitative quadrant plot with a quantified analysis
  R1 c12 -- decomposes the difference into duration and mean power
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import ENERGY_DEFS, load, save_table


def within_ecosystem_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eco, phase), s in df.groupby(["ecosystem", "phase"], observed=True):
        for col in ENERGY_DEFS:
            r_p, p_p = stats.pearsonr(s["duration_s"], s[col])
            r_s, p_s = stats.spearmanr(s["duration_s"], s[col])
            rows.append(
                {
                    "ecosystem": eco,
                    "phase": phase,
                    "energy_definition": col,
                    "pearson_r": r_p,
                    "r_squared": r_p**2,
                    "spearman_rho": r_s,
                    "p_spearman": p_s,
                }
            )
    return pd.DataFrame(rows)


def rank_inversions(df: pd.DataFrame) -> pd.DataFrame:
    """How often does the energy ranking disagree with the time ranking?"""
    rows = []
    for phase, sub in df.groupby("phase", observed=True):
        t = sub.groupby("ecosystem", observed=True)["duration_s"].mean()
        for col, label in ENERGY_DEFS.items():
            e = sub.groupby("ecosystem", observed=True)[col].mean()
            common = t.index
            rho, p = stats.spearmanr(t[common], e[common])
            tau, ptau = stats.kendalltau(t[common], e[common])
            # count discordant ordered pairs
            names = list(common)
            disc = [
                (a, b)
                for i, a in enumerate(names)
                for b in names[i + 1 :]
                if (t[a] - t[b]) * (e[a] - e[b]) < 0
            ]
            rows.append(
                {
                    "phase": phase,
                    "energy_definition": label,
                    "spearman_rho": rho,
                    "kendall_tau": tau,
                    "discordant_pairs": len(disc),
                    "total_pairs": len(names) * (len(names) - 1) // 2,
                    "examples": "; ".join(f"{a} vs {b}" for a, b in disc[:3]),
                }
            )
    return pd.DataFrame(rows)


def power_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """energy = mean power x duration.  Which factor drives the spread?"""
    rows = []
    for phase, sub in df.groupby("phase", observed=True):
        g = sub.groupby("ecosystem", observed=True)
        e = g["energy_harm_j"].mean()
        t = g["duration_s"].mean()
        p = e / t
        rows.append(
            {
                "phase": phase,
                "energy_spread_x": e.max() / e.min(),
                "duration_spread_x": t.max() / t.min(),
                "mean_power_spread_x": p.max() / p.min(),
                "log_share_duration_pct": 100 * np.log(t.max() / t.min()) / np.log(e.max() / e.min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    print("=" * 78)
    print("RQ3: ENERGY vs EXECUTION TIME  (R3 major comment 4, R1 comment 12)")
    print("=" * 78)

    wc = within_ecosystem_correlation(df)
    piv = (
        wc[wc["energy_definition"] == "energy_j"]
        .pivot(index="ecosystem", columns="phase", values="r_squared")
        .round(4)
    )
    print("\n--- within-ecosystem R^2 of energy on duration (as measured) ---")
    print(piv.to_string())
    print(
        f"\n  Median within-ecosystem R^2 = {wc[wc.energy_definition=='energy_j'].r_squared.median():.3f}.\n"
        "  Within a single ecosystem, energy is very nearly a deterministic linear\n"
        "  function of wall-clock time. This is expected once the audit is taken into\n"
        "  account: for the CodeCarbon 2.8.4 stacks about two thirds of the reported\n"
        "  energy is a constant host power multiplied by duration."
    )
    save_table(wc.round(6), "rq3_within_ecosystem_correlation",
               "Energy-duration correlation inside each ecosystem")

    ri = rank_inversions(df)
    print("\n--- cross-ecosystem agreement between the energy and time rankings ---")
    print(ri.round(4).to_string(index=False))
    save_table(ri.round(6), "rq3_rank_inversions",
               "Agreement between the energy ranking and the time ranking")

    pdc = power_decomposition(df).round(2)
    print("\n--- decomposition: energy = mean power x duration ---")
    print(pdc.to_string(index=False))
    save_table(pdc, "rq3_power_decomposition", "Share of the energy spread attributable to duration")

    print(
        "\nCONCLUSION (supersedes the manuscript's RQ3 claim):\n"
        "  * Within an ecosystem, execution time predicts energy almost perfectly\n"
        f"    (median R^2 = {wc[wc.energy_definition=='energy_j'].r_squared.median():.2f}).\n"
        "  * Across ecosystems, the as-measured energy ranking does disagree with the\n"
        "    time ranking, but under a harmonised measurement boundary the\n"
        "    disagreement shrinks to a single pair in training and disappears\n"
        "    entirely in inference.\n"
        "  * The 'faster is not greener' examples given in the manuscript (R vs Java,\n"
        "    MATLAB vs PyTorch) are artefacts of the differing CodeCarbon host power\n"
        "    models, not of ecosystem behaviour.\n"
        "  * The spread is driven mostly by duration, not by mean power\n"
        f"    ({pdc['log_share_duration_pct'].min():.0f}-{pdc['log_share_duration_pct'].max():.0f}% of the log spread), "
        "which is consistent with a lightly loaded GPU."
    )


if __name__ == "__main__":
    main()
