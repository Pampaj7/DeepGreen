#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency Index: sensitivity, normalisation, and the Pareto alternative.

Addresses:
  R1 c10 / R3 M4 -- alpha is arbitrary, normalisation is inconsistent between
                    the tables (divide by min) and the heatmap (divide by max),
                    and energy and time are correlated so the composite
                    double-counts. Reports an alpha sweep, a single consistent
                    normalisation, and the Pareto set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import load, save_table


def index_table(df: pd.DataFrame, phase: str, energy_col: str, alphas) -> pd.DataFrame:
    sub = df[df["phase"] == phase]
    g = sub.groupby("ecosystem", observed=True)
    e = g[energy_col].mean()
    t = g["duration_s"].mean()
    # Single consistent normalisation: divide by the minimum, so 1.0 = best and
    # the value reads as "x times the best ecosystem".
    en = e / e.min()
    tn = t / t.min()
    out = pd.DataFrame({"ecosystem": e.index, "norm_energy": en.values, "norm_time": tn.values})
    for a in alphas:
        out[f"EI_alpha_{a:.2f}"] = a * en.values + (1 - a) * tn.values
    for a in alphas:
        out[f"rank_alpha_{a:.2f}"] = out[f"EI_alpha_{a:.2f}"].rank().astype(int)
    return out.sort_values(f"EI_alpha_0.50").reset_index(drop=True)


def pareto_set(df: pd.DataFrame, phase: str, energy_col: str) -> pd.DataFrame:
    sub = df[df["phase"] == phase]
    g = sub.groupby("ecosystem", observed=True)
    pts = pd.DataFrame({"energy_J": g[energy_col].mean(), "duration_s": g["duration_s"].mean()})
    dominated = []
    for a in pts.index:
        dom = any(
            (pts.loc[b, "energy_J"] <= pts.loc[a, "energy_J"])
            and (pts.loc[b, "duration_s"] <= pts.loc[a, "duration_s"])
            and (b != a)
            and (
                pts.loc[b, "energy_J"] < pts.loc[a, "energy_J"]
                or pts.loc[b, "duration_s"] < pts.loc[a, "duration_s"]
            )
            for b in pts.index
        )
        dominated.append(dom)
    pts["pareto_optimal"] = ~np.array(dominated)
    return pts.reset_index()


def main() -> None:
    df = load()
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("=" * 78)
    print("EFFICIENCY INDEX: SENSITIVITY AND ALTERNATIVES  (R1 c10, R3 M4)")
    print("=" * 78)

    for phase in ["Training", "Inference"]:
        t = index_table(df, phase, "energy_harm_j", alphas)
        rank_cols = [c for c in t.columns if c.startswith("rank_alpha")]
        spread = t[rank_cols].max(axis=1) - t[rank_cols].min(axis=1)
        t["rank_range_over_alpha"] = spread
        print(f"\n--- {phase} (harmonised energy, normalised by the minimum) ---")
        print(t.round(3).to_string(index=False))
        rho = stats.spearmanr(t["norm_energy"], t["norm_time"]).statistic
        print(
            f"  Spearman(normalised energy, normalised time) = {rho:.3f}. "
            "The two components are near-collinear, so the composite adds little\n"
            f"  beyond either one: the ranking moves by at most {int(spread.max())} position(s) "
            "as alpha sweeps 0 -> 1."
        )
        save_table(
            t.round(6),
            f"efficiency_index_alpha_sweep_{phase.lower()}",
            f"{phase}: Efficiency Index under alpha in {alphas}, normalised by the minimum "
            "(1.0 = best) throughout, in contrast to the manuscript which normalises by "
            "the minimum in the tables and by the maximum in the heatmap.",
        )

        p = pareto_set(df, phase, "energy_harm_j").round(2)
        print(f"\n  Pareto set ({phase}): "
              f"{', '.join(p.loc[p.pareto_optimal, 'ecosystem'])}")
        save_table(p, f"pareto_set_{phase.lower()}", f"{phase}: Pareto-optimal ecosystems")

    print(
        "\nRECOMMENDATION: with alpha fixed at 0.5 the index is the unweighted mean of\n"
        "  two near-collinear normalised quantities, and it is insensitive to alpha over\n"
        "  the whole range. It should either be dropped in favour of reporting the\n"
        "  Pareto set directly, or retained only with this sweep shown."
    )


if __name__ == "__main__":
    main()
