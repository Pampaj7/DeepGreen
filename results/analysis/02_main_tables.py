#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected main results tables (paper Tables 3 and 4) plus the
instrument-sensitivity table that supersedes them.

Addresses:
  R1 c4  / R3 M1 -- energy reported in Joules, correctly converted
  R1 c5  / R3 M2 -- median alongside mean, dispersion, pseudo-replicate CIs
  R1 c4 (last)   -- per-component CPU/GPU/RAM breakdown
  R1 c8, c15     -- ranking under a common measurement boundary
  R3 M8          -- ecosystem naming
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    ENERGY_DEFS,
    HOST_POWER_W,
    bootstrap_ci,
    load,
    order_ecosystems,
    save_table,
)


def phase_table(df: pd.DataFrame, phase: str, energy_col: str) -> pd.DataFrame:
    sub = df[df["phase"] == phase]
    rows = []
    for eco, s in sub.groupby("ecosystem", observed=True):
        e = s[energy_col].to_numpy()
        mean, lo, hi = bootstrap_ci(e, np.mean)
        rows.append(
            {
                "ecosystem": eco,
                "mean_energy_J": mean,
                "ci95_lo_J": lo,
                "ci95_hi_J": hi,
                "median_energy_J": float(np.median(e)),
                "iqr_J": float(np.percentile(e, 75) - np.percentile(e, 25)),
                "cv_pct": 100 * float(np.std(e, ddof=1) / np.mean(e)),
                "mean_duration_s": float(s["duration_s"].mean()),
                "median_duration_s": float(s["duration_s"].median()),
                "mean_power_W": float(s[energy_col].sum() / s["duration_s"].sum()),
            }
        )
    tbl = pd.DataFrame(rows).sort_values("mean_energy_J").reset_index(drop=True)
    tbl.insert(1, "rank_energy", np.arange(1, len(tbl) + 1))
    tbl["rank_time"] = tbl["mean_duration_s"].rank().astype(int)
    tbl["x_vs_best"] = tbl["mean_energy_J"] / tbl["mean_energy_J"].min()
    return tbl


def component_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eco, phase), s in df.groupby(["ecosystem", "phase"], observed=True):
        n = len(s)
        rows.append(
            {
                "ecosystem": eco,
                "phase": phase,
                "cpu_J_per_epoch": s["cpu_energy_j"].sum() / n,
                "gpu_J_per_epoch": s["gpu_energy_j"].sum() / n,
                "ram_J_per_epoch": s["ram_energy_j"].sum() / n,
                "total_J_per_epoch": s["energy_j"].sum() / n,
                "gpu_pct": 100 * s["gpu_energy_j"].sum() / s["energy_j"].sum(),
                "host_pct": 100
                * (s["cpu_energy_j"].sum() + s["ram_energy_j"].sum())
                / s["energy_j"].sum(),
            }
        )
    tbl = pd.DataFrame(rows)
    order = order_ecosystems(tbl["ecosystem"].unique())
    tbl["_o"] = tbl["ecosystem"].map({e: i for i, e in enumerate(order)})
    return tbl.sort_values(["phase", "_o"]).drop(columns="_o").reset_index(drop=True)


def sensitivity_table(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    sub = df[df["phase"] == phase]
    out = sub.groupby("ecosystem", observed=True).agg(
        **{
            "as_measured_J": ("energy_j", "mean"),
            "gpu_only_J": ("energy_gpu_j", "mean"),
            "harmonised_J": ("energy_harm_j", "mean"),
            "duration_s": ("duration_s", "mean"),
        }
    )
    for c in ["as_measured_J", "gpu_only_J", "harmonised_J", "duration_s"]:
        out["rank_" + c.replace("_J", "").replace("_s", "")] = out[c].rank().astype(int)
    out = out.sort_values("as_measured_J").reset_index()
    return out


def spread_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase in ["Training", "Inference"]:
        sub = df[df["phase"] == phase]
        for col, label in ENERGY_DEFS.items():
            m = sub.groupby("ecosystem", observed=True)[col].mean()
            rows.append(
                {
                    "phase": phase,
                    "energy_definition": label,
                    "best": m.idxmin(),
                    "worst": m.idxmax(),
                    "spread_x": m.max() / m.min(),
                }
            )
        t = sub.groupby("ecosystem", observed=True)["duration_s"].mean()
        rows.append(
            {
                "phase": phase,
                "energy_definition": "execution time (s)",
                "best": t.idxmin(),
                "worst": t.idxmax(),
                "spread_x": t.max() / t.min(),
            }
        )
    return pd.DataFrame(rows).round(2)


def main() -> None:
    df = load()

    print("=" * 78)
    print("CORRECTED MAIN TABLES  (energy in Joules)")
    print("=" * 78)

    for phase, tag in [("Training", "table3_training"), ("Inference", "table4_inference")]:
        t = phase_table(df, phase, "energy_j").round(3)
        print(f"\n--- {phase}: per-epoch energy, as measured ---")
        print(t.to_string(index=False))
        save_table(
            t,
            tag + "_as_measured",
            f"{phase}: per-epoch energy (J) as measured. "
            "CI95 is a bootstrap over the 30 epochs of a single run "
            "(pseudo-replicates), not a between-run interval.",
        )

        t2 = phase_table(df, phase, "energy_harm_j").round(3)
        save_table(
            t2,
            tag + "_harmonised",
            f"{phase}: per-epoch energy (J) under the harmonised boundary "
            f"(GPU counter + {HOST_POWER_W:.0f} W uniform host model).",
        )

    print("\n--- per-component breakdown (J/epoch) ---")
    comp = component_table(df).round(2)
    print(comp.to_string(index=False))
    save_table(comp, "table_component_breakdown", "CPU / GPU / RAM energy per epoch")

    print("\n--- ranking sensitivity to the energy definition ---")
    for phase in ["Training", "Inference"]:
        s = sensitivity_table(df, phase).round(1)
        print(f"\n{phase}:")
        print(s.to_string(index=False))
        save_table(
            s,
            f"table_ranking_sensitivity_{phase.lower()}",
            f"{phase}: ecosystem ranking under three energy definitions",
        )

    print("\n--- headline spreads ---")
    sp = spread_summary(df)
    print(sp.to_string(index=False))
    save_table(sp, "table_headline_spreads", "Best/worst ecosystem and spread by metric")

    print(
        "\nNOTE: the manuscript's headline '4.6x training / 7.3x inference' figures are\n"
        "  the as-measured spreads. Under the harmonised boundary they become "
        f"{sp.loc[(sp.phase=='Training') & sp.energy_definition.str.startswith('harmonised'),'spread_x'].iloc[0]:.1f}x and "
        f"{sp.loc[(sp.phase=='Inference') & sp.energy_definition.str.startswith('harmonised'),'spread_x'].iloc[0]:.1f}x,\n"
        "  and the identity of the best and worst ecosystem changes in three of the\n"
        "  four phase/definition combinations. The rankings are not robust to the\n"
        "  measurement boundary."
    )


if __name__ == "__main__":
    main()
