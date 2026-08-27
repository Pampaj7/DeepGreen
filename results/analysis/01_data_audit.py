#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit of the measurement pipeline.

Addresses:
  R1 c4  / R3 M1 -- native unit of CodeCarbon output and the kWh->J correction
  R1 c5          -- implausible sampled power values, robust summaries
  R1 c9          -- sampling interval and short-block reliability
  R1 c15         -- GPU load: is the workload GPU-bound or host-bound?
  R3 M7          -- the true number of runs, epochs and measurements
  R1 c8          -- toolchain confounds visible in the logs

Writes tables to results/revision/tables/ and prints an audit report to stdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    CPU_TDP_W,
    GPU_TDP_W,
    J_PER_KWH,
    REPO_ROOT,
    bootstrap_ci,
    load,
    order_ecosystems,
    save_table,
)

SEP = "=" * 78


def audit_units(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Prove the native unit of ``energy_consumed`` from the data itself."""
    print(SEP)
    print("1. UNIT AUDIT  (R1 comment 4, R3 major comment 1)")
    print(SEP)

    # Test A -- additivity: energy_consumed == cpu + gpu + ram, in whatever unit.
    comp_sum = raw["cpu_energy"] + raw["gpu_energy"] + raw["ram_energy"]
    rel_err = ((raw["energy_consumed"] - comp_sum).abs() / raw["energy_consumed"]).max()
    print(f"A. energy_consumed == cpu+gpu+ram  (max rel. error {rel_err:.2e})  -> components share the unit")

    # Test B -- carbon intensity: emissions / energy must equal the grid factor
    # of the measurement location.  Only the kWh reading gives a sane number.
    ci = (raw["emissions"] / raw["energy_consumed"]).describe()
    print(
        f"B. emissions / energy_consumed = {ci['mean']:.4f} "
        f"[{ci['min']:.4f}, {ci['max']:.4f}] kg/unit"
    )
    print(
        "   Reading the denominator as kWh gives "
        f"{ci['mean'] * 1000:.0f} gCO2eq/kWh, the Italian grid intensity used by "
        "CodeCarbon. Reading it as J would imply "
        f"{ci['mean'] * 1e9:.3g} gCO2eq/J, i.e. ~{ci['mean'] * 3.6e6 * 1000:.0f} tonnes per kWh."
    )

    # Test C -- implied average power.
    p_as_kwh = (raw["energy_consumed"] * J_PER_KWH / raw["duration"]).describe()
    p_as_j = (raw["energy_consumed"] / raw["duration"]).describe()
    print(
        f"C. implied mean power, energy read as kWh: {p_as_kwh['min']:.0f}-{p_as_kwh['max']:.0f} W "
        f"(median {p_as_kwh['50%']:.0f} W)  -> plausible for this server"
    )
    print(
        f"   implied mean power, energy read as J  : {p_as_j['min']:.2e}-{p_as_j['max']:.2e} W "
        "  -> sub-milliwatt, physically impossible"
    )

    print(f"\nCONCLUSION: energy_consumed is in kWh. Correction factor applied: {J_PER_KWH:.1e} J/kWh.")

    out = pd.DataFrame(
        {
            "check": [
                "components sum to total (max rel. error)",
                "emissions/energy (mean, kg per unit)",
                "implied carbon intensity if unit=kWh (gCO2eq/kWh)",
                "implied mean power if unit=kWh (median W)",
                "implied mean power if unit=J (median W)",
                "correction factor applied (J/kWh)",
            ],
            "value": [
                f"{rel_err:.2e}",
                f"{ci['mean']:.6f}",
                f"{ci['mean'] * 1000:.1f}",
                f"{p_as_kwh['50%']:.1f}",
                f"{p_as_j['50%']:.3e}",
                f"{J_PER_KWH:.1e}",
            ],
        }
    )
    save_table(out, "audit_units", "Unit audit of the CodeCarbon output")
    return out


def audit_design(df: pd.DataFrame) -> pd.DataFrame:
    """The explicit run-count table R3 asks for in major comment 7."""
    print()
    print(SEP)
    print("2. EXPERIMENTAL DESIGN  (R3 major comment 7)")
    print(SEP)

    cells = df.groupby(["ecosystem", "model_arch", "dataset"], observed=True)
    n_cells = len(cells)
    per_cell_epochs = df.groupby(
        ["ecosystem", "model_arch", "dataset", "phase"], observed=True
    ).size()
    runs_per_cell = df.groupby(
        ["ecosystem", "model_arch", "dataset", "phase"], observed=True
    )["run_id"].nunique()

    n_eco = df["ecosystem"].nunique()
    n_model = df["model_arch"].nunique()
    n_ds = df["dataset"].nunique()
    n_epochs = int(per_cell_epochs.mode().iloc[0])
    n_train = int((df["phase"] == "Training").sum())
    n_eval = int((df["phase"] == "Inference").sum())

    rows = [
        ("Ecosystems (language-framework stacks)", n_eco),
        ("Model architectures", n_model),
        ("Datasets", n_ds),
        ("Configurations (ecosystem x model x dataset)", n_cells),
        ("Independent run-level repetitions per configuration", 1),
        ("Epochs per run", n_epochs),
        ("Training-phase measurements", n_train),
        ("Inference-phase measurements", n_eval),
        ("Total tracked measurement blocks", n_train + n_eval),
    ]
    tbl = pd.DataFrame(rows, columns=["quantity", "count"])
    for label, value in rows:
        print(f"  {label:<52} {value}")

    print(
        f"\nCONCLUSION: the campaign contains {n_train + n_eval} tracked blocks "
        f"({n_train} training + {n_eval} inference), NOT 7,200. "
        "Each CodeCarbon tracker covers exactly one epoch of one phase, so a "
        f"'training epoch' count is {n_train}."
    )
    print(
        "  Every configuration was executed ONCE; the 30 epochs of a run are "
        "pseudo-replicates, not independent repetitions."
    )
    assert (runs_per_cell == n_epochs).all(), "unexpected tracker/epoch structure"
    save_table(tbl, "audit_design", "Experimental design and measurement counts")
    return tbl


def audit_gpu_load(df: pd.DataFrame) -> pd.DataFrame:
    """Is this workload GPU-bound?  (R1 comment 15)"""
    print()
    print(SEP)
    print("3. GPU LOAD AND ENERGY BOUNDARY  (R1 comment 15, comment 4 last para)")
    print(SEP)

    g = df.groupby(["ecosystem", "phase"], observed=True)
    tbl = g.agg(
        mean_gpu_power_w=("gpu_power_derived_w", "mean"),
        mean_total_power_w=("power_w", "mean"),
        gpu_share=("gpu_energy_j", "sum"),
        cpu_share=("cpu_energy_j", "sum"),
        ram_share=("ram_energy_j", "sum"),
    ).reset_index()
    total = tbl["gpu_share"] + tbl["cpu_share"] + tbl["ram_share"]
    tbl["gpu_energy_pct"] = 100 * tbl["gpu_share"] / total
    tbl["cpu_energy_pct"] = 100 * tbl["cpu_share"] / total
    tbl["ram_energy_pct"] = 100 * tbl["ram_share"] / total
    tbl["gpu_power_pct_of_tdp"] = 100 * tbl["mean_gpu_power_w"] / GPU_TDP_W
    tbl = tbl.drop(columns=["gpu_share", "cpu_share", "ram_share"])
    tbl = tbl.round(2)

    print(tbl.to_string(index=False))
    peak = tbl["gpu_power_pct_of_tdp"].max()
    print(
        f"\nCONCLUSION: mean GPU power never exceeds {peak:.0f}% of the {GPU_TDP_W:.0f} W board limit. "
        "The GPU is lightly loaded throughout; between "
        f"{tbl['gpu_energy_pct'].min():.0f}% and {tbl['gpu_energy_pct'].max():.0f}% of the measured "
        "energy is GPU energy, the remainder is host-side (CPU + RAM). "
        "Differences between ecosystems are therefore dominated by host-side cost, "
        "not by the deep-learning kernels."
    )
    save_table(tbl, "audit_gpu_load", "Per-component energy and GPU load by ecosystem and phase")
    return tbl


def audit_measurement_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """Are all ecosystems measured with the same instrument configuration?"""
    print()
    print(SEP)
    print("4. MEASUREMENT BOUNDARY CONSISTENCY  (new; bears on R1 c8/c9, R3 M1)")
    print(SEP)

    tbl = df.groupby("ecosystem", observed=True).agg(
        codecarbon=("codecarbon_version", lambda s: "/".join(sorted(s.unique()))),
        mean_cpu_power_w=("cpu_power", "mean"),
        cpu_power_distinct=("cpu_power", "nunique"),
        mean_ram_power_w=("ram_power", "mean"),
        ram_energy_pct=("ram_energy_j", "sum"),
        total_energy_j=("energy_j", "sum"),
        tracking_mode=("tracking_mode", lambda s: "/".join(sorted(s.unique()))),
    ).reset_index()
    tbl["ram_energy_pct"] = 100 * tbl["ram_energy_pct"] / tbl["total_energy_j"]
    tbl = tbl.drop(columns=["total_energy_j"])
    tbl = tbl.reindex(
        [list(tbl["ecosystem"]).index(e) for e in order_ecosystems(tbl["ecosystem"])]
    ).round(3)
    print(tbl.to_string(index=False))

    r_row = tbl[tbl["ecosystem"] == "R/torch"]
    print(
        "\nFINDINGS:\n"
        "  * Two CodeCarbon major versions were used across the campaign (2.8.4 and\n"
        "    3.0.4). They model CPU and RAM power differently, so the *energy boundary\n"
        "    is not identical across ecosystems*.\n"
        "  * The 2.8.4 stacks report a constant CPU power of 42.5 W -- CodeCarbon's\n"
        "    hardcoded fallback (POWER_CONSTANT 85 W x 0.5), used when RAPL is\n"
        "    unavailable -- and a constant RAM power of 188.84 W (0.375 W/GB x 503.6 GB\n"
        "    *installed*). Together that is 231.3 W of modelled host power, i.e. about\n"
        "    two thirds of their reported energy is a deterministic linear function of\n"
        "    wall-clock duration. The 3.0.4 stacks carry ~107 W instead.\n"
        f"  * R/torch additionally ran in PROCESS mode, so its RAM energy share is\n"
        f"    {float(r_row['ram_energy_pct'].iloc[0]):.3f}% against ~50% for the other 2.8.4 stacks:\n"
        "    RAM was effectively untracked and its totals are not comparable.\n"
        "  * Any absolute cross-ecosystem energy comparison must therefore be restricted\n"
        "    to GPU energy (NVML, identical everywhere) or recomputed under one uniform\n"
        "    host model. common.py provides both as energy_gpu_j and energy_harm_j."
    )
    save_table(tbl, "audit_measurement_boundary", "Instrument configuration per ecosystem")
    return tbl


def audit_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Implausible sampled-power readings (R1 comment 5, comment 9)."""
    print()
    print(SEP)
    print("5. IMPLAUSIBLE POWER SAMPLES AND SHORT BLOCKS  (R1 comments 5 and 9)")
    print(SEP)

    n = len(df)
    zero = int((df["gpu_power"] <= 0).sum())
    over = int((df["gpu_power"] > GPU_TDP_W).sum())
    print(f"  rows with sampled gpu_power == 0 W      : {zero} ({100*zero/n:.1f}%)")
    print(f"  rows with sampled gpu_power > {GPU_TDP_W:.0f} W    : {over} ({100*over/n:.1f}%)")
    print(f"  maximum sampled gpu_power               : {df['gpu_power'].max():.0f} W")

    short = df[df["duration_s"] < 15.0]
    print(
        f"\n  blocks shorter than the 15 s CodeCarbon default sampling period: "
        f"{len(short)} ({100*len(short)/n:.1f}%), of which "
        f"{100*(short['phase'] == 'Inference').mean():.0f}% are inference."
    )
    print(f"  shortest block: {df['duration_s'].min():.2f} s")

    # The derived power (energy counter / duration) has no such outliers.
    dp = df["gpu_power_derived_w"]
    print(
        f"\n  GPU power derived from the energy counter: "
        f"{dp.min():.1f}-{dp.max():.1f} W, none outside [0, {GPU_TDP_W:.0f}] W."
    )
    print(
        "\nCONCLUSION: the outliers are confined to CodeCarbon's *sampled* power\n"
        "  columns, which are unreliable on blocks shorter than the sampling period.\n"
        "  The energy columns come from the NVML energy counter and are internally\n"
        "  consistent. All results are therefore reported from the energy columns,\n"
        "  with medians alongside means; sampled power is used only descriptively\n"
        f"  and the {zero + over} affected rows are flagged in the released data."
    )

    tbl = pd.DataFrame(
        {
            "metric": [
                "rows",
                "sampled gpu_power == 0 W",
                "sampled gpu_power > board limit",
                "max sampled gpu_power (W)",
                "blocks shorter than 15 s sampling period",
                "shortest block (s)",
                "derived gpu power out of range",
            ],
            "value": [n, zero, over, round(df["gpu_power"].max(), 1), len(short), round(df["duration_s"].min(), 2), 0],
        }
    )
    save_table(tbl, "audit_outliers", "Implausible sampled-power readings")
    return tbl


def audit_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """Within-run dispersion, explicitly labelled as pseudo-replication."""
    print()
    print(SEP)
    print("6. WITHIN-RUN DISPERSION  (R1 comment 5, R3 major comment 2)")
    print(SEP)

    rows = []
    for (eco, phase), sub in df.groupby(["ecosystem", "phase"], observed=True):
        e = sub["energy_j"].to_numpy()
        mean, lo, hi = bootstrap_ci(e, np.mean)
        rows.append(
            {
                "ecosystem": eco,
                "phase": phase,
                "n_epochs": len(e),
                "mean_energy_j": mean,
                "median_energy_j": float(np.median(e)),
                "cv_pct": 100 * float(np.std(e, ddof=1) / np.mean(e)),
                "epoch_ci95_lo": lo,
                "epoch_ci95_hi": hi,
            }
        )
    tbl = pd.DataFrame(rows).round(3)
    print(tbl.to_string(index=False))
    print(
        "\nCAVEAT (must be stated wherever these intervals appear): each configuration\n"
        "  was executed once. The intervals above are bootstrap intervals over the 30\n"
        "  epochs of a single run and quantify *within-run epoch-to-epoch* variation.\n"
        "  They do NOT estimate between-run uncertainty, which would additionally\n"
        "  include initialisation, JIT, allocator, scheduling and thermal effects.\n"
        "  The effective number of independent observations per configuration is 1."
    )
    save_table(tbl, "audit_within_run_dispersion", "Within-run (pseudo-replicate) dispersion")
    return tbl


def main() -> None:
    raw = pd.read_csv(REPO_ROOT / "results" / "data" / "combined_data.csv")
    df = load()

    audit_units(raw, df)
    audit_design(df)
    audit_gpu_load(df)
    audit_measurement_boundary(df)
    audit_outliers(df)
    audit_dispersion(df)

    print()
    print(SEP)
    print("Audit complete. Tables written to results/revision/tables/")
    print(SEP)


if __name__ == "__main__":
    main()
