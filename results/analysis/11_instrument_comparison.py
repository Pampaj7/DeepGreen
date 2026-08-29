#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-instrument comparison over the whole replicated campaign.

Every measured epoch in ``results/campaign_v2/`` carries two independent
readings of the same interval:

  * ``counters.csv``   -- NVML ``nvmlDeviceGetTotalEnergyConsumption`` for the
                          GPU and Intel RAPL for the CPU package, i.e. energy
                          counters read by the harness at the exact phase
                          boundaries (``tools/hardware_counters.py``)
  * ``emissions_*.csv`` -- CodeCarbon's own accounting for the same window

The original submission reported CodeCarbon numbers alone, and Reviewer #3 (M5)
asked what those numbers actually are. This script answers it with the whole
campaign rather than a spot check: per-instrument ratios by ecosystem, model,
dataset and phase, plus the domains where the two instruments are measuring
genuinely different things (CodeCarbon adds a modelled RAM term that no counter
on this machine can confirm).

Writes results/revision/tables/v2_instrument_*.{md,csv}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import J_PER_KWH, REPO_ROOT, save_table  # noqa: E402

CAMPAIGN_DIR = REPO_ROOT / "results" / "campaign_v2"

# A run that died part-way leaves a directory behind, with a counters.csv
# holding the handful of blocks it managed before the crash. Twenty of the
# campaign's runs did exactly that, and their partial totals -- 16 J against a
# neighbouring stack's 300 kJ -- are not small measurements of the same thing,
# they are fragments of a run that never happened. Including them silently
# produced ecosystem spreads of 20,000x. Completeness is therefore a gate, not a
# judgement call: a run counts only if it recorded every epoch of both phases.
EXPECTED_EPOCHS = int(__import__("os").environ.get("DEEPGREEN_EPOCHS", "30"))
PHASES = ("train", "eval")


def parse_run_name(name: str) -> dict:
    """``Rust-tch_resnet18_fashionmnist_rep3`` -> its four fields."""
    eco, model, dataset, rep = name.rsplit("_", 3)
    return {
        "ecosystem": eco.replace("-", "/", 1),
        "model": model,
        "dataset": dataset,
        "repetition": int(rep.replace("rep", "")),
    }


def collect() -> pd.DataFrame:
    """One row per (run, phase, epoch) with both instruments side by side."""
    rows: list[dict] = []
    incomplete: list[tuple[str, dict]] = []
    for run_dir in sorted(p for p in CAMPAIGN_DIR.glob("*") if p.is_dir()):
        counters_path = run_dir / "counters.csv"
        if not counters_path.exists():
            continue
        try:
            hw = pd.read_csv(counters_path)
        except pd.errors.EmptyDataError:
            continue
        if hw.empty:
            continue

        # -- completeness gate -------------------------------------------
        counts = hw.groupby("phase").epoch.nunique()
        missing = [ph for ph in PHASES
                   if int(counts.get(ph, 0)) < EXPECTED_EPOCHS]
        if missing:
            incomplete.append(
                (run_dir.name,
                 {ph: int(counts.get(ph, 0)) for ph in PHASES}))
            continue

        meta = parse_run_name(run_dir.name)
        for _, h in hw.iterrows():
            phase = str(h["phase"])
            epoch = int(h["epoch"])
            cc_path = run_dir / f"emissions_{phase}_epoch{epoch}.csv"
            if not cc_path.exists():
                continue
            try:
                cc = pd.read_csv(cc_path)
            except pd.errors.EmptyDataError:
                continue
            if cc.empty:
                continue
            c = cc.iloc[-1]

            rows.append(
                {
                    **meta,
                    "phase": {"train": "Training", "eval": "Inference"}[phase],
                    "epoch": epoch,
                    "duration_hw_s": float(h["duration_s"]),
                    "duration_cc_s": float(c["duration"]),
                    # hardware counters, already in joules
                    "hw_gpu_j": float(h["gpu_j"]),
                    "hw_cpu_j": float(h.get("cpu_package_total_j", np.nan)),
                    "hw_total_j": float(h["hw_total_j"]),
                    # CodeCarbon, kWh on disk
                    "cc_gpu_j": float(c["gpu_energy"]) * J_PER_KWH,
                    "cc_cpu_j": float(c["cpu_energy"]) * J_PER_KWH,
                    "cc_ram_j": float(c["ram_energy"]) * J_PER_KWH,
                    "cc_total_j": float(c["energy_consumed"]) * J_PER_KWH,
                    "cc_cpu_power_w": float(c["cpu_power"]),
                    "cc_ram_power_w": float(c["ram_power"]),
                    "timestamp": str(c["timestamp"]),
                }
            )

    if incomplete:
        print(f"excluded {len(incomplete)} incomplete run(s) "
              f"(fewer than {EXPECTED_EPOCHS} epochs in a phase):")
        for name, counts in incomplete:
            print(f"    {name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        print()

    if not rows:
        print(f"No campaign data under {CAMPAIGN_DIR}. Run scripts/run_campaign.py first.")
        raise SystemExit(0)

    df = pd.DataFrame(rows)
    # The comparable quantity is GPU + CPU package: it is the only part both
    # instruments claim to *measure*. cc_ram_j is a linear model of installed
    # DIMM capacity, so it is reported separately rather than folded in.
    df["hw_meas_j"] = df["hw_gpu_j"] + df["hw_cpu_j"]
    df["cc_meas_j"] = df["cc_gpu_j"] + df["cc_cpu_j"]
    for dom in ("gpu", "cpu", "meas"):
        df[f"ratio_{dom}"] = df[f"cc_{dom}_j"] / df[f"hw_{dom}_j"]
    df["ratio_total"] = df["cc_total_j"] / df["hw_total_j"]
    df["ram_share_pct"] = 100.0 * df["cc_ram_j"] / df["cc_total_j"]
    return df


def measurement_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each run actually lies inside a measurement block?

    Energy is attributed only to the phases each stack brackets. Anything a
    stack does between them -- rebuilding a loader, synchronising, reshuffling
    on the host -- is real work that consumes real energy and is charged to
    nobody. If that fraction differed a lot between ecosystems it would bias the
    comparison in favour of whichever stack does more of its work outside its
    own phase boundaries, so it has to be reported rather than assumed small.

    Coverage is the tracked time as a fraction of the elapsed wall time from the
    start of a run's first block to the end of its last, which excludes one-time
    start-up and shutdown and isolates the per-epoch gaps.

    CodeCarbon stamps each record at the moment the block *stops*, so the
    interval between the first and last stamps omits the first block entirely.
    Adding it back matters: without it this function reported coverage above
    100%, which is how the omission was noticed.
    """
    rows = []
    for keys, g in df.groupby(["ecosystem", "model", "dataset", "repetition"]):
        g = g.assign(_ts=pd.to_datetime(g.timestamp, format="mixed", utc=True))
        g = g.sort_values("_ts")
        first_block = float(g.duration_hw_s.iloc[0])
        span = (g._ts.max() - g._ts.min()).total_seconds() + first_block
        tracked = float(g.duration_hw_s.sum())
        if span <= 0:
            continue
        rows.append({
            "ecosystem": keys[0], "model": keys[1], "dataset": keys[2],
            "repetition": keys[3],
            "span_s": round(span, 1), "tracked_s": round(tracked, 1),
            "coverage_pct": round(100 * tracked / span, 1),
        })
    per_run = pd.DataFrame(rows)
    return (per_run.groupby("ecosystem")
            .agg(n_runs=("coverage_pct", "size"),
                 coverage_pct=("coverage_pct", "mean"),
                 coverage_min_pct=("coverage_pct", "min"),
                 untracked_s_per_run=("span_s", "mean"))
            .assign(untracked_s_per_run=lambda d:
                    (d.untracked_s_per_run * (1 - d.coverage_pct / 100)).round(1))
            .round(1).reset_index())


def agreement_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean ratio and dispersion of cc/hw within each group."""
    out = (
        df.groupby(keys)
        .agg(
            n_epochs=("ratio_meas", "size"),
            gpu_ratio=("ratio_gpu", "mean"),
            gpu_ratio_sd=("ratio_gpu", "std"),
            cpu_ratio=("ratio_cpu", "mean"),
            cpu_ratio_sd=("ratio_cpu", "std"),
            measured_ratio=("ratio_meas", "mean"),
            total_ratio=("ratio_total", "mean"),
            ram_share_pct=("ram_share_pct", "mean"),
        )
        .reset_index()
    )
    return out.round(4)


def duration_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """A window mismatch would invalidate every energy comparison above."""
    d = df.copy()
    d["window_delta_pct"] = 100.0 * (d["duration_cc_s"] - d["duration_hw_s"]) / d["duration_hw_s"]
    return (
        d.groupby(["ecosystem", "phase"])["window_delta_pct"]
        .agg(n="size", mean="mean", sd="std", worst=lambda s: s.abs().max())
        .round(3)
        .reset_index()
    )


def agreement_by_phase_length(df: pd.DataFrame) -> pd.DataFrame:
    """The campaign-wide agreement figure hides where it comes from.

    CodeCarbon samples at a fixed interval, one second here. When a measurement
    block is long relative to that interval the two instruments are reading the
    same accumulating counters and agree to a hundredth of a percent. When the
    block is shorter than the interval there is nothing to accumulate, the
    estimator interpolates from sampled power, and the disagreement grows.

    Reporting only the mean over all blocks would therefore overstate how well an
    estimator does on exactly the blocks a cross-ecosystem study cares about --
    the inference passes, and the epochs of the fastest stacks.
    """
    d = df.copy()
    d["phase_length"] = pd.cut(
        d.duration_hw_s, [0, 1, 3, 10, 30, np.inf],
        labels=["<1 s", "1-3 s", "3-10 s", "10-30 s", ">30 s"])
    t = (d.groupby("phase_length", observed=True)
         .agg(n_epochs=("ratio_meas", "size"),
              mean_ratio=("ratio_meas", "mean"),
              p95_ratio=("ratio_meas", lambda s: s.quantile(0.95)),
              worst_ratio=("ratio_meas", "max"))
         .reset_index())
    t["mean_error_pct"] = ((t.mean_ratio - 1) * 100).round(2)
    t["worst_error_pct"] = ((t.worst_ratio - 1) * 100).round(1)
    return t.round(4)


def duration_floor_model(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """CodeCarbon will not report a window shorter than a fixed floor.

    The two instruments agree on *energy* to within half a percent, but the
    ``duration`` CodeCarbon writes next to that energy is not the duration the
    energy was accumulated over. Whenever the phase is longer than roughly four
    seconds the two windows coincide to 15 ms; below that, CodeCarbon reports
    its floor instead of the phase. A JAX inference epoch that takes 0.31 s is
    filed as 3.98 s.

    This matters because power is energy over time. Anything derived by dividing
    a CodeCarbon energy by a CodeCarbon duration -- the energy-versus-time
    analysis of the original submission among them -- inherits the floor, and
    inherits it unevenly: only the phases shorter than the floor are affected,
    which is to say the inference phases and the fastest ecosystems.
    """
    d = df.copy()
    d["excess_s"] = d.duration_cc_s - d.duration_hw_s
    long_phases = d[d.duration_hw_s > 6.0]
    floor = d[d.duration_hw_s < 1.0].duration_cc_s.median()
    pred = np.maximum(d.duration_hw_s, floor)
    ss_res = float(((pred - d.duration_cc_s) ** 2).sum())
    ss_tot = float(((d.duration_cc_s - d.duration_cc_s.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    rows = pd.DataFrame(
        {
            "quantity": [
                "floor on the reported window (s)",
                "median window excess, phases longer than 6 s (s)",
                "95th pct window excess, phases longer than 6 s (s)",
                "median |error| of max(phase, floor) (s)",
                "R2 of max(phase, floor)",
            ],
            "value": [
                floor,
                long_phases.excess_s.median(),
                long_phases.excess_s.quantile(0.95),
                float(np.median(np.abs(pred - d.duration_cc_s))),
                r2,
            ],
        }
    ).round(4)
    return rows, floor


def power_distortion(df: pd.DataFrame) -> pd.DataFrame:
    """What the window floor does to power, as a function of phase length."""
    d = df.copy()
    d["power_counters_w"] = d.hw_meas_j / d.duration_hw_s
    d["power_codecarbon_w"] = d.cc_total_j / d.duration_cc_s
    d["phase_length"] = pd.cut(
        d.duration_hw_s,
        [0, 1, 3, 10, 30, np.inf],
        labels=["<1 s", "1-3 s", "3-10 s", "10-30 s", ">30 s"],
    )
    t = (
        d.groupby("phase_length", observed=True)
        .agg(
            n_epochs=("power_counters_w", "size"),
            power_counters_w=("power_counters_w", "mean"),
            power_codecarbon_w=("power_codecarbon_w", "mean"),
        )
        .reset_index()
    )
    t["understated_by"] = (t.power_counters_w / t.power_codecarbon_w).round(2)
    return t.round(3)


def main() -> None:
    df = collect()
    print(f"epochs with both instruments: {len(df)} "
          f"over {df.run_key.nunique() if 'run_key' in df else df.groupby(['ecosystem','model','dataset','repetition']).ngroups} runs")
    print()

    # -- 1. does the CodeCarbon window match the counter window? ------------
    dur = duration_agreement(df)
    print("--- measurement window agreement (%) ---")
    print(dur.to_string(index=False))
    save_table(dur, "v2_instrument_windows",
               "CodeCarbon window vs counter window, per ecosystem and phase")

    cov = measurement_coverage(df)
    print()
    print("--- share of each run that lies inside a measured block ---")
    print(cov.to_string(index=False))
    save_table(cov, "v2_instrument_coverage",
               "Tracked time as a share of the span from first block to last")

    # -- 2. agreement per ecosystem x phase ---------------------------------
    per_eco = agreement_by(df, ["ecosystem", "phase"])
    print()
    print("--- CodeCarbon / hardware-counter ratio, by ecosystem ---")
    print(per_eco.to_string(index=False))
    save_table(per_eco, "v2_instrument_by_ecosystem",
               "CodeCarbon over hardware counters, by ecosystem and phase")

    # -- 3. agreement per block (the full design) ---------------------------
    per_block = agreement_by(df, ["ecosystem", "model", "dataset", "phase"])
    save_table(per_block, "v2_instrument_by_block",
               "CodeCarbon over hardware counters, every block of the design")

    # -- 4. where the two instruments structurally disagree -----------------
    summary = pd.DataFrame(
        {
            # CodeCarbon does not sample the GPU: on this platform it reads
            # nvmlDeviceGetTotalEnergyConsumption, the same register the harness
            # reads, so the first row compares one register with itself. The
            # label said "pynvml sampling" and that was wrong.
            "quantity": [
                "GPU, ratio (both read the NVML energy register)",
                "CPU package, ratio (both read RAPL energy_uj)",
                "GPU + CPU, ratio (the part both read)",
                "CodeCarbon total over counters, ratio (incl. modelled RAM)",
                "RAM share of the CodeCarbon total (per cent, not a ratio)",
            ],
            "mean_ratio_or_share": [
                df.ratio_gpu.mean(),
                df.ratio_cpu.mean(),
                df.ratio_meas.mean(),
                df.ratio_total.mean(),
                df.ram_share_pct.mean(),
            ],
            "sd": [
                df.ratio_gpu.std(),
                df.ratio_cpu.std(),
                df.ratio_meas.std(),
                df.ratio_total.std(),
                df.ram_share_pct.std(),
            ],
            "p05": [
                df.ratio_gpu.quantile(0.05),
                df.ratio_cpu.quantile(0.05),
                df.ratio_meas.quantile(0.05),
                df.ratio_total.quantile(0.05),
                df.ram_share_pct.quantile(0.05),
            ],
            "p95": [
                df.ratio_gpu.quantile(0.95),
                df.ratio_cpu.quantile(0.95),
                df.ratio_meas.quantile(0.95),
                df.ratio_total.quantile(0.95),
                df.ram_share_pct.quantile(0.95),
            ],
        }
    ).round(4)
    print()
    print("--- campaign-wide instrument summary ---")
    print(summary.to_string(index=False))
    # Mean of per-block ratios. A campaign-weighted ratio is a different and
    # much smaller number -- the per-block mean is dominated by short blocks
    # carrying a fixed offset -- so both are reported rather than one.
    summary["campaign_weighted"] = [
        df.cc_gpu_j.sum() / df.hw_gpu_j.sum(),
        df.cc_cpu_j.sum() / df.hw_cpu_j.sum(),
        df.cc_meas_j.sum() / df.hw_meas_j.sum(),
        df.cc_total_j.sum() / df.hw_total_j.sum(),
        100 * df.cc_ram_j.sum() / df.cc_total_j.sum(),
    ]
    save_table(summary, "v2_instrument_summary",
               "Campaign-wide agreement between the two instruments")

    # -- 5. does the choice of instrument change the ranking? ---------------
    # Training only, once: the manuscript reported "4 of 6 blocks" from this
    # table while arguing that inference is where the apparatus bites, which
    # drew the reassurance from the phase the paper calls safe. Both phases now.
    ranks = []
    for (model, dataset, phase), g in df.groupby(["model", "dataset", "phase"]):
        per_eco_energy = g.groupby("ecosystem")[["hw_meas_j", "cc_total_j"]].sum()
        ranks.append(
            {
                "model": model,
                "dataset": dataset,
                "phase": phase,
                "n_ecosystems": len(per_eco_energy),
                "rank_by_counters": " < ".join(per_eco_energy.hw_meas_j.sort_values().index),
                "rank_by_codecarbon": " < ".join(per_eco_energy.cc_total_j.sort_values().index),
                "identical": (
                    list(per_eco_energy.hw_meas_j.sort_values().index)
                    == list(per_eco_energy.cc_total_j.sort_values().index)
                ),
            }
        )
    rank_df = pd.DataFrame(ranks)
    print()
    print("--- does the instrument change the ranking? ---")
    print(rank_df[["model", "dataset", "phase", "n_ecosystems", "identical"]].to_string(index=False))
    save_table(rank_df, "v2_instrument_ranking",
               "Ecosystem ranking under each instrument, per block")

    abl = agreement_by_phase_length(df)
    print()
    print("--- energy agreement, by phase length ---")
    print(abl.to_string(index=False))
    save_table(abl, "v2_instrument_agreement_by_length",
               "Energy agreement between the instruments as a function of phase length")

    # -- 6. the window floor, and what it does to power ---------------------
    floor_rows, floor = duration_floor_model(df)
    print()
    print("--- CodeCarbon's reported window is max(phase, floor) ---")
    print(floor_rows.to_string(index=False))
    save_table(floor_rows, "v2_instrument_duration_floor",
               "CodeCarbon's reported duration is a floor, not the phase length")

    dist = power_distortion(df)
    print()
    print("--- implied power, by phase length ---")
    print(dist.to_string(index=False))
    save_table(dist, "v2_instrument_power_distortion",
               "Power implied by each instrument, by phase length")

    df.to_csv(REPO_ROOT / "results" / "revision" / "tables" / "v2_instrument_epochs.csv",
              index=False)


if __name__ == "__main__":
    main()
