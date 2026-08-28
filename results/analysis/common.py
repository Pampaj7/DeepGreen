#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared loading, unit handling and quality control for the DeepGreen AI re-analysis.

This module is the single point where raw CodeCarbon output is turned into the
quantities used by every table and figure in the paper.  It exists because the
original pipeline read CodeCarbon's ``energy_consumed`` column (kilowatt-hours)
and labelled it as Joules, an error of 3.6e6.

Unit contract
-------------
CodeCarbon writes, per tracked block:
    duration            seconds
    cpu_energy          kWh
    gpu_energy          kWh
    ram_energy          kWh
    energy_consumed     kWh   (== cpu + gpu + ram)
    emissions           kg CO2eq
    *_power             W  (mean instantaneous power over the tracked block)

Every column produced by this module carries its unit in the name:
``energy_j``, ``cpu_energy_j``, ``duration_s``, ``power_w``, ``emissions_kg``.
Nothing downstream may use ``energy_consumed`` directly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import pandas as pd
from scipy import stats as scipy_stats

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "results" / "data" / "combined_data.csv"
OUT_DIR = REPO_ROOT / "results" / "revision"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

# --------------------------------------------------------------------------
# Unit constants
# --------------------------------------------------------------------------
J_PER_KWH = 3.6e6  # 1 kWh = 3.6 MJ

# Hardware reference values for the measurement platform (used only for
# plausibility checks and utilisation reporting, never to derive energy).
GPU_TDP_W = 350.0  # NVIDIA L40S board power limit
CPU_TDP_W = 185.0  # Intel Xeon Gold 5418Y

# --------------------------------------------------------------------------
# Ecosystem naming
# --------------------------------------------------------------------------
# Reviewer 3, major comment 8: the study evaluates eight *language-framework
# ecosystems*, not eight languages.  ``language`` in the raw CSV mixes the two.
ECOSYSTEM = {
    "PyTorch": "Python/PyTorch",
    "TensorFlow": "Python/TensorFlow",
    "JAX": "Python/JAX",
    "C++": "C++/LibTorch",
    "java": "Java/DL4J",
    "Java": "Java/DL4J",
    "R": "R/torch",
    "Rust": "Rust/tch",
    "MATLAB": "MATLAB/DLT",
}

# Backend family, used for the shared-backend control analysis
# (Reviewer 1, comment 3).
BACKEND = {
    "Python/PyTorch": "LibTorch",
    "C++/LibTorch": "LibTorch",
    "R/torch": "LibTorch",
    "Rust/tch": "LibTorch",
    "Python/TensorFlow": "TensorFlow",
    "MATLAB/DLT": "MATLAB DLT",
    "Java/DL4J": "ND4J",
    "Python/JAX": "XLA",
}

ECOSYSTEM_ORDER = [
    "Rust/tch",
    "C++/LibTorch",
    "Python/PyTorch",
    "Python/JAX",
    "Python/TensorFlow",
    "MATLAB/DLT",
    "R/torch",
    "Java/DL4J",
]

PHASE_LABEL = {"train": "Training", "eval": "Inference"}

#: Ecosystems dropped from the replicated campaign, with the reason.
#: The first-campaign analysis keeps them -- removing an ecosystem from the
#: audit of the submitted results would misrepresent what was submitted.
EXCLUDED_FROM_V2 = {
    "MATLAB/DLT": "proprietary toolbox, no license available on the replication "
                  "machine; it also cannot be pinned to the shared LibTorch build "
                  "that the other stacks are aligned on",
}

#: The seven ecosystems the replicated campaign covers.
V2_ECOSYSTEMS = [e for e in ECOSYSTEM_ORDER if e not in EXCLUDED_FROM_V2]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(csv_path: Path | str = DATA_CSV, drop_outliers: bool = False) -> pd.DataFrame:
    """Load the raw campaign CSV and return it with explicit units.

    Parameters
    ----------
    drop_outliers
        If True, rows flagged by :func:`flag_power_outliers` are removed.  The
        default is False: outliers are *flagged* and reported, and robust
        summaries (median / trimmed mean) are used instead of silent deletion.
    """
    df = pd.read_csv(csv_path)

    df["fase"] = df["fase"].str.lower().replace({"test": "eval"})
    df["ecosystem"] = df["language"].map(ECOSYSTEM)
    if df["ecosystem"].isna().any():
        unknown = sorted(df.loc[df["ecosystem"].isna(), "language"].unique())
        raise ValueError(f"Unmapped language labels: {unknown}")
    df["backend"] = df["ecosystem"].map(BACKEND)

    # --- unit conversion, the single place it happens ---------------------
    df["duration_s"] = df["duration"].astype(float)
    for src, dst in [
        ("energy_consumed", "energy_j"),
        ("cpu_energy", "cpu_energy_j"),
        ("gpu_energy", "gpu_energy_j"),
        ("ram_energy", "ram_energy_j"),
    ]:
        df[dst] = df[src].astype(float) * J_PER_KWH
    df["emissions_kg"] = df["emissions"].astype(float)

    # Mean power over the tracked block, derived from energy and duration.
    # This is independent of CodeCarbon's sampled *_power columns and is the
    # quantity used for plausibility checks.
    df["power_w"] = df["energy_j"] / df["duration_s"]
    df["gpu_power_derived_w"] = df["gpu_energy_j"] / df["duration_s"]
    df["gpu_utilisation_proxy"] = df["gpu_power_derived_w"] / GPU_TDP_W

    df["phase"] = df["fase"].map(PHASE_LABEL)
    df = df.rename(columns={"modello": "model_arch"})

    df = flag_power_outliers(df)
    df = add_harmonised(df)
    if drop_outliers:
        df = df[~df["is_power_outlier"]].copy()

    return df


def flag_power_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose *sampled* GPU power is physically implausible.

    CodeCarbon's ``gpu_power`` column is the mean of instantaneous samples over
    the tracked block.  On short blocks (inference is 1.6-13 s against a 15 s
    default sampling period) the sample count is small and the mean is
    unreliable: the campaign logs contain both 0 W readings and readings above
    the board power limit.  The *energy* columns come from the NVML energy
    counter and are unaffected, which is why these rows are flagged rather than
    dropped.
    """
    gpu = df["gpu_power"].astype(float)
    df = df.copy()
    df["is_power_outlier"] = (gpu <= 0.0) | (gpu > GPU_TDP_W)
    return df


# --------------------------------------------------------------------------
# Harmonised energy boundary
# --------------------------------------------------------------------------
# The campaign was not measured with a single instrument configuration:
#
#   CodeCarbon 2.8.4, machine mode (JAX, PyTorch, TensorFlow, Rust)
#       cpu_power = 42.5 W   constant  (CodeCarbon's hardcoded fallback,
#                                       POWER_CONSTANT 85 W x 0.5, used when
#                                       RAPL is unavailable -- not measured)
#       ram_power = 188.84 W constant  (= 0.375 W/GB x 503.6 GB installed)
#       -> 231.3 W of *modelled* host power, i.e. ~66% of the reported energy
#          is a deterministic linear function of wall-clock duration.
#
#   CodeCarbon 3.0.4, machine mode (C++, Java, MATLAB)
#       cpu_power ~ 37 W     measured via RAPL
#       ram_power = 70 W     constant cap
#       -> ~107 W of host power, ~43-53% of the reported energy.
#
#   CodeCarbon 2.8.4, PROCESS mode (R)
#       ram_power ~ 0.04 W   -> RAM essentially untracked, host share ~31%.
#
# Comparing raw ``energy_consumed`` across ecosystems therefore compares
# instrument configurations as much as it compares ecosystems.  Two derived
# quantities are defined instead:
#
#   energy_gpu_j    NVML energy counter only.  The same instrument everywhere,
#                   and the only quantity attributable to the DL computation.
#
#   energy_harm_j   GPU energy plus a *single* host power model applied
#                   uniformly to every ecosystem.  The constant below is the
#                   CodeCarbon 3.0.4 host model (37 W CPU + 70 W RAM), chosen
#                   because it is the more recent and the only one whose CPU
#                   term is measured rather than assumed.
HOST_POWER_W = 107.0


def add_harmonised(df: pd.DataFrame, host_power_w: float = HOST_POWER_W) -> pd.DataFrame:
    """Add ``energy_gpu_j`` and ``energy_harm_j`` to ``df``."""
    df = df.copy()
    df["energy_gpu_j"] = df["gpu_energy_j"]
    df["energy_harm_j"] = df["gpu_energy_j"] + host_power_w * df["duration_s"]
    return df


#: The three energy definitions reported side by side throughout the revision.
ENERGY_DEFS = {
    "energy_j": "as measured (CodeCarbon total, instrument-confounded)",
    "energy_gpu_j": "GPU only (NVML counter, identical instrument)",
    "energy_harm_j": f"harmonised (GPU + {HOST_POWER_W:.0f} W uniform host model)",
}


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260818,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``values``.

    Returns ``(point, lo, hi)``.  Note that when ``values`` are the 30 epochs of
    a single training run they are *pseudo-replicates*: the interval describes
    within-run epoch-to-epoch dispersion, not between-run uncertainty.  Every
    caller must label it as such.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    if values.size == 1:
        v = float(values[0])
        return (v, v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = statistic(values[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(statistic(values)), float(lo), float(hi))


def t_ci(
    values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Student-t confidence interval for the mean. Returns ``(point, lo, hi)``.

    Used instead of the percentile bootstrap wherever the sample is the five
    independent runs of a configuration. At n = 5 the percentile bootstrap is
    badly anti-conservative -- its intervals here came out a median 37% narrower
    than the t-interval, because resampling five points cannot see beyond the
    five points. The t-interval assumes approximate normality of the mean, which
    is the weaker assumption of the two at this sample size, and it does not
    pretend to a coverage it has not got.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(values.mean())
    if values.size == 1:
        return (mean, mean, mean)
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    half = float(scipy_stats.t.ppf(1 - alpha / 2, values.size - 1)) * sem
    return (mean, mean - half, mean + half)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Cliff's delta effect size and its conventional magnitude label."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return (np.nan, "undefined")
    # rank-based O(n log n) formulation
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)
    # average ranks for ties
    s = pd.Series(combined)
    ranks = s.rank(method="average").to_numpy()
    ra = ranks[: a.size].sum()
    u = ra - a.size * (a.size + 1) / 2.0
    delta = 2.0 * u / (a.size * b.size) - 1.0
    mag = np.abs(delta)
    if mag < 0.147:
        label = "negligible"
    elif mag < 0.33:
        label = "small"
    elif mag < 0.474:
        label = "medium"
    else:
        label = "large"
    return (float(delta), label)


def fmt_sci(x: float, digits: int = 2) -> str:
    """Format a number in the ``m.mm x 10^e`` style used in the tables."""
    if not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp
    if -1 <= exp <= 3:
        return f"{x:,.{digits}f}"
    return f"{mant:.{digits}f}e{exp:+d}"


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str, caption: str = "") -> None:
    """Write a table as CSV and Markdown next to each other."""
    ensure_dirs()
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    md = TABLE_DIR / f"{name}.md"
    with md.open("w") as fh:
        if caption:
            fh.write(f"**{caption}**\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")
    print(f"  wrote {md.relative_to(REPO_ROOT)}")


def order_ecosystems(index) -> list[str]:
    """Return ``index`` sorted by the canonical ecosystem order."""
    present = [e for e in ECOSYSTEM_ORDER if e in set(index)]
    extra = [e for e in index if e not in set(ECOSYSTEM_ORDER)]
    return present + sorted(extra)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    d = load()
    print(d[["ecosystem", "phase", "energy_j", "duration_s", "power_w"]].head())
    print(f"\n{len(d)} measurement rows, {d['ecosystem'].nunique()} ecosystems")
