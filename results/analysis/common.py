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

import functools
import os
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

# --------------------------------------------------------------------------
# The completeness gate
# --------------------------------------------------------------------------
#: A run counts only if it recorded every epoch of both phases.
#:
#: A run that dies part-way leaves its directory behind holding the blocks it
#: managed before the exception. Those are fragments, not small measurements of
#: the same thing -- 16 J beside a neighbouring stack's 300 kJ -- and averaging
#: them in produced ecosystem spreads of 20,000x in the first campaign. The
#: second campaign made the point again: four JAX VGG-16 runs died on their
#: first training batch, each leaving one row, and a per-stack mean over the
#: directory reported JAX at 135 W where the complete runs say 218 W.
#:
#: This lived in three copies -- 09_campaign_v2, 11_instrument_comparison and
#: 17_window_calibration -- and the third hard-coded 30 epochs where the other
#: two read DEEPGREEN_EPOCHS, so a shorter campaign would have passed every
#: fragment through one of the three. One definition, here.
EXPECTED_EPOCHS = int(os.environ.get("DEEPGREEN_EPOCHS", "30"))
PHASES = ("train", "eval")


#: Runs a complete campaign produces: 7 ecosystems x 2 models x 3 datasets x 5.
EXPECTED_RUNS = int(os.environ.get("DEEPGREEN_EXPECTED_RUNS", "210"))
CAMPAIGN_DIR = REPO_ROOT / "results" / "campaign_v2"


@functools.lru_cache(maxsize=1)
def campaign_status() -> tuple[int, int]:
    """``(complete runs on disk, runs a full campaign produces)``."""
    if not CAMPAIGN_DIR.is_dir():
        return (0, EXPECTED_RUNS)
    n = sum(1 for d in sorted(CAMPAIGN_DIR.glob("*"))
            if d.is_dir() and read_complete_counters(d)[0] is not None)
    return (n, EXPECTED_RUNS)


def campaign_is_partial() -> bool:
    done, want = campaign_status()
    return done < want


class _TableDir:
    """``TABLES / "x.csv"`` -- writes divert, reads fall back.

    Every script addresses tables as ``TABLES / name``. While the campaign is
    partial, ``save_table`` writes into the ``_partial`` sibling, so a name that
    this run has produced resolves there; a name it has not -- anything derived
    from the first campaign, which does not change -- resolves to the committed
    directory as before. One partial pipeline therefore reads its own fresh
    output and the stable inputs, and writes over neither.
    """

    def __truediv__(self, name) -> Path:
        candidate = tables_dir() / name
        return candidate if candidate.exists() else TABLE_DIR / name

    def __fspath__(self) -> str:
        return str(tables_dir())

    def __repr__(self) -> str:
        return f"<TABLES {tables_dir()}>"


def tables_dir() -> Path:
    """Where this run may write tables: the real directory, or a partial sibling.

    The analysis scripts write into the same directory the manuscript reads, so
    running one against a campaign still in flight replaces the paper's inputs
    with numbers from however many runs happen to have finished. That is not
    hypothetical: it happened while monitoring the second campaign, and
    `\vRuns` went from 210 to 59 and `\vRepetitions` from 5 to 2 in a single
    commit, silently, because every script involved did exactly what it was
    asked.

    Monitoring a live campaign is legitimate and worth keeping easy. So the
    answer is not to forbid the run but to send it somewhere else: while the
    campaign is short of `EXPECTED_RUNS`, output goes to a `_partial` sibling
    and the committed directory is untouched.
    """
    real = TABLE_DIR
    return real if not campaign_is_partial() else real.with_name(real.name + "_partial")


def figures_dir() -> Path:
    """Where this run may write figures: the real directory, or a partial sibling.

    The same diversion as :func:`tables_dir`, for the same reason and with a
    sharper edge -- a figure carries no run count on its face, so a substituted
    one looks exactly like the right one. 12 and 13 divert; 08 wrote straight to
    ``FIG_DIR``, so a monitoring run of the first-campaign audit replaced
    committed figures with whatever was on disk at the time.
    """
    real = FIG_DIR
    return real if not campaign_is_partial() else real.with_name(real.name + "_partial")


def freeman_halton_exact(rows: "np.ndarray") -> tuple[float, float]:
    """Exact test of homogeneity for an r x 2 table. Returns ``(p, p_observed)``.

    The r x c generalisation of Fisher's exact test. Conditional on both sets of
    margins, every table has a multivariate hypergeometric probability

        P = prod_i C(n_i, k_i) / C(N, K)

    and the p-value is the total probability of the tables no more likely than
    the one observed. Nothing is approximated and nothing is simulated, so the
    answer does not depend on a permutation count or a seed.

    This exists because the collapse table has seven ecosystems, ten runs each
    and twelve collapses in total: chi-square wants five expected per cell and
    gets 1.71, and the permutation test the script used instead measures only
    ``max(rate) - min(rate)``, which throws away the shape of the table and
    returned p = 0.0676 where chi-square returned 0.0065. Neither number should
    be quoted for a table this small when the exact one is a few thousand terms
    away.

    ``rows`` is (r, 2): successes and failures per group.
    """
    from math import comb, prod
    rows = np.asarray(rows, dtype=int)
    n = rows.sum(axis=1)              # group sizes
    K = int(rows[:, 0].sum())         # total successes
    N = int(n.sum())
    denom = comb(N, K)

    def prob(ks) -> float:
        return prod(comb(int(ni), int(ki)) for ni, ki in zip(n, ks)) / denom

    observed = prob(rows[:, 0])
    total = 0.0
    # depth-first over allocations of K successes to the groups, pruned by what
    # the remaining groups can still hold
    capacity = np.cumsum(n[::-1])[::-1]

    def walk(i: int, left: int, acc: float) -> None:
        nonlocal total
        if i == len(n) - 1:
            if 0 <= left <= n[i]:
                pr = acc * comb(int(n[i]), int(left)) / denom
                if pr <= observed * (1 + 1e-12):
                    total += pr
            return
        lo = max(0, left - int(capacity[i + 1]))
        hi = min(int(n[i]), left)
        for k in range(lo, hi + 1):
            walk(i + 1, left - k, acc * comb(int(n[i]), k))

    walk(0, K, 1.0)
    return float(min(total, 1.0)), float(observed)


def write_table_path(name: str) -> Path:
    """Where a table of this name may be written, creating the directory.

    ``TABLES / name`` is for reading and falls back to the committed directory,
    so writing through it would write there. Three tables bypassed
    ``save_table`` and did exactly that.
    """
    out = tables_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out / name


#: The object every v2 script assigns to its ``TABLES``.
TABLES_RESOLVER = _TableDir()


def announce_scope(script: str) -> None:
    """Print, once, what this run is reading and where it may write."""
    done, want = campaign_status()
    if done < want:
        print(f"[{script}] campaign is PARTIAL: {done} of {want} runs complete.")
        print(f"[{script}] writing to {tables_dir().relative_to(REPO_ROOT)}/ "
              f"-- the manuscript's inputs are not touched.")
    else:
        print(f"[{script}] campaign complete: {done} runs.")


def read_complete_counters(run_dir: Path) -> tuple[pd.DataFrame | None, dict]:
    """``(counters, per-phase epoch counts)`` for a run, or ``(None, counts)``.

    ``None`` means the run is missing, unreadable, empty, or short of
    ``EXPECTED_EPOCHS`` in either phase. The counts come back either way so a
    caller can report exactly what was rejected rather than dropping it
    silently -- an analysis that quietly discards runs is as hard to trust as
    one that quietly keeps broken ones.
    """
    path = Path(run_dir) / "counters.csv"
    if not path.exists():
        return None, {}
    try:
        hw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None, {}
    if hw.empty:
        return None, {}
    counts = hw.groupby("phase").epoch.nunique()
    counts = {ph: int(counts.get(ph, 0)) for ph in PHASES}
    if any(counts[ph] < EXPECTED_EPOCHS for ph in PHASES):
        return None, counts
    return hw, counts

#: The columns every stack's metrics.csv carries. Three header shapes coexist in
#: the campaign -- 30 Python/PyTorch runs have no ``train_acc``, 58 runs add a
#: ``precision`` column -- so a union of columns is expected and only these are
#: guaranteed.
METRICS_COLUMNS = ["run", "ecosystem", "model", "dataset", "repetition", "seed",
                   "epoch", "train_loss", "test_loss", "test_acc"]


def read_campaign_metrics(complete_only: bool = True,
                          root: Path | None = None) -> pd.DataFrame:
    """Every run's per-epoch quality rows, concatenated, with a ``run`` column.

    One reader, because there were three, over three different populations of
    the same campaign: 09's collector joins metrics onto the energy blocks
    behind the completeness gate, 15_convergence walks the run directories with
    no gate at all, and scripts/check_consistency read
    ``results/replication/metrics.csv.gz`` -- a package the pipeline writes and
    nothing rebuilds. That last one is how the manuscript came to disclose a
    conformance failure the campaign it describes does not have: Java/DL4J's
    ``test_loss`` is NaN in every row of the first campaign and in none of the
    second, and the checker was reading the first.

    ``root`` defaults to the current campaign and exists so that the superseded
    one can be read through the same gate -- the collapse finding belongs to it
    and the manuscript now presents it as its own -- rather than through a
    second walk of the directories.

    Returns an empty frame with :data:`METRICS_COLUMNS` when there is nothing to
    read, so a caller can test ``.empty`` rather than guess at the shape.
    """
    frames = []
    for run_dir in sorted(p for p in (root or CAMPAIGN_DIR).glob("*") if p.is_dir()):
        if complete_only and read_complete_counters(run_dir)[0] is None:
            continue
        path = run_dir / "metrics.csv"
        # The file exists from the moment the harness opens it and has no header
        # until the first epoch closes, so a live campaign produces both of
        # these.
        if not path.exists() or not path.stat().st_size:
            continue
        try:
            m = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if m.empty:
            continue
        m["run"] = run_dir.name
        frames.append(m)
    if not frames:
        return pd.DataFrame(columns=METRICS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


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
    tables_dir().mkdir(parents=True, exist_ok=True)
    figures_dir().mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str, caption: str = "") -> None:
    """Write a table as CSV and Markdown next to each other.

    Into ``tables_dir()``, which diverts to a ``_partial`` sibling while the
    campaign is incomplete. See that function for why.
    """
    ensure_dirs()
    out = tables_dir()
    df.to_csv(out / f"{name}.csv", index=False)
    md = out / f"{name}.md"
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
