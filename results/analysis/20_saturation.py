#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The accelerator-saturation cell (spec S7), against the campaign it contrasts.

The manuscript's 210-run campaign trains at 32x32 with batch 128, and at that
shape the GPU is idle for most of the wall clock: 4.7% mean utilisation
(R/torch, ResNet-18) to 79.9% (Java/DL4J, VGG-16), most stacks between 20% and
50%.  The objection that follows is foundational rather than technical -- a
study in which the accelerator is lightly loaded measures host-side overhead
and calls the result the energy cost of deep learning.

S7 answers it with one declared contrast cell: the same two networks, the same
seven ecosystems, the same harness and the same instruments, on ten ImageNet
classes at 224x224 with batch 32, where the accelerator is the bottleneck.  If
the ecosystem spread compresses when the GPU saturates, the study says so and
delimits its conclusion to the regime it measured; if it does not, the
objection does not survive its own test.  This script computes the contrast.

Two campaigns, and the frozen one stays frozen
----------------------------------------------
The saturation runs live in a campaign directory of their own -- 70 runs in
``results/campaign_saturation``, overridable with ``DEEPGREEN_SATURATION_DIR``.
Pointing an analysis script at a second campaign is exactly the manoeuvre that
once replaced the manuscript's inputs with a partial campaign's numbers, so
this file uses the mechanism ``common`` already provides for it rather than
inventing one:

  * ``DEEPGREEN_CAMPAIGN_DIR`` is set to the saturation directory *before*
    ``common`` is imported, so ``common.CAMPAIGN_DIR`` is the saturation cell
    for the life of this process.  ``common.tables_dir()`` then resolves,
    unconditionally, to ``results/revision/tables_<campaign dir name>`` --
    ``results/revision/tables_campaign_saturation`` for the real cell -- and
    every ``save_table`` call in this file writes there.  The 210-run
    campaign's ``results/revision/tables/`` is never opened for writing.

  * Reads of the main campaign's tables go through :data:`V2_TABLES`, which is
    ``common.TABLE_DIR`` named explicitly, never through ``TABLES_RESOLVER``.
    That is deliberate and is the same discipline 18_precision_ablation applies
    to the superseded campaign: a cross-campaign read is named at the point of
    use, so no number can arrive from the other campaign by accident.

  * The saturation directory may not *be* ``results/campaign_v2``.  It is
    refused if it is, because that pairing is the one case where the diversion
    above would resolve back onto the manuscript's own directory.

  * The two campaigns are never pooled.  They differ in resolution, batch size
    and class count; every number below is either from one side or an explicit
    ratio between the two.

Completeness
------------
A run counts only if it recorded every epoch of both phases
(``common.read_complete_counters``, ``DEEPGREEN_EPOCHS``).  The cell is
7 ecosystems x 2 models x 1 dataset x 5 repetitions = 70 such runs.  Short of
that the script refuses with a non-zero exit, because a spread computed over
whichever stacks happened to finish first is a spread over the fastest stacks.
``--allow-partial`` lifts the refusal and puts ``_partial`` on the name of
every file written, so a monitoring run's output can never be mistaken for the
cell's.

Definitions, stated once
------------------------
**Energy.**  The reported quantity is the same one 09_campaign_v2 reports for
the main campaign: the hardware counters over the counter-bracketed window,
NVML board energy for the accelerator (``counters.csv`` ``gpu_j``) plus the
RAPL package domains for the host CPU (``cpu_package_total_j``), summing to
``hw_total_j``.  CodeCarbon's own total is not used anywhere in this file: it
carries a modelled RAM term.  ``gpu``, ``cpu`` and ``total`` columns are all
carried, and ``total`` is what the headline spreads are computed on, with the
GPU-only spread beside them because the objection this cell answers is about
the accelerator specifically.

**Per-epoch energy of a cell.**  The independent run is the unit of analysis
(R1 c5 / R3 M2), so: for one run and one phase, the phase's counter energy
summed over its epochs and divided by the epoch count; then the median over the
five runs of the cell, with the interquartile range over those same five runs.
The IQR therefore describes between-run dispersion, not epoch-to-epoch noise.

**The 32x32 reference.**  ``results/revision/tables/v2_instrument_epochs.csv``,
the per-epoch counter record 11_instrument_comparison writes for the 210-run
campaign, put through the identical aggregation above (``hw_meas_j`` is
``hw_gpu_j + hw_cpu_j``, checked here rather than assumed).  Same instrument,
same window, same statistic on both sides of every ratio.

**rho(time, energy).**  Identical to the definition behind ``\\vRhoTrain`` and
``\\vRhoInfer`` in 12_paper_numbers, which reads them from
14_v2_statistics.energy_time: run totals -- the counter energy and the counter
duration each summed over the run's epochs -- then the median over repetitions
within each (ecosystem, model, dataset) cell, then ``scipy.stats.spearmanr``
across those cells, one correlation per phase.  At 224 there is one dataset, so
the correlation is over 14 (ecosystem, model) cells rather than the main
campaign's 42.  Pearson on the same medians is carried beside it.

**The spread.**  ``sat_spread`` (and the spread rows of the table) use the
definition behind the paper's ``tab_spread``, not the medians above: per run,
the mean over epochs of the per-epoch counter energy; per cell, the *mean* over
runs (09_campaign_v2.between_run_stats, called on both sides); the spread is
max/min of that over the ecosystems, and the cheapest/dearest stack is the
lowest/highest such mean.  The 32x32 side is checked against
``v2_between_run_statistics.csv``, the file ``tab_spread`` is built from.

**Utilisation.**  The 1 Hz ``nvidia-smi`` record in
``results/gpu_utilisation.csv``, over each run's *training blocks* -- the
counter-bracketed interval of every training epoch, recovered from the
``emissions_train_epoch*.csv`` mtimes and durations and ``counters.csv``
(see :func:`training_windows`) -- not over the whole run as
19_gpu_utilisation takes it.  The whole-run window also contains start-up,
evaluation and tracker overhead, a share of the wall clock that differs
between 32x32 and 224.  Both campaigns go through the same function; the 32x32
side is recomputed from ``results/campaign_v2`` rather than read from 19's
tables.  A run counts only if all its training blocks fall inside the record.
Sample counts and a coverage flag are reported per cell, because a per-stack
utilisation computed from an unstated subset is the same defect as an energy
figure computed from one.

Writes ``results/revision/tables_<campaign>/sat_*.{csv,md}`` and, for the
manuscript, ``paper/generated/numbers_saturation.tex`` (macros named
``\\vSat...``) and ``paper/generated/tab_saturation.tex``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

# --------------------------------------------------------------------------
# The campaign this process reads -- chosen before ``common`` is imported
# --------------------------------------------------------------------------
#: 7 ecosystems x 2 models x 1 dataset x 5 repetitions.
EXPECTED_SATURATION_RUNS = 7 * 2 * 1 * 5

#: The dataset key the cell is defined on. Anything else in the directory is a
#: sign that DEEPGREEN_SATURATION_DIR is pointing at the wrong campaign.
SATURATION_DATASET = "imagenette"

_default = _REPO / "results" / "campaign_saturation"
# A relative DEEPGREEN_SATURATION_DIR is resolved against the repository root,
# never the working directory, and an empty value means the default -- the
# same two rules the gate in run_all.sh applies (it runs from results/analysis,
# so a cwd-relative test there and a root-relative one here would disagree).
SATURATION_DIR = Path(os.environ.get("DEEPGREEN_SATURATION_DIR") or _default)
if not SATURATION_DIR.is_absolute():
    SATURATION_DIR = _REPO / SATURATION_DIR

# The one pairing the diversion in common.tables_dir() cannot protect against,
# because a script reading results/campaign_v2 is by definition reading the
# default campaign and writing to the directory the manuscript compiles from.
if SATURATION_DIR.resolve() == (_REPO / "results" / "campaign_v2").resolve():
    raise SystemExit(
        "20_saturation: DEEPGREEN_SATURATION_DIR points at results/campaign_v2, "
        "the frozen 210-run campaign. The saturation cell is a separate "
        "campaign directory (spec S7); this script will not write saturation "
        "tables into the manuscript's own table directory.")

# From a clone the raw tree is absent; its committed replication package is
# not. The package is restored into a temporary directory of the same name
# (common.tables_dir() keys on the name, so the outputs land where they would
# from the raw tree) and analysed exactly as the raw tree would be. Restored
# files do not carry the original mtimes, so the utilisation windows come from
# the windows frozen in the same package (common.utilisation_windows).
SATURATION_PACKAGE = (_REPO / "results" / "replication_saturation"
                      if SATURATION_DIR.name == "campaign_saturation" else None)
_RESTORED = None
if (not SATURATION_DIR.is_dir() and SATURATION_PACKAGE is not None
        and (SATURATION_PACKAGE / "counters.csv.gz").exists()):
    _spec = importlib.util.spec_from_file_location(
        "restore_from_replication",
        _REPO / "scripts" / "restore_from_replication.py")
    _rfr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rfr)
    _RESTORED = tempfile.TemporaryDirectory(prefix="deepgreen_saturation_")
    _target = Path(_RESTORED.name) / SATURATION_DIR.name
    _tables = _rfr.load_tables(SATURATION_PACKAGE)
    with contextlib.redirect_stdout(io.StringIO()):
        _ok = sum(_rfr.restore(r, _tables, force=True, dry_run=False,
                               out_root=_target)
                  for r in sorted(_tables["counters"].run.unique()))
    print(f"[20_saturation] {SATURATION_DIR} absent: restored {_ok} run(s) from "
          f"{SATURATION_PACKAGE.relative_to(_REPO)}/ into a temporary tree")
    SATURATION_DIR = _target

os.environ["DEEPGREEN_CAMPAIGN_DIR"] = str(SATURATION_DIR)
os.environ["DEEPGREEN_EXPECTED_RUNS"] = str(EXPECTED_SATURATION_RUNS)

sys.path.insert(0, str(_HERE))
from common import (DEFAULT_CAMPAIGN_DIR, EXPECTED_EPOCHS,  # noqa: E402
                    GPU_RECORD, MacroSet, REPO_ROOT, TABLE_DIR,
                    ECOSYSTEM_ORDER, announce_scope, campaign_energy,
                    campaign_status, load_gpu_record, num, order_ecosystems,
                    read_complete_counters, save_table, tables_dir,
                    utilisation_windows)

#: The 210-run campaign's committed tables. Named explicitly, and only ever
#: read. Everything this file writes goes through ``save_table``, which writes
#: into ``tables_dir()`` -- a sibling named for the saturation campaign.
V2_TABLES = TABLE_DIR

#: The 210-run campaign's run directories, read (never written) for the one
#: quantity its tables do not carry on the window this file uses: utilisation
#: over the training blocks. Named explicitly for the same reason as V2_TABLES.
#: ``DEEPGREEN_V2_CAMPAIGN_DIR`` overrides it -- a missing path named
#: ``campaign_v2`` makes this read the windows frozen in results/replication/,
#: which is how the clone path is tested.
V2_CAMPAIGN = Path(os.environ.get("DEEPGREEN_V2_CAMPAIGN_DIR")
                   or DEFAULT_CAMPAIGN_DIR)

#: Where the manuscript's generated inputs live. This file adds two names of its
#: own there and overwrites nothing 12_paper_numbers writes.
PAPER_OUT = REPO_ROOT / "paper" / "generated"

#: 11_instrument_comparison derives the ecosystem from the run-directory slug
#: (``Cpp-LibTorch`` -> ``Cpp/LibTorch``); 09_campaign_v2 reads it from
#: metrics.csv (``C++/LibTorch``). One spelling, or the join silently drops a
#: stack.
ECOSYSTEM_CANONICAL = {"Cpp/LibTorch": "C++/LibTorch"}

#: The stacks that share the LibTorch backend, as 14_v2_statistics defines them:
#: the three that load the byte-identical exported TorchScript module on the
#: pinned build, plus R, which links its own bundled LibTorch and builds an
#: equivalent architecture because the R binding cannot switch a script module
#: between train and eval mode.
LIBTORCH_FAMILY = ["Python/PyTorch", "C++/LibTorch", "Rust/tch", "R/torch"]

#: The six ecosystems other than Java/DL4J, for the spread with the dearest
#: stack set aside.
NO_JAVA = [e for e in ECOSYSTEM_ORDER if e not in ("Java/DL4J", "MATLAB/DLT")]


def shared_module_trio() -> list[str]:
    """The control group of tab_control: 14_v2_statistics.SHARED_MODULE.

    The stacks that load the byte-identical exported TorchScript module on the
    pinned LibTorch build (\\vCtrlStacks of them). Read from 14 rather than
    restated, in this file's spelling of the ecosystems.
    """
    shared = {ECOSYSTEM_CANONICAL.get(e, e)
              for e in sibling("14_v2_statistics").SHARED_MODULE}
    return [e for e in ECOSYSTEM_ORDER if e in shared]

#: Short names for the manuscript, as 12_paper_numbers spells them.
SHORT = {"Python/PyTorch": "PyTorch", "Python/TensorFlow": "TensorFlow",
         "Python/JAX": "JAX", "C++/LibTorch": "C++", "Java/DL4J": "Java",
         "R/torch": "R", "Rust/tch": "Rust"}

MODEL_LABEL = {"resnet18": "ResNet-18", "vgg16": "VGG-16"}
DATASET_LABEL = {"fashionmnist": "Fashion-MNIST", "cifar100": "CIFAR-100",
                 "tinyimagenet": "Tiny ImageNet", "imagenette": "Imagenette"}

#: The utilisation the paper calls high, for \vSatRefUtil<Model>HighCount.
SAT_UTIL_HIGH_PCT = 85.0

#: Macro name fragments. Letters only: TeX has no other kind of macro name.
PHASE_TAG = {"Training": "Train", "Inference": "Infer"}
MODEL_TAG = {"resnet18": "Resnet", "vgg16": "Vgg"}
DATASET_TAG = {"fashionmnist": "Fashion", "cifar100": "Cifar",
               "tinyimagenet": "Tiny"}

macros = MacroSet()
macro = macros.add


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def sibling(stem: str):
    """Import one of the numbered analysis scripts as a module.

    ``09_campaign_v2`` is not an importable identifier. 18_precision_ablation
    reaches for 09's collector this way for the same reason: a second copy of
    the parsing is a second copy that can drift away from the completeness
    gate.
    """
    if stem in _SIBLINGS:
        return _SIBLINGS[stem]
    path = _HERE / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SIBLINGS[stem] = mod
    return mod


_SIBLINGS: dict = {}


def canonical(series: pd.Series) -> pd.Series:
    return series.map(lambda x: ECOSYSTEM_CANONICAL.get(x, x))


def saturation_blocks() -> pd.DataFrame:
    """Every measurement block of the saturation cell, through 09's collector.

    ``energy_j`` is the counter total (NVML GPU + RAPL CPU package),
    ``gpu_energy_j`` the accelerator alone; the CPU package term is their
    difference, which is how ``counters.csv`` composes ``hw_total_j``.
    """
    df = sibling("09_campaign_v2").collect(SATURATION_DIR)
    if df.empty:
        return df
    df = df.copy()
    # metrics.csv's train_acc is not one quantity across the stacks: absent
    # for C++, Java, R and Rust, a percentage for some, a fraction for
    # TensorFlow's VGG-16. Nothing here may use it, so it is dropped at the
    # point of entry; test_acc (a percentage in every stack) is the only
    # quality figure this file reports.
    df = df.drop(columns=["train_acc"], errors="ignore")
    df["ecosystem"] = canonical(df["ecosystem"])
    df["cpu_energy_j"] = df["energy_j"] - df["gpu_energy_j"]
    seen = sorted(str(d) for d in df["dataset"].unique())
    if seen != [SATURATION_DATASET]:
        raise SystemExit(
            f"20_saturation: {SATURATION_DIR} holds datasets {seen}; the "
            f"saturation cell is {SATURATION_DATASET!r} alone (spec S7). "
            "DEEPGREEN_SATURATION_DIR is pointing at the wrong campaign.")
    # A configuration's five runs are told apart by the repetition index their
    # own metrics.csv carries, not by the directory name. Two directories
    # claiming the same index are two runs of one repetition -- a re-run left
    # beside the run it replaced -- and they would be summed into one
    # implausibly expensive "run" without a word. Refuse and name them.
    clash = (df.groupby(["ecosystem", "model", "dataset", "repetition"],
                        observed=True).run_dir.nunique())
    clash = clash[clash > 1]
    if len(clash):
        offenders = sorted(df[df.set_index(
            ["ecosystem", "model", "dataset", "repetition"]).index.isin(
                clash.index)].run_dir.unique())
        raise SystemExit(
            "20_saturation: these run directories report the same repetition "
            "index as another directory of the same configuration, so they are "
            "two runs of one repetition: " + ", ".join(offenders) +
            ". Remove the superseded directory; the analysis will not guess "
            "which one the cell is.")
    return df


def per_run(blocks: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per run and phase: run totals and the per-epoch quantities.

    ``*_per_epoch`` is the run's phase total divided by its epoch count, which
    is the quantity the cell medians below are taken over. The run totals are
    kept beside them because rho(time, energy) is defined on the totals.
    """
    g = (blocks.groupby(keys + ["phase"], observed=True)
         .agg(n_epochs=("epoch", "nunique"),
              total_j=("energy_j", "sum"),
              gpu_j=("gpu_energy_j", "sum"),
              cpu_j=("cpu_energy_j", "sum"),
              duration_s=("duration_s", "sum"))
         .reset_index())
    for column in ("total_j", "gpu_j", "cpu_j", "duration_s"):
        g[f"{column}_per_epoch"] = g[column] / g["n_epochs"]
    # Mean power over the phase's blocks: energy over time, not a sample mean.
    g["gpu_power_w"] = g["gpu_j"] / g["duration_s"]
    g["total_power_w"] = g["total_j"] / g["duration_s"]
    return g


def v2_blocks() -> pd.DataFrame:
    """The 210-run campaign's measurement blocks, in 09's column names.

    From ``v2_instrument_epochs.csv``, the per-epoch counter record. The
    columns are the same instrument under different names -- ``hw_meas_j`` is
    the counter total, ``duration_hw_s`` the counter-bracketed duration -- and
    the identity ``hw_meas_j == hw_gpu_j + hw_cpu_j`` is checked rather than
    assumed, because the whole contrast rests on both sides of every ratio
    being the same quantity.
    """
    path = V2_TABLES / "v2_instrument_epochs.csv"
    if not path.exists():
        raise SystemExit(
            f"20_saturation: {path.relative_to(REPO_ROOT)} is missing, so the "
            "32x32 side of the contrast cannot be computed. Run "
            "11_instrument_comparison.py against results/campaign_v2 first.")
    e = pd.read_csv(path)
    drift = float((e.hw_meas_j - (e.hw_gpu_j + e.hw_cpu_j)).abs().max())
    if drift > 1e-6:
        raise SystemExit(
            f"20_saturation: v2_instrument_epochs.csv has hw_meas_j differing "
            f"from hw_gpu_j + hw_cpu_j by up to {drift:g} J, so the 32x32 "
            "reference is not the same quantity as the 224 side.")
    e = e.rename(columns={"hw_meas_j": "energy_j", "hw_gpu_j": "gpu_energy_j",
                          "hw_cpu_j": "cpu_energy_j",
                          "duration_hw_s": "duration_s"})
    e["ecosystem"] = canonical(e["ecosystem"])
    return e


def v2_per_run() -> pd.DataFrame:
    """The 210-run campaign's runs, in the shape :func:`per_run` produces."""
    return per_run(v2_blocks(), ["ecosystem", "model", "dataset", "repetition"])


#: (column of the blocks frame, name of the cell mean it becomes).
SPREAD_QUANTITIES = (("energy_j", "energy_total_J_mean"),
                     ("gpu_energy_j", "energy_gpu_J_mean"),
                     ("duration_s", "duration_s_mean"))


def spread_means(blocks: pd.DataFrame) -> pd.DataFrame:
    """Per-cell means on the definition behind ``tab_spread``.

    ``tab_spread`` (12_paper_numbers.energy_facts) ranks the ecosystems of a
    block by ``mean_energy_J`` from ``v2_between_run_statistics.csv``, which is
    09_campaign_v2.between_run_stats: the mean over a run's epochs of the
    per-epoch counter energy, then the *mean* over the runs of the cell. That
    function is called here rather than re-implemented, on both sides of the
    contrast, and once per quantity -- the GPU-only energy and the duration go
    through the same two means by being put in its ``energy_j`` slot.
    """
    brs = sibling("09_campaign_v2").between_run_stats
    keys = ["ecosystem", "model", "dataset", "phase"]
    out = None
    for source, label in SPREAD_QUANTITIES:
        b = (brs(blocks.assign(energy_j=blocks[source]))[keys + ["mean_energy_J"]]
             .rename(columns={"mean_energy_J": label}))
        out = b if out is None else out.merge(b, on=keys, how="outer")
    return out


def v2_spread_means() -> pd.DataFrame:
    """:func:`spread_means` for the 32x32 side, checked against tab_spread's input.

    Recomputed from the per-epoch record so the GPU-only and time spreads are
    available on the same definition, and then compared with the
    ``mean_energy_J`` that ``tab_spread`` itself is built from: the reference a
    reader compares against must be the number that table prints, not a
    neighbour of it.
    """
    means = spread_means(v2_blocks())
    path = V2_TABLES / "v2_between_run_statistics.csv"
    if not path.exists():
        raise SystemExit(
            f"20_saturation: {path.relative_to(REPO_ROOT)} is missing, so the "
            "32x32 spread cannot be checked against tab_spread. Run "
            "09_campaign_v2.py against results/campaign_v2 first.")
    published = pd.read_csv(path)
    published["ecosystem"] = canonical(published["ecosystem"])
    keys = ["ecosystem", "model", "dataset", "phase"]
    check = means.merge(published[keys + ["mean_energy_J"]], on=keys,
                        how="outer")
    # v2_between_run_statistics is written rounded to 3 decimals.
    delta = (check.energy_total_J_mean - check.mean_energy_J).abs().max()
    if not np.isfinite(delta) or delta > 2e-3:
        raise SystemExit(
            "20_saturation: the 32x32 cell means recomputed from "
            "v2_instrument_epochs.csv differ from v2_between_run_statistics.csv "
            f"(tab_spread's input) by up to {delta} J, or cover different "
            "cells; one of them is stale. Re-run 09_campaign_v2.py and "
            "11_instrument_comparison.py against results/campaign_v2.")
    return means


def per_cell(runs: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR over the runs of each (ecosystem, model, dataset, phase).

    Between-run dispersion, at n = 5. The run is the unit; the epochs inside it
    are not repeated measurements of anything (R1 c5, R3 M2).
    """
    def q1(s):
        return float(np.percentile(s, 25))

    def q3(s):
        return float(np.percentile(s, 75))

    rows = []
    for keys, g in runs.groupby(["ecosystem", "model", "dataset", "phase"],
                                observed=True):
        eco, model, dataset, phase = keys
        rec = {"ecosystem": eco, "model": model, "dataset": dataset,
               "phase": phase, "n_runs": int(len(g)),
               "n_epochs": int(g.n_epochs.iloc[0])}
        for column, label in (("total_j_per_epoch", "energy_total_J"),
                              ("gpu_j_per_epoch", "energy_gpu_J"),
                              ("cpu_j_per_epoch", "energy_cpu_J"),
                              ("duration_s_per_epoch", "duration_s")):
            v = g[column].to_numpy(dtype=float)
            rec[f"{label}_median"] = float(np.median(v))
            rec[f"{label}_q1"] = q1(v)
            rec[f"{label}_q3"] = q3(v)
            rec[f"{label}_iqr"] = q3(v) - q1(v)
        # Energy-weighted mean power over every block of the cell, which is what
        # "mean GPU power during the training blocks" means when the blocks have
        # different lengths.
        rec["gpu_power_w_counter"] = float(g.gpu_j.sum() / g.duration_s.sum())
        rec["total_power_w_counter"] = float(g.total_j.sum() / g.duration_s.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def final_accuracy_by_run(blocks: pd.DataFrame) -> pd.DataFrame:
    """Final test accuracy per run: the last epoch that reported one.

    09's quality_normalised takes the same value the same way.
    """
    rows = []
    if "test_acc" not in blocks.columns:
        return pd.DataFrame(columns=["ecosystem", "model", "repetition",
                                     "final_test_acc_pct"])
    for (eco, model, rep), g in blocks.groupby(
            ["ecosystem", "model", "repetition"], observed=True):
        by_epoch = g.groupby("epoch")["test_acc"].max().dropna()
        if by_epoch.empty:
            continue
        rows.append({"ecosystem": eco, "model": model, "repetition": rep,
                     "final_test_acc_pct": float(by_epoch.iloc[-1])})
    return pd.DataFrame(rows, columns=["ecosystem", "model", "repetition",
                                       "final_test_acc_pct"])


def final_accuracy(blocks: pd.DataFrame) -> pd.DataFrame:
    """Final test accuracy per run, and its median over the runs of a cell.

    The last epoch that reported a test accuracy, per run -- 09's
    quality_normalised takes the same value the same way.
    """
    if "test_acc" not in blocks.columns:
        return pd.DataFrame(columns=["ecosystem", "model", "n_runs_with_acc",
                                     "final_test_acc_pct_median"])
    per = final_accuracy_by_run(blocks)
    if per.empty:
        return pd.DataFrame(columns=["ecosystem", "model", "n_runs_with_acc",
                                     "final_test_acc_pct_median"])
    return (per.groupby(["ecosystem", "model"], observed=True)
            .final_test_acc_pct.agg(n_runs_with_acc="size",
                                    final_test_acc_pct_median="median")
            .reset_index())


# --------------------------------------------------------------------------
# Utilisation over the training blocks, one code path for both campaigns
# --------------------------------------------------------------------------
#: The 1 Hz nvidia-smi record -- the file 19_gpu_utilisation reads -- or, from
#: a clone, each campaign's packaged excerpt of it (common.load_gpu_record).
UTIL_RECORD = GPU_RECORD

#: Per-run columns of the utilisation join; 19's names, so the two read alike.
UTIL_BY_RUN_COLUMNS = ["run", "ecosystem", "model", "dataset", "repetition",
                       "n_blocks", "n_samples", "util_mean_pct",
                       "mem_mean_mib", "power_mean_w", "power_min_w",
                       "power_max_w"]


def training_utilisation(campaign_dir: Path) -> tuple[pd.DataFrame, list[str], int]:
    """``(per-run utilisation, uncovered or rejected runs, complete runs)``.

    The window definition, stated once because the contrast rests on it: a
    run's utilisation is taken over its *training blocks* only -- the
    counter-bracketed interval of each training epoch -- not over the whole
    run. 19_gpu_utilisation uses the whole run, [manifest
    ``machine_state.utc``, ``counters.csv`` mtime], which also takes in
    start-up, data loading, every evaluation block and CodeCarbon's own
    start/stop; that share of the wall clock differs between 32x32 and 224, so
    a whole-run mean is diluted by a different amount on each side. How a
    block's interval is recovered from the emissions files' mtimes, and how
    those mtimes are checked, is in common._block_intervals; from a clone the
    windows come frozen in the campaign's replication package
    (common.utilisation_windows).

    The 1 Hz record is joined to those windows. A sample counts if its
    ``unix_s`` lies inside a training block's window, inclusive; ``unix_s`` is
    ``date +%s`` taken just after the query, and ``utilization.gpu`` describes
    the sample period before the query, so the stamp sits near the middle of
    the interval the sample describes and the edge error is under a second
    per block on both sides alike. Coverage follows 19's rule on the narrower
    window: a run counts only if every one of its training blocks lies inside
    the record. Called with the saturation directory and with
    ``results/campaign_v2``, so the two sides of the contrast share every line.
    """
    run_w, windows, rejected, complete, source = utilisation_windows(campaign_dir)
    record, record_source = load_gpu_record(campaign_dir, run_w, windows)
    print(f"  {Path(campaign_dir).name}: windows from the {source}, samples "
          f"from the {record_source}")
    if record is None or windows.empty:
        return (pd.DataFrame(columns=UTIL_BY_RUN_COLUMNS),
                sorted(set(rejected) | set(windows.run)), complete)
    record = record.sort_values("unix_s", kind="stable").reset_index(drop=True)
    t = record.unix_s.to_numpy(dtype=float)
    rows, uncovered = [], list(rejected)
    for run, w in windows.groupby("run", sort=True):
        if w.start_unix.min() < t[0] or w.end_unix.max() > t[-1]:
            uncovered.append(run)
            continue
        lo = np.searchsorted(t, w.start_unix.to_numpy(dtype=float), side="left")
        hi = np.searchsorted(t, w.end_unix.to_numpy(dtype=float), side="right")
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
        if idx.size == 0:
            uncovered.append(run)
            continue
        s = record.iloc[idx]
        meta = w.iloc[0]
        rows.append({
            "run": run,
            "ecosystem": meta.ecosystem, "model": meta.model,
            "dataset": meta.dataset, "repetition": meta.repetition,
            "n_blocks": len(w), "n_samples": int(idx.size),
            "util_mean_pct": float(s.utilisation_pct.mean()),
            "mem_mean_mib": float(s.memory_used_mib.mean()),
            "power_mean_w": float(s.power_w.mean()),
            "power_min_w": float(s.power_w.min()),
            "power_max_w": float(s.power_w.max()),
        })
    runs = pd.DataFrame(rows, columns=UTIL_BY_RUN_COLUMNS)
    if not runs.empty:
        runs["ecosystem"] = canonical(runs["ecosystem"])
    return runs, sorted(uncovered), complete


def saturation_utilisation() -> tuple[pd.DataFrame, list[str], int]:
    """``(per-run utilisation, uncovered run names, complete runs)`` at 224."""
    return training_utilisation(SATURATION_DIR)


def v2_utilisation() -> pd.DataFrame:
    """The 32x32 utilisation and sampled board power, per (ecosystem, model).

    Recomputed from ``results/campaign_v2`` by :func:`training_utilisation`,
    the code path the 224 side goes through, and deliberately *not* read from
    19's ``v2_gpu_utilisation_*`` tables: those are whole-run means, a
    different window. Aggregated as 19 aggregates: the mean over the covered
    runs of each (ecosystem, model), pooled over the three datasets.
    """
    by_run, uncovered, complete = training_utilisation(V2_CAMPAIGN)
    print(f"32x32 utilisation (training blocks): {len(by_run)} of {complete} "
          f"complete run(s) covered by the record")
    if by_run.empty:
        raise SystemExit(
            "20_saturation: no 32x32 run's training blocks fall inside "
            f"{UTIL_RECORD} or its packaged excerpt; the utilisation contrast "
            "has no reference side.")
    agg = (by_run.groupby(["ecosystem", "model"], observed=True)
           .agg(n_runs_covered=("run", "size"),
                util_mean_pct=("util_mean_pct", "mean"),
                power_mean_w=("power_mean_w", "mean"))
           .reset_index())
    # How many runs each pair has to be covered out of (3 datasets x 5), from
    # the per-block record of the 210 runs, so the coverage can be stated.
    complete = (v2_blocks().groupby(["ecosystem", "model"], observed=True)
                [["dataset", "repetition"]].apply(
                    lambda g: len(g.drop_duplicates()))
                .rename("n_runs_complete").reset_index())
    return agg.merge(complete, on=["ecosystem", "model"], how="left")


# --------------------------------------------------------------------------
# Output 1 -- energy, time, accuracy and utilisation, per cell
# --------------------------------------------------------------------------
def energy_by_ecosystem(cells: pd.DataFrame, accuracy: pd.DataFrame,
                        util: pd.DataFrame, complete_runs: int) -> pd.DataFrame:
    """One row per (ecosystem, model): everything the cell measured at 224."""
    train = cells[cells.phase == "Training"].set_index(["ecosystem", "model"])
    infer = cells[cells.phase == "Inference"].set_index(["ecosystem", "model"])
    acc = accuracy.set_index(["ecosystem", "model"]) if not accuracy.empty \
        else pd.DataFrame()
    if util.empty:
        util_by_cell = pd.DataFrame()
    else:
        util_by_cell = (util.groupby(["ecosystem", "model"], observed=True)
                        .agg(n_runs_covered=("run", "size"),
                             n_util_samples=("n_samples", "sum"),
                             util_mean_pct=("util_mean_pct", "mean"),
                             sampler_power_mean_w=("power_mean_w", "mean"))
                        .reset_index()
                        .set_index(["ecosystem", "model"]))
    rows = []
    for key in sorted(train.index, key=lambda k: (ECOSYSTEM_ORDER.index(k[0])
                                                  if k[0] in ECOSYSTEM_ORDER
                                                  else 99, k[1])):
        t, i = train.loc[key], infer.loc[key]
        eco, model = key
        rec = {
            "ecosystem": eco, "model": model,
            "n_runs": int(t.n_runs), "n_epochs": int(t.n_epochs),
            "train_energy_total_J_median": t.energy_total_J_median,
            "train_energy_total_J_iqr": t.energy_total_J_iqr,
            "train_energy_gpu_J_median": t.energy_gpu_J_median,
            "train_energy_gpu_J_iqr": t.energy_gpu_J_iqr,
            "train_energy_cpu_J_median": t.energy_cpu_J_median,
            "train_energy_cpu_J_iqr": t.energy_cpu_J_iqr,
            "train_duration_s_median": t.duration_s_median,
            "train_duration_s_iqr": t.duration_s_iqr,
            "train_gpu_power_w_counter": t.gpu_power_w_counter,
            "infer_energy_total_J_median": i.energy_total_J_median,
            "infer_energy_total_J_iqr": i.energy_total_J_iqr,
            "infer_energy_gpu_J_median": i.energy_gpu_J_median,
            "infer_energy_gpu_J_iqr": i.energy_gpu_J_iqr,
            "infer_duration_s_median": i.duration_s_median,
            "infer_duration_s_iqr": i.duration_s_iqr,
            "infer_gpu_power_w_counter": i.gpu_power_w_counter,
        }
        if not acc.empty and key in acc.index:
            rec["final_test_acc_pct_median"] = float(
                acc.loc[key, "final_test_acc_pct_median"])
            rec["n_runs_with_acc"] = int(acc.loc[key, "n_runs_with_acc"])
        else:
            rec["final_test_acc_pct_median"] = np.nan
            rec["n_runs_with_acc"] = 0
        if not util_by_cell.empty and key in util_by_cell.index:
            u = util_by_cell.loc[key]
            covered = int(u.n_runs_covered)
            rec.update({
                "util_mean_pct": float(u.util_mean_pct),
                "util_n_runs_covered": covered,
                "util_n_samples": int(u.n_util_samples),
                "sampler_power_mean_w": float(u.sampler_power_mean_w),
                "util_coverage": ("full" if covered >= int(t.n_runs)
                                  else "partial"),
            })
        else:
            rec.update({"util_mean_pct": np.nan, "util_n_runs_covered": 0,
                        "util_n_samples": 0, "sampler_power_mean_w": np.nan,
                        "util_coverage": "none"})
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs["complete_runs"] = complete_runs
    return out


# --------------------------------------------------------------------------
# Output 2 -- the spread, at 224 and at 32x32
# --------------------------------------------------------------------------
def _spread(frame: pd.DataFrame, column: str, subset: list[str] | None):
    """``(max/min, cheapest, dearest, n)`` over the ecosystems of one cell."""
    s = frame.set_index("ecosystem")[column]
    if subset is not None:
        s = s[[e for e in subset if e in s.index]]
    s = s.dropna()
    if len(s) < 2 or s.min() <= 0:
        return np.nan, None, None, len(s)
    return float(s.max() / s.min()), str(s.idxmin()), str(s.idxmax()), len(s)


def spread_table(cells_224: pd.DataFrame, cells_32: pd.DataFrame) -> pd.DataFrame:
    """The ecosystem spread at 224 against the same spread at 32x32.

    Both arguments come from :func:`spread_means`, so the spread on both sides
    is the one ``tab_spread`` prints: max/min over the ecosystems of the
    between-run *mean* of the per-run per-epoch energy (not the median that
    ``sat_energy_by_ecosystem`` reports per cell). ``best_*``/``worst_*`` are
    the lowest/highest such mean, which is how ``tab_spread`` names the
    "Lowest"/"Highest" stack of a block.

    One row per (subset, model, phase, 32x32 dataset). ``compression_factor``
    is the ratio of ratios: how many times narrower the ecosystem spread is
    when the accelerator is loaded. Above 1 the spread compresses, which is the
    outcome that would delimit the study's conclusion; at or below 1 the
    objection does not survive.
    """
    subsets = [("all seven", None), ("LibTorch family", LIBTORCH_FAMILY),
               ("shared module", shared_module_trio()),
               ("all but Java", NO_JAVA)]
    rows = []
    for label, subset in subsets:
        for model in sorted(cells_224.model.unique()):
            for phase in ("Training", "Inference"):
                here = cells_224[(cells_224.model == model)
                                 & (cells_224.phase == phase)]
                if here.empty:
                    continue
                s224, best224, worst224, n224 = _spread(
                    here, "energy_total_J_mean", subset)
                g224, _, _, _ = _spread(here, "energy_gpu_J_mean", subset)
                t224, _, _, _ = _spread(here, "duration_s_mean", subset)
                for dataset in sorted(cells_32.dataset.unique()):
                    there = cells_32[(cells_32.model == model)
                                     & (cells_32.phase == phase)
                                     & (cells_32.dataset == dataset)]
                    if there.empty:
                        continue
                    s32, best32, worst32, n32 = _spread(
                        there, "energy_total_J_mean", subset)
                    g32, _, _, _ = _spread(there, "energy_gpu_J_mean", subset)
                    t32, _, _, _ = _spread(there, "duration_s_mean", subset)
                    rows.append({
                        "subset": label, "model": model, "phase": phase,
                        "reference_dataset": dataset,
                        "n_ecosystems_224": n224, "n_ecosystems_32": n32,
                        "spread_224": s224,
                        "best_224": best224, "worst_224": worst224,
                        "spread_32": s32,
                        "best_32": best32, "worst_32": worst32,
                        "compression_factor": (s32 / s224 if s224 and
                                               np.isfinite(s224) else np.nan),
                        "spread_gpu_224": g224, "spread_gpu_32": g32,
                        "compression_factor_gpu": (g32 / g224 if g224 and
                                                   np.isfinite(g224) else np.nan),
                        "spread_time_224": t224, "spread_time_32": t32,
                        "compression_factor_time": (t32 / t224 if t224 and
                                                    np.isfinite(t224) else np.nan),
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Output 3 -- does the ordering survive?
# --------------------------------------------------------------------------
def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided permutation p for Spearman's rho, by exact enumeration.

    The pairings permutation test ``scipy.stats.permutation_test`` performs
    with ``permutation_type="pairings"``: the null holds one ranking fixed and
    permutes the other, and with seven ecosystems there are 7! = 5040
    permutations, so the p-value is enumerated rather than sampled and does not
    depend on a resample count or a seed. (scipy's own implementation calls
    ``spearmanr`` once per permutation and takes minutes over a table this
    size; the closed form for rho on ranks is the same statistic.)

    ``scipy.stats.spearmanr``'s own p-value is asymptotic and is carried beside
    this one in the table, because at n = 7 the two differ by a factor of two
    and quoting the asymptotic one alone would overstate the evidence.
    """
    n = len(x)
    rx = np.asarray(stats.rankdata(x), dtype=float)
    ry = np.asarray(stats.rankdata(y), dtype=float)
    observed = float(stats.spearmanr(x, y).statistic)
    if not np.isfinite(observed):
        return float("nan")
    perms = np.array(list(itertools.permutations(range(n))))
    d = rx[None, :] - ry[perms]
    rho = 1.0 - 6.0 * (d * d).sum(axis=1) / (n * (n * n - 1))
    return float((np.abs(rho) >= abs(observed) - 1e-12).mean())


def rank_agreement(cells_224: pd.DataFrame, cells_32: pd.DataFrame) -> pd.DataFrame:
    """Does 224 order the ecosystems the way 32x32 does?

    Spearman between the two orderings of the seven ecosystems by median
    per-epoch counter energy, one row per (model, phase, 32x32 dataset), with
    the ecosystems whose rank moves by two places or more named rather than
    counted. Rank 1 is the cheapest.
    """
    rows = []
    for model in sorted(cells_224.model.unique()):
        for phase in ("Training", "Inference"):
            a = cells_224[(cells_224.model == model) & (cells_224.phase == phase)]
            if a.empty:
                continue
            a = a.set_index("ecosystem")["energy_total_J_median"]
            for dataset in sorted(cells_32.dataset.unique()):
                b = cells_32[(cells_32.model == model)
                             & (cells_32.phase == phase)
                             & (cells_32.dataset == dataset)]
                if b.empty:
                    continue
                b = b.set_index("ecosystem")["energy_total_J_median"]
                shared = [e for e in order_ecosystems(a.index) if e in b.index]
                if len(shared) < 3:
                    continue
                x = a.loc[shared].to_numpy(dtype=float)
                y = b.loc[shared].to_numpy(dtype=float)
                rho = float(stats.spearmanr(x, y).statistic)
                p_asym = float(stats.spearmanr(x, y).pvalue)
                p_perm = exact_spearman_p(x, y)
                rank_224 = stats.rankdata(x)
                rank_32 = stats.rankdata(y)
                moved = [f"{SHORT.get(e, e)} {int(r32)}->{int(r224)}"
                         for e, r224, r32 in zip(shared, rank_224, rank_32)
                         if abs(r224 - r32) >= 2]
                rows.append({
                    "model": model, "phase": phase,
                    "reference_dataset": dataset,
                    "n_ecosystems": len(shared),
                    # The rho is rounded where it is computed and the p-values
                    # are not, exactly as 14_v2_statistics does it: a p of
                    # 1e-9 rounded for display is a p of zero on the page.
                    "spearman_rho": round(rho, 3),
                    "p_permutation_exact": round(p_perm, 6),
                    "p_asymptotic": p_asym,
                    "n_permutations": int(np.prod(
                        np.arange(1, len(shared) + 1, dtype=np.int64))),
                    "n_moved_two_or_more": len(moved),
                    "moved_two_or_more": "; ".join(moved) if moved else "none",
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Output 4 -- time against energy, the definition behind \vRhoTrain
# --------------------------------------------------------------------------
def time_energy(runs_224: pd.DataFrame) -> pd.DataFrame:
    """rho(time, energy) over the 14 cells at 224, as 14_v2_statistics defines it.

    Run totals (counter energy and counter duration each summed over the run's
    epochs), median over repetitions within a cell, then the correlation across
    cells, one row per phase. That is exactly what feeds ``\\vRhoTrain`` and
    ``\\vRhoInfer``; the main campaign's own values are read from
    ``v2_stats_energy_time.csv`` and carried beside these so the two are
    comparable on the page without being recomputed here.
    """
    reference = {}
    ref_path = V2_TABLES / "v2_stats_energy_time.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path).set_index("phase")
        reference = ref.to_dict("index")
    rows = []
    for phase, g in runs_224.groupby("phase", observed=True):
        med = (g.groupby(["ecosystem", "model", "dataset"], observed=True)
               [["total_j", "duration_s"]].median())
        if len(med) < 3:
            continue
        rho, p_rho = stats.spearmanr(med.total_j, med.duration_s)
        r, p_r = stats.pearsonr(med.total_j, med.duration_s)
        vals = med.reset_index()
        inversions = sum(
            1 for (_, a), (_, b) in itertools.combinations(vals.iterrows(), 2)
            if (a.total_j - b.total_j) * (a.duration_s - b.duration_s) < 0)
        total = len(vals) * (len(vals) - 1) // 2
        ref = reference.get(phase, {})
        rows.append({
            "phase": phase, "n_cells": len(vals),
            "spearman_rho": round(float(rho), 3), "spearman_p": float(p_rho),
            "pearson_r": round(float(r), 3), "pearson_p": float(p_r),
            "discordant_pairs": inversions, "total_pairs": total,
            "discordant_pct": round(100 * inversions / total, 1),
            "reference_spearman_rho_32": ref.get("spearman_rho", np.nan),
            "reference_n_configurations_32": ref.get("n_configurations", np.nan),
            "reference_discordant_pct_32": ref.get("discordant_pct", np.nan),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Output 5 -- utilisation, side by side
# --------------------------------------------------------------------------
def utilisation_contrast(by_eco: pd.DataFrame, v2_util: pd.DataFrame) -> pd.DataFrame:
    """The number the objection is about: how busy the card is, at both shapes.

    Both sides are the same 1 Hz record over the same window -- the training
    blocks, through :func:`training_utilisation` on both campaigns -- so the
    difference is the workload rather than the instrument. The board power
    beside it is the sampler's, on both sides, not the counter-derived power in
    ``sat_energy_by_ecosystem``: those are different quantities over different
    windows and putting them in one column would be the defect this revision
    catalogues.
    """
    left = v2_util.rename(columns={"util_mean_pct": "util_mean_pct_32",
                                   "power_mean_w": "power_mean_w_32",
                                   "n_runs_covered": "n_runs_covered_32",
                                   "n_runs_complete": "n_runs_complete_32"})
    right = by_eco[["ecosystem", "model", "util_mean_pct",
                    "sampler_power_mean_w", "util_n_runs_covered",
                    "util_n_samples", "util_coverage"]].rename(
        columns={"util_mean_pct": "util_mean_pct_224",
                 "sampler_power_mean_w": "power_mean_w_224",
                 "util_n_runs_covered": "n_runs_covered_224",
                 "util_n_samples": "n_samples_224",
                 "util_coverage": "coverage_224"})
    m = right.merge(left, on=["ecosystem", "model"], how="left")
    m["util_increase_pp"] = m.util_mean_pct_224 - m.util_mean_pct_32
    m["util_ratio"] = m.util_mean_pct_224 / m.util_mean_pct_32
    m["power_increase_w"] = m.power_mean_w_224 - m.power_mean_w_32
    m["power_ratio"] = m.power_mean_w_224 / m.power_mean_w_32
    m["ecosystem_order"] = m.ecosystem.map(
        lambda e: ECOSYSTEM_ORDER.index(e) if e in ECOSYSTEM_ORDER else 99)
    return (m.sort_values(["ecosystem_order", "model"])
            .drop(columns="ecosystem_order").reset_index(drop=True))


# --------------------------------------------------------------------------
# Macros
# --------------------------------------------------------------------------
def loader_threads(campaign_dir: Path) -> int:
    """``DEEPGREEN_LOADER_THREADS`` as every complete run's manifest records it.

    Refuses unless all of them record it and record the same value: one number
    on the page is only honest if it is the number for every run.
    """
    values = {}
    for run_dir in sorted(p for p in Path(campaign_dir).glob("*") if p.is_dir()):
        if read_complete_counters(run_dir)[0] is None:
            continue
        try:
            env = json.loads((run_dir / "manifest.json").read_text()).get("env") or {}
        except (OSError, ValueError):
            env = {}
        values[run_dir.name] = env.get("DEEPGREEN_LOADER_THREADS")
    distinct = set(values.values())
    if not values or len(distinct) != 1 or None in distinct:
        raise SystemExit(
            "20_saturation: the manifests do not agree on DEEPGREEN_LOADER_THREADS "
            f"({sorted(map(str, distinct))} over {len(values)} runs).")
    return int(distinct.pop())


def check_control_against_tab_control(spread: pd.DataFrame) -> None:
    """The 32x32 shared-module spread must be tab_control's, cell for cell.

    tab_control (12_paper_numbers, from 14_v2_statistics.libtorch_control)
    takes per (model, dataset), training only, the mean over runs of each
    run's *total* energy; ``sat_spread`` takes the mean over runs of each run's
    per-epoch mean. Every run has the same epoch count, so the two ratios are
    the same number -- checked here against the table the paper prints.
    """
    path = V2_TABLES / "v2_stats_libtorch_control.csv"
    if not path.exists():
        raise SystemExit(f"20_saturation: {path.relative_to(REPO_ROOT)} is "
                         "missing; run 14_v2_statistics.py first.")
    ct = pd.read_csv(path)
    ours = spread[(spread.subset == "shared module")
                  & (spread.phase == "Training")]
    m = ours.merge(ct, left_on=["model", "reference_dataset"],
                   right_on=["model", "dataset"], how="outer")
    bad = m[~((m.spread_32 - m.spread_shared_module).abs() <= 0.0051)
            | (m.n_ecosystems_32 != m.n_shared_module)]
    if len(bad):
        raise SystemExit(
            "20_saturation: the 32x32 shared-module spread does not reproduce "
            "v2_stats_libtorch_control.csv (tab_control):\n"
            + bad[["model", "reference_dataset", "spread_32",
                   "spread_shared_module"]].to_string(index=False))


def emit_macros(by_eco: pd.DataFrame, spread: pd.DataFrame,
                ranks: pd.DataFrame, rho: pd.DataFrame,
                util: pd.DataFrame, complete_runs: int,
                blocks: pd.DataFrame, run_accuracy: pd.DataFrame,
                threads: int) -> None:
    """Every number the paper needs to state the contrast without typing digits."""
    macro("vSatRuns", complete_runs)
    # The cell's total measured energy, on the definition behind
    # \vCampaignEnergyMJ/KWh (common.campaign_energy, which 12 calls too):
    # counter energy, GPU + CPU package, over every measured block, both phases.
    energy_mj, energy_kwh = campaign_energy(blocks.energy_j)
    macro("vSatCampaignEnergyMJ", energy_mj)
    macro("vSatCampaignEnergyKWh", energy_kwh)
    macro("vSatRunsExpected", EXPECTED_SATURATION_RUNS)
    macro("vSatEcosystems", by_eco.ecosystem.nunique())
    macro("vSatModels", by_eco.model.nunique())
    macro("vSatEpochs", EXPECTED_EPOCHS)
    macro("vSatReps", int(by_eco.n_runs.max()))

    # --- energy and time at 224 ------------------------------------------
    macro("vSatEnergyMinKJ", num(by_eco.train_energy_total_J_median.min() / 1000, 1))
    macro("vSatEnergyMaxKJ", num(by_eco.train_energy_total_J_median.max() / 1000, 1))
    macro("vSatTrainTimeMin", num(by_eco.train_duration_s_median.min(), 1))
    macro("vSatTrainTimeMax", num(by_eco.train_duration_s_median.max(), 1))
    macro("vSatInferTimeMin", num(by_eco.infer_duration_s_median.min(), 1))
    macro("vSatInferTimeMax", num(by_eco.infer_duration_s_median.max(), 1))
    macro("vSatPowerCounterMin", num(by_eco.train_gpu_power_w_counter.min(), 0))
    macro("vSatPowerCounterMax", num(by_eco.train_gpu_power_w_counter.max(), 0))

    # --- the spread, and how far it compresses ---------------------------
    seven = spread[spread.subset == "all seven"]
    family = spread[spread.subset == "LibTorch family"]

    def spread_macros(frame: pd.DataFrame, prefix: str) -> None:
        for (model, phase), g in frame.groupby(["model", "phase"], observed=True):
            tag = f"{PHASE_TAG[phase]}{MODEL_TAG[model]}"
            macro(f"{prefix}Spread{tag}", num(g.spread_224.iloc[0], 1))
            macro(f"{prefix}SpreadGpu{tag}", num(g.spread_gpu_224.iloc[0], 1))
            macro(f"{prefix}SpreadTime{tag}", num(g.spread_time_224.iloc[0], 1))
            for _, r in g.iterrows():
                ds = DATASET_TAG[r.reference_dataset]
                macro(f"{prefix}RefSpread{tag}{ds}", num(r.spread_32, 1))
                macro(f"{prefix}Compression{tag}{ds}", num(r.compression_factor, 2))
            macro(f"{prefix}RefSpread{tag}Min", num(g.spread_32.min(), 1))
            macro(f"{prefix}RefSpread{tag}Max", num(g.spread_32.max(), 1))
            macro(f"{prefix}Compression{tag}Min", num(g.compression_factor.min(), 2))
            macro(f"{prefix}Compression{tag}Max", num(g.compression_factor.max(), 2))
        macro(f"{prefix}SpreadMin", num(frame.spread_224.min(), 1))
        macro(f"{prefix}SpreadMax", num(frame.spread_224.max(), 1))
        macro(f"{prefix}RefSpreadMin", num(frame.spread_32.min(), 1))
        macro(f"{prefix}RefSpreadMax", num(frame.spread_32.max(), 1))
        macro(f"{prefix}CompressionMin", num(frame.compression_factor.min(), 2))
        macro(f"{prefix}CompressionMax", num(frame.compression_factor.max(), 2))
        for phase, g in frame.groupby("phase", observed=True):
            macro(f"{prefix}Compression{PHASE_TAG[phase]}Min",
                  num(g.compression_factor.min(), 2))
            macro(f"{prefix}Compression{PHASE_TAG[phase]}Max",
                  num(g.compression_factor.max(), 2))

    spread_macros(seven, "vSat")
    spread_macros(family, "vSatLibtorch")
    # The control trio (tab_control's group) and the six without Java, on the
    # same definition; the 32x32 control side is checked against tab_control in
    # check_control_against_tab_control.
    spread_macros(spread[spread.subset == "shared module"], "vSatCtrl")
    spread_macros(spread[spread.subset == "all but Java"], "vSatNoJava")
    macro("vSatLoaderThreads", threads)

    # The cheapest and dearest training stack at 32x32, per architecture, on
    # the sat_spread definition -- named only where all three 32x32 datasets
    # agree on it, because one name for three blocks is otherwise a choice.
    for model, g in seven[seven.phase == "Training"].groupby("model", observed=True):
        for column, word in (("best_32", "Best"), ("worst_32", "Worst")):
            names = sorted(set(g[column]))
            if len(names) == 1:
                macro(f"vSatRef{word}Train{MODEL_TAG[model]}",
                      SHORT.get(names[0], names[0]))
            else:
                print(f"  !! no \\vSatRef{word}Train{MODEL_TAG[model]}: the "
                      f"32x32 datasets disagree ({', '.join(names)})")
    # The two the task list names without an architecture: the worse of the two
    # networks, so a sentence quoting them cannot understate the spread.
    for phase in ("Training", "Inference"):
        tag = PHASE_TAG[phase]
        g = family[family.phase == phase]
        if not g.empty:
            macro(f"vSatLibtorchSpread{tag}", num(g.spread_224.max(), 1))
    # The cheapest and dearest stack are named per architecture and phase only,
    # as tab_spread names the "Lowest"/"Highest" stack of each block: lowest
    # (highest) between-run mean per-epoch energy. There is no pooled-across-
    # models winner anywhere else in the paper, so none is emitted here.
    for (model, phase), g in seven.groupby(["model", "phase"], observed=True):
        tag = f"{PHASE_TAG[phase]}{MODEL_TAG[model]}"
        macro(f"vSatBest{tag}", SHORT.get(g.best_224.iloc[0], g.best_224.iloc[0]))
        macro(f"vSatWorst{tag}", SHORT.get(g.worst_224.iloc[0], g.worst_224.iloc[0]))

    # --- rank agreement ---------------------------------------------------
    if not ranks.empty:
        for _, r in ranks.iterrows():
            tag = (f"{PHASE_TAG[r.phase]}{MODEL_TAG[r.model]}"
                   f"{DATASET_TAG[r.reference_dataset]}")
            macro(f"vSatRankRho{tag}", num(r.spearman_rho, 2))
        macro("vSatRankRhoMin", num(ranks.spearman_rho.min(), 2))
        macro("vSatRankRhoMax", num(ranks.spearman_rho.max(), 2))
        macro("vSatRankComparisons", len(ranks))
        macro("vSatRankMovedMax", int(ranks.n_moved_two_or_more.max()))
        macro("vSatRankMovedTotal", int(ranks.n_moved_two_or_more.sum()))
        macro("vSatRankPermMin", num(ranks.p_permutation_exact.min(), 4))
        macro("vSatRankPermMax", num(ranks.p_permutation_exact.max(), 4))

    # --- time against energy ---------------------------------------------
    if not rho.empty:
        r = rho.set_index("phase")
        for phase, tag in (("Training", "Train"), ("Inference", "Infer")):
            if phase not in r.index:
                continue
            macro(f"vSatRho{tag}", num(r.loc[phase, "spearman_rho"], 2))
            macro(f"vSatRho{tag}Pearson", num(r.loc[phase, "pearson_r"], 2))
            macro(f"vSatDiscordant{tag}Pct",
                  num(r.loc[phase, "discordant_pct"], 1))
            ref = r.loc[phase, "reference_spearman_rho_32"]
            if np.isfinite(ref):
                macro(f"vSatRefRho{tag}", num(ref, 2))
        macro("vSatRhoCells", int(r.n_cells.max()))

    # --- accuracy ---------------------------------------------------------
    acc = by_eco.dropna(subset=["final_test_acc_pct_median"])
    if not acc.empty:
        lo = acc.loc[acc.final_test_acc_pct_median.idxmin()]
        hi = acc.loc[acc.final_test_acc_pct_median.idxmax()]
        macro("vSatAccMin", num(lo.final_test_acc_pct_median, 1))
        macro("vSatAccMax", num(hi.final_test_acc_pct_median, 1))
        macro("vSatAccMinEco", SHORT.get(lo.ecosystem, lo.ecosystem))
        macro("vSatAccMaxEco", SHORT.get(hi.ecosystem, hi.ecosystem))
        macro("vSatAccMinModel", MODEL_LABEL.get(lo.model, lo.model))
        macro("vSatAccMaxModel", MODEL_LABEL.get(hi.model, hi.model))
        macro("vSatAccMedian", num(acc.final_test_acc_pct_median.median(), 1))
        if not run_accuracy.empty:
            worst = run_accuracy.loc[run_accuracy.final_test_acc_pct.idxmin()]
            macro("vSatAccRunMin", num(worst.final_test_acc_pct, 1))
            macro("vSatAccRunMinEco", SHORT.get(worst.ecosystem, worst.ecosystem))
            macro("vSatAccRunMinModel", MODEL_LABEL.get(worst.model, worst.model))
        for model, g in acc.groupby("model", observed=True):
            macro(f"vSatAcc{MODEL_TAG[model]}Min",
                  num(g.final_test_acc_pct_median.min(), 1))
            macro(f"vSatAcc{MODEL_TAG[model]}Max",
                  num(g.final_test_acc_pct_median.max(), 1))

    # --- utilisation ------------------------------------------------------
    covered = util.dropna(subset=["util_mean_pct_224"])
    if not covered.empty:
        lo = covered.loc[covered.util_mean_pct_224.idxmin()]
        hi = covered.loc[covered.util_mean_pct_224.idxmax()]
        macro("vSatUtilMin", num(lo.util_mean_pct_224, 1))
        macro("vSatUtilMax", num(hi.util_mean_pct_224, 1))
        macro("vSatUtilMinEco", SHORT.get(lo.ecosystem, lo.ecosystem))
        macro("vSatUtilMaxEco", SHORT.get(hi.ecosystem, hi.ecosystem))
        macro("vSatUtilMinModel", MODEL_LABEL.get(lo.model, lo.model))
        macro("vSatUtilMaxModel", MODEL_LABEL.get(hi.model, hi.model))
        macro("vSatUtilMedian", num(covered.util_mean_pct_224.median(), 1))
        macro("vSatUtilCells", len(covered))
        macro("vSatUtilRuns", int(covered.n_runs_covered_224.sum()))
        macro("vSatUtilRunsOf", complete_runs)
        macro("vSatUtilSamples", int(covered.n_samples_224.sum()))
        both = covered.dropna(subset=["util_mean_pct_32"])
        if not both.empty:
            rlo = both.loc[both.util_mean_pct_32.idxmin()]
            rhi = both.loc[both.util_mean_pct_32.idxmax()]
            # Coverage of the 32x32 side: the record began after the main
            # campaign did, so each (ecosystem, model) pair is covered by a
            # subset of its runs.
            macro("vSatRefUtilRunsMin", int(both.n_runs_covered_32.min()))
            macro("vSatRefUtilRunsMax", int(both.n_runs_covered_32.max()))
            macro("vSatRefUtilRunsPerPair", int(both.n_runs_complete_32.max()))
            macro("vSatRefUtilRuns", int(both.n_runs_covered_32.sum()))
            macro("vSatRefUtilRunsOf", int(both.n_runs_complete_32.sum()))
            macro("vSatRefUtilMin", num(rlo.util_mean_pct_32, 1))
            macro("vSatRefUtilMax", num(rhi.util_mean_pct_32, 1))
            macro("vSatRefUtilMinEco", SHORT.get(rlo.ecosystem, rlo.ecosystem))
            macro("vSatRefUtilMaxEco", SHORT.get(rhi.ecosystem, rhi.ecosystem))
            # On the training-block window, over the 14 (ecosystem, model)
            # pairs, and per architecture.
            macro("vSatRefUtilMedian", num(both.util_mean_pct_32.median(), 1))
            for model, g in both.groupby("model", observed=True):
                tag = MODEL_TAG[model]
                macro(f"vSatRefUtil{tag}Min", num(g.util_mean_pct_32.min(), 1))
                macro(f"vSatRefUtil{tag}Max", num(g.util_mean_pct_32.max(), 1))
                macro(f"vSatRefUtil{tag}HighCount",
                      int((g.util_mean_pct_32 >= SAT_UTIL_HIGH_PCT).sum()))
            macro("vSatRefUtilHighPct", num(SAT_UTIL_HIGH_PCT, 0))
            macro("vSatUtilGainMin", num(both.util_increase_pp.min(), 1))
            macro("vSatUtilGainMax", num(both.util_increase_pp.max(), 1))
            macro("vSatUtilGainMedian", num(both.util_increase_pp.median(), 1))
            macro("vSatPowerMin", num(both.power_mean_w_224.min(), 0))
            macro("vSatPowerMax", num(both.power_mean_w_224.max(), 0))
            macro("vSatRefPowerMin", num(both.power_mean_w_32.min(), 0))
            macro("vSatRefPowerMax", num(both.power_mean_w_32.max(), 0))
            macro("vSatPowerGainMin", num(both.power_increase_w.min(), 0))
            macro("vSatPowerGainMax", num(both.power_increase_w.max(), 0))


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------
def _span(lo: float, hi: float, digits: int = 1) -> str:
    """``lo--hi`` at one precision, or a single value when the ends round equal."""
    a, b = f"{lo:.{digits}f}", f"{hi:.{digits}f}"
    return a if a == b else f"{a}--{b}"


def saturation_table(by_eco: pd.DataFrame, spread: pd.DataFrame,
                     util: pd.DataFrame, path: Path) -> None:
    """Table~\\ref{tab:saturation}: the cell, stack by stack, against 32x32.

    Set to fit a two-column float at \\footnotesize rather than scaled down to
    fit it: a \\resizebox changes the type size of one table relative to the
    rest of the paper and hides that it does. The two architectures are
    stacked as panels rather than set side by side: side by side the table was
    nine columns and 35 pt wider than the single-column page of the EMSE
    layout (paper/emse/main.tex); stacked it is five and fits both layouts.
    """
    models = [m for m in ("resnet18", "vgg16") if m in set(by_eco.model)]
    util_by_cell = util.set_index(["ecosystem", "model"]) if not util.empty \
        else pd.DataFrame()
    energy = by_eco.set_index(["ecosystem", "model"])

    def cell(eco: str, model: str) -> list[str]:
        key = (eco, model)
        if key not in energy.index:
            return ["--", "--", "--", "--"]
        e = energy.loc[key]
        out = [f"{e.train_energy_total_J_median / 1000:.1f}",
               f"{e.train_duration_s_median:.1f}"]
        if not util_by_cell.empty and key in util_by_cell.index:
            u = util_by_cell.loc[key]
            out.append("--" if not np.isfinite(u.util_mean_pct_224)
                       else f"{u.util_mean_pct_224:.1f}")
            out.append("--" if not np.isfinite(u.util_mean_pct_32)
                       else f"{u.util_mean_pct_32:.1f}")
        else:
            out += ["--", "--"]
        return out

    ecosystems = order_ecosystems(by_eco.ecosystem.unique())

    def util_range(model: str, column: str) -> str:
        """min--max of the utilisation column, over the seven ecosystems.

        A spread is a ratio for energy and time and a range for a percentage;
        the row label covers both and the note under the table says so.
        """
        if util.empty:
            return "--"
        s = util[util.model == model][column].dropna()
        return "--" if s.empty else _span(float(s.min()), float(s.max()))

    def spread_row(label: str, pick):
        """One row per model: ``model -> row``."""
        rows = {}
        for model in models:
            g = spread[(spread.subset == "all seven") & (spread.model == model)
                       & (spread.phase == "Training")]
            rows[model] = f"{label} & " + " & ".join(pick(g, model)) + r" \\"
        return rows

    rows_224 = spread_row(
        r"\textbf{Spread, 224}",
        lambda g, m: ["--" if g.empty else f"{g.spread_224.iloc[0]:.1f}$\\times$",
                      "--" if g.empty
                      else f"{g.spread_time_224.iloc[0]:.1f}$\\times$",
                      util_range(m, "util_mean_pct_224"), "--"])
    rows_32 = spread_row(
        r"\textbf{Spread, $32{\times}32$}",
        lambda g, m: ["--" if g.empty
                      else _span(g.spread_32.min(), g.spread_32.max())
                      + r"$\times$",
                      "--" if g.empty
                      else _span(g.spread_time_32.min(),
                                 g.spread_time_32.max()) + r"$\times$",
                      "--", util_range(m, "util_mean_pct_32")])
    rows_compress = spread_row(
        r"\textbf{Compression}",
        lambda g, m: ["--" if g.empty
                      else _span(g.compression_factor.min(),
                                 g.compression_factor.max(), 2) + r"$\times$",
                      "--" if g.empty
                      else _span(g.compression_factor_time.min(),
                                 g.compression_factor_time.max(), 2)
                      + r"$\times$",
                      "--", "--"])

    unit_row = " & ".join([r"\textbf{kJ/epoch}", r"\textbf{s/epoch}",
                           r"\textbf{util.\ (\%)}", r"\textbf{util.\ (\%)}"])
    where_row = " & ".join([r"224", r"224", r"224", r"$32{\times}32$"])
    panels = []
    for i, model in enumerate(models):
        if i:
            panels.append(r"\midrule")
        panels.append(r"\multicolumn{5}{@{}l}{\textbf{%s}} \\"
                      % MODEL_LABEL.get(model, model))
        panels.append(r"\midrule")
        for eco in ecosystems:
            panels.append(f"{SHORT.get(eco, eco)} & "
                          + " & ".join(cell(eco, model)) + r" \\")
        panels += [r"\cmidrule(l){1-5}", rows_224[model], rows_32[model],
                   rows_compress[model]]

    lines = [
        r"% generated by results/analysis/20_saturation.py -- do not edit",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{The accelerator-saturation cell (\S S7) beside the campaign "
        r"it contrasts. Training energy is the hardware counters over the "
        r"counter-bracketed window -- NVML board energy plus the RAPL CPU "
        r"package domains -- for one epoch: the median over the "
        r"\vSatReps{} independent runs of each cell, each run contributing its "
        r"own per-epoch mean. Utilisation is the 1\,Hz \texttt{nvidia-smi} "
        r"record over each run's training blocks (the counter-bracketed "
        r"interval of every training epoch, excluding start-up, evaluation "
        r"and tracker overhead), on both sides; the record covers all "
        r"\vSatUtilRuns{} of the \vSatUtilRunsOf{} runs at 224 and, having "
        r"started after the main campaign did, "
        r"\vSatRefUtilRunsMin--\vSatRefUtilRunsMax{} of the "
        r"\vSatRefUtilRunsPerPair{} runs of each ecosystem and architecture "
        r"at $32{\times}32$. The last row of each panel is the "
        r"ratio of the two spreads, $32{\times}32$ over 224: above $1\times$ "
        r"the ecosystem spread is narrower at $224{\times}224$, batch 32, on "
        r"Imagenette than at $32{\times}32$. In training it narrows within "
        r"the LibTorch lineage "
        r"(\vSatLibtorchCompressionTrainMin--\vSatLibtorchCompressionTrainMax"
        r"$\times$) and widens across all seven ecosystems "
        r"(\vSatCompressionTrainMin--\vSatCompressionTrainMax$\times$). The "
        r"cell changes resolution, batch size and dataset together, so neither "
        r"ratio is attributed to any one of them.}",
        r"\label{tab:saturation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Ecosystem} & " + unit_row + r" \\",
        r" & " + where_row + r" \\",
    ]
    lines += panels
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [
        r"\par\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"In the energy and time columns a spread is the ratio of the dearest "
        r"ecosystem to the cheapest, over the seven ecosystems of one "
        r"architecture in the training phase, each ecosystem taken at its mean "
        r"over runs as in Table~\ref{tab:spread} (the cells above are "
        r"medians), and the $32{\times}32$ row gives "
        r"the range of that ratio over the three datasets of the main "
        r"campaign; in the utilisation columns, which are percentages, it is "
        r"the range across the same seven. Compression is the "
        r"$32{\times}32$ spread over the 224 spread. A cell reading `--' is a "
        r"quantity that row does not define, or a cell no run of which the "
        r"utilisation record covers.",
        r"\end{minipage}",
        r"\end{table*}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-partial", action="store_true",
                    help="analyse a campaign short of its %d runs; every output "
                         "name gains a _partial suffix"
                         % EXPECTED_SATURATION_RUNS)
    args = ap.parse_args()

    print("=" * 78)
    print("ACCELERATOR-SATURATION CELL  (spec S7)")
    print("=" * 78)
    announce_scope("20_saturation")
    print(f"[20_saturation] 32x32 reference read from "
          f"{V2_TABLES.relative_to(REPO_ROOT)}/v2_*.csv -- read only.")

    if not SATURATION_DIR.is_dir():
        print(f"\nno saturation campaign at "
              f"{SATURATION_DIR}; nothing to analyse.")
        return 1

    done, want = campaign_status()
    if done < want:
        message = (f"the saturation cell has {done} complete run(s) of {want} "
                   f"({EXPECTED_EPOCHS} epochs in both phases required)")
        if not args.allow_partial:
            print(f"\nREFUSED: {message}.\n"
                  "A spread computed over whichever stacks finished first is a "
                  "spread over the fastest stacks. Re-run when the cell is "
                  "complete, or pass --allow-partial to write the numbers under "
                  "_partial names.")
            return 2
        print(f"\n!! PARTIAL: {message}; every output name gains _partial.")
    suffix = "_partial" if done < want else ""

    blocks = saturation_blocks()
    if blocks.empty:
        print("\nno complete measurement blocks in the saturation campaign.")
        return 1
    print(f"\ncollected {len(blocks)} measurement blocks from "
          f"{blocks.run_dir.nunique()} runs")

    # The run directory is in the key as well as the repetition index: the
    # guard in saturation_blocks has already refused a campaign where the two
    # disagree, and carrying both means a future one cannot silently sum two
    # directories into one run.
    runs_224 = per_run(blocks, ["ecosystem", "model", "dataset", "repetition",
                                "run_dir"])
    cells_224 = per_cell(runs_224)
    cells_32 = per_cell(v2_per_run())
    accuracy = final_accuracy(blocks)
    util_runs, uncovered, n_windows = saturation_utilisation()
    print(f"utilisation record covers the training blocks of "
          f"{len(util_runs)} of {n_windows} complete run(s)")
    if uncovered:
        print(f"  not covered ({len(uncovered)}): {', '.join(uncovered[:6])}"
              + (" ..." if len(uncovered) > 6 else ""))

    by_eco = energy_by_ecosystem(cells_224, accuracy, util_runs, done)
    print("\n--- per-epoch energy, time, accuracy and utilisation at 224 ---")
    print(by_eco[["ecosystem", "model", "n_runs",
                  "train_energy_total_J_median", "train_duration_s_median",
                  "final_test_acc_pct_median", "util_mean_pct",
                  "util_coverage"]].round(2).to_string(index=False))
    save_table(by_eco.round(4), f"sat_energy_by_ecosystem{suffix}",
               "The saturation cell per ecosystem and architecture: per-epoch "
               "counter energy (GPU + CPU package) and time, median and IQR "
               "over the runs, final accuracy, and accelerator utilisation")

    spread = spread_table(spread_means(blocks), v2_spread_means())
    check_control_against_tab_control(spread)
    print("\n--- ecosystem spread at 224 against 32x32 ---")
    print(spread[spread.subset == "all seven"][
        ["model", "phase", "reference_dataset", "spread_224", "spread_32",
         "compression_factor", "best_224", "worst_224"]].round(3)
        .to_string(index=False))
    save_table(spread.round(4), f"sat_spread{suffix}",
               "Max/min ecosystem spread in per-epoch energy at 224 against "
               "the same spread on each 32x32 dataset, with the ratio of the "
               "two; all seven ecosystems and the LibTorch family alone. Both "
               "sides on tab_spread's definition: per-run mean over epochs, "
               "then the mean over the runs of a cell")

    ranks = rank_agreement(cells_224, cells_32)
    print("\n--- does 224 order the ecosystems the way 32x32 does? ---")
    print(ranks.to_string(index=False) if not ranks.empty
          else "  (not computable)")
    save_table(ranks, f"sat_rank_agreement{suffix}",
               "Spearman between the ecosystem ordering at 224 and at each "
               "32x32 dataset, with an exact permutation p and the ecosystems "
               "whose rank moves by two places or more")

    rho = time_energy(runs_224)
    print("\n--- rho(time, energy) at 224, over the (ecosystem, model) cells ---")
    print(rho.to_string(index=False) if not rho.empty
          else "  (not computable)")
    save_table(rho, f"sat_time_energy{suffix}",
               "Spearman and Pearson correlation of run time with run energy "
               "over the 224 cells, on the definition behind \\vRhoTrain, with "
               "the main campaign's value beside it")

    util = utilisation_contrast(by_eco, v2_utilisation())
    print("\n--- utilisation and board power, 32x32 against 224 ---")
    print(util[["ecosystem", "model", "util_mean_pct_32", "util_mean_pct_224",
                "util_increase_pp", "power_mean_w_32", "power_mean_w_224",
                "coverage_224"]].round(1).to_string(index=False))
    save_table(util.round(4), f"sat_utilisation_contrast{suffix}",
               "Accelerator utilisation and sampled board power per ecosystem "
               "and architecture at 32x32 and at 224, from the same 1 Hz "
               "record over the same window on both sides: the training "
               "blocks' counter-bracketed intervals, not the whole run")

    emit_macros(by_eco, spread, ranks, rho, util, done, blocks,
                final_accuracy_by_run(blocks), loader_threads(SATURATION_DIR))
    n = macros.write(
        PAPER_OUT / f"numbers_saturation{suffix}.tex",
        "generated by results/analysis/20_saturation.py -- do not edit",
        "the accelerator-saturation cell (spec S7), straight from "
        f"{tables_dir().relative_to(REPO_ROOT)}/")
    print(f"\n  wrote "
          f"{(PAPER_OUT / f'numbers_saturation{suffix}.tex').relative_to(REPO_ROOT)}"
          f" ({n} macros)")
    saturation_table(by_eco, spread, util,
                     PAPER_OUT / f"tab_saturation{suffix}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
