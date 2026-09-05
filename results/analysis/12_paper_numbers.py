#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emit every number the manuscript quotes as LaTeX, so none of them is typed by
hand.

Reviewer #3 (m7) asked for the exact provenance of the reported values. The
honest answer for the submitted version was that the tables were transcribed
from spreadsheets, which is also how kilowatt-hours came to be labelled as
Joules. This script closes that gap: it writes

  paper/generated/numbers.tex   -- \\newcommand macros, one per quoted value
  paper/generated/tab_*.tex     -- the result tables themselves

and ``paper.tex`` reads both. A number that changes in the data changes in the
manuscript on the next build, or the build fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAMPAIGN_DIR, REPO_ROOT, announce_scope,  # noqa: E402
                    campaign_is_partial, read_campaign_metrics,
                    TABLES_RESOLVER, tables_dir)

TABLES = TABLES_RESOLVER  # writes divert on a live campaign, reads fall back
# The manuscript's own inputs. Diverted alongside the tables while the campaign
# is incomplete, for the reason in common.tables_dir: a monitoring run must not
# be able to rewrite the numbers the paper quotes.
OUT = REPO_ROOT / "paper" / ("generated" if not campaign_is_partial()
                             else "generated_partial")
announce_scope("12_paper_numbers")
OUT.mkdir(parents=True, exist_ok=True)

# Short labels for the manuscript; the analysis uses the long ecosystem names.
SHORT = {
    "Python/PyTorch": "PyTorch",
    "Python/TensorFlow": "TensorFlow",
    "Python/JAX": "JAX",
    "Cpp/LibTorch": "C++",
    "C++/LibTorch": "C++",
    "Java/DL4J": "Java",
    "R/torch": "R",
    "Rust/tch": "Rust",
}
#: The superseded campaign. Read only by first_campaign_collapse_facts, which
#: says why; everything else in this file is the campaign the paper reports.
FIRST_CAMPAIGN = REPO_ROOT / "results" / "campaign_v2_first_campaign"

DATASET = {"fashionmnist": "Fashion-MNIST", "cifar100": "CIFAR-100",
           "tinyimagenet": "Tiny ImageNet"}
MODEL = {"resnet18": "ResNet-18", "vgg16": "VGG-16"}

macros: dict[str, str] = {}


def sibling(stem: str):
    """Import one of the numbered analysis scripts as a module.

    ``09_campaign_v2`` is not an importable identifier, which is presumably why
    the two frames below were built here from ``results/replication/`` instead
    of from the campaign. That package is an output of the pipeline, not an
    input: it held the FIRST campaign for a fortnight while this script quoted
    ten macros from it, and nothing said so. Reaching through importlib for the
    collection code that already exists is cheaper than a second copy of it
    drifting away from the completeness gate.
    """
    import importlib.util
    path = Path(__file__).resolve().parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: What a float that is not a number formats to, in every spelling pandas,
#: numpy and Python produce.
NOT_A_NUMBER = {"nan", "-nan", "inf", "-inf", "infinity", "-infinity", "none",
                "<na>", "nat"}


def macro(name: str, value) -> None:
    """Register one \\newcommand. Names must be letters only (TeX restriction).

    A macro is refused rather than typeset if it would carry a value that is
    not a number. ``num()`` is ``f"{x:.4f}"``, and ``f"{nan:.4f}"`` is the
    string ``"nan"``: with zero collapses in the campaign chi2_contingency
    raises on a zero expected frequency, 15_convergence records NaN, and
    paper.tex typeset "a chi^2 test on that table returns p = nan, which would
    license a claim that ecosystems differ in robustness". A paper cannot print
    nan, and a build that silently prints it is worse than one that fails: the
    caller must decide what the quantity is when it is undefined, or leave the
    macro out and let LaTeX fail on the undefined command.
    """
    assert name.isalpha(), f"macro name {name!r} must be letters only"
    s = str(value)
    if s.strip().lower() in NOT_A_NUMBER:
        raise ValueError(f"macro {name} would emit {s!r} into the manuscript; "
                         "give the quantity a defined value or do not emit it")
    macros[name] = s


def only_row(frame: pd.DataFrame, table: str, macro_name: str):
    """The row of a one-row table, or a refusal that says which macro is lost.

    An empty table is now what a script writes when it has nothing to report,
    so ``.iloc[0]`` on one is reachable. Pandas' IndexError names neither the
    table nor the quantity the manuscript wanted; this does both, which is the
    difference between a build that tells you the campaign has no collapses and
    one that tells you position 0 is out of bounds.
    """
    if frame.empty:
        raise ValueError(f"{table} has no rows, so \\{macro_name} and the "
                         f"macros beside it are not defined for this campaign")
    return frame.iloc[0]


def span(lo: float, hi: float, digits: int = 1, unit: str = "") -> str:
    """``lo--hi`` at one precision, or a single value when the ends round equal.

    The padding table printed C++'s inference blocks as "0.3--0\\,s" -- the low
    end at one decimal and the high end at none, so the range read as ending
    below where it starts. A range is one quantity: both ends carry the same
    precision, and if that precision cannot tell them apart it prints once
    rather than as a range from a number to itself.
    """
    a, b = f"{lo:.{digits}f}", f"{hi:.{digits}f}"
    return f"{a}{unit}" if a == b else f"{a}--{b}{unit}"


def num(x, digits: int = 0) -> str:
    """A number formatted for siunitx, without thousands separators.

    Non-finite input is left to format as "nan"/"inf" and rejected by
    ``macro()``, which is the one place that knows which macro is at fault.
    """
    return f"{x:.{digits}f}"


def sci_upper(x: float, digits: int = 2) -> str:
    """A LaTeX scientific literal guaranteed to be >= x, for quoting as a bound.

    ``f"{1.177e-5:.0e}"`` gives ``1e-05``, and the manuscript quoted that as
    ``p <= 1e-5`` for a maximum of 1.1770e-5 -- a bound its own data violate.
    Round-to-nearest is right for reporting a value and wrong for reporting a
    limit: a limit may only ever be loosened by rounding. The mantissa is
    therefore ceilinged, not rounded.

    The old expression also built the exponent with ``.replace("e-0", ...)``,
    which silently produces nonsense for any exponent past -9, because there is
    no leading zero left to match.
    """
    import math
    if not math.isfinite(x):
        # Zero is a legitimate bound; NaN is not a bound at all, and returning
        # "0" for one would quote the tightest limit possible from no data.
        raise ValueError(f"cannot bound {x!r}: not a finite number")
    if x <= 0:
        return "0"
    exp = math.floor(math.log10(x))
    scale = 10.0 ** (digits - 1)
    # 1e-12 absorbs the float error that would round 1.20 up to 1.3
    mant = math.ceil(x / 10.0 ** exp * scale - 1e-9) / scale
    if mant >= 10.0:
        mant, exp = mant / 10.0, exp + 1
    mant_s = f"{mant:.{digits - 1}f}".rstrip("0").rstrip(".")
    return f"{mant_s}\\times 10^{{{exp}}}"


# ---------------------------------------------------------------- design ----
def design_facts() -> pd.DataFrame:
    epochs = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    runs = epochs.groupby(["ecosystem", "model", "dataset", "repetition"]).ngroups
    macro("vRuns", runs)
    macro("vBlocks", len(epochs))
    macro("vEcosystems", epochs.ecosystem.nunique())
    macro("vRepetitions", epochs.repetition.nunique())
    macro("vConfigurations", epochs.groupby(["ecosystem", "model", "dataset"]).ngroups)
    total_j = epochs.hw_meas_j.sum()
    macro("vCampaignEnergyMJ", num(total_j / 1e6, 1))
    macro("vCampaignEnergyKWh", num(total_j / 3.6e6, 1))
    return epochs


def apparatus_facts() -> None:
    """How many conformance checks enforce the specification.

    The manuscript quoted 57 in three places and 56 in a fourth while the
    checker actually ran 63: the count drifted the moment checks were added,
    which is the failure mode this whole file exists to prevent. Ask the
    checker.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import check_consistency  # noqa: E402  -- path set immediately above

    results = check_consistency.run()
    macro("vConformanceChecks", len(results))
    # A skipped check has not passed. This counted passes as "everything that
    # did not fail", so a checker run in which three checks could not read their
    # input published 92 of 92 passing -- the checker's own EXPECTED_CHECKS note
    # says a check that stops running is indistinguishable from one that passed,
    # and this is where that becomes a number in the manuscript. Refuse the run
    # instead: the conformance figures may only come from a run in which every
    # check actually ran.
    skipped = [x for x in results if x.status == check_consistency.SKIP]
    if skipped:
        raise ValueError(
            "conformance checks did not run, so \\vConformancePassing cannot be "
            "derived: " + ", ".join(f"{x.check} ({x.detail})" for x in skipped))
    failing = [x for x in results if x.status == check_consistency.FAIL]
    macro("vConformanceFailing", len(failing))
    macro("vConformancePassing",
          sum(1 for x in results if x.status == check_consistency.PASS))
    if failing:
        macro("vConformanceFailName", failing[0].check.replace("_", r"\_"))
        macro("vConformanceFailDetail", failing[0].detail.split(" missing")[0])
    catalogue_facts()


def catalogue_facts() -> None:
    """How many defect classes the catalogue lists, and how many are ours.

    The prose said five entries were ours while four carried the mark, because
    a table row and a sentence about it are edited at different times. Count
    the rows. Reading the manuscript to generate a number the manuscript then
    quotes is circular only in appearance: the table is the source, the
    sentence is derived from it, and this makes that direction explicit.
    """
    tex = (REPO_ROOT / "paper" / "paper.tex").read_text()

    # Every table* float whose caption mentions the catalogue, not just the one
    # carrying \label{tab:catalogue}. The catalogue outgrew a single float and
    # is being split across two, and a parser anchored on the label would have
    # counted the first half and reported it as the total -- silently, and in a
    # macro whose whole purpose is to stop a table and a sentence about it
    # drifting apart.
    floats = []
    for start in _positions(tex, r"\begin{table*}"):
        end = tex.index(r"\end{table*}", start)
        block = tex[start:end]
        caption = block[block.index(r"\caption{"):] if r"\caption{" in block else ""
        if "catalogue" in caption[:400].lower() or r"\label{tab:catalogue}" in block:
            floats.append(block)
    assert floats, "no catalogue float found in paper.tex"

    total = ours = 0
    for block in floats:
        for body_start in _positions(block, r"\begin{tabularx}"):
            body = block[body_start:block.index(r"\end{tabularx}", body_start)]
            for row in body.split("\\\\\n"):
                lines = [ln.strip() for ln in row.strip().split("\n") if ln.strip()]
                lines = [ln for ln in lines
                         if not ln.startswith((r"\toprule", r"\midrule",
                                               r"\addlinespace", r"\bottomrule",
                                               r"\begin{tabularx}"))]
                if not lines or lines[0].startswith((r"\multicolumn", r"\textbf")):
                    continue
                total += 1
                ours += r"$^\ast$" in row
    assert total > 10 and 0 < ours < total, f"catalogue parse looks wrong: {total}/{ours}"
    print(f"  catalogue: {total} rows ({ours} ours) across {len(floats)} float(s)")
    macro("vDefectCount", total)
    macro("vDefectOurs", ours)


def _positions(text: str, needle: str) -> list[int]:
    out, i = [], text.find(needle)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + 1)
    return out


# ------------------------------------------------------------ instrument ----
def instrument_facts(epochs: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLES / "v2_instrument_summary.csv")
    by_key = dict(zip(summary.quantity, summary.mean_ratio_or_share))
    gpu = [v for k, v in by_key.items() if k.startswith("GPU,")][0]
    cpu = [v for k, v in by_key.items() if k.startswith("CPU package,")][0]
    meas = [v for k, v in by_key.items() if k.startswith("GPU + CPU")][0]
    total = [v for k, v in by_key.items() if k.startswith("CodeCarbon total")][0]
    ram = [v for k, v in by_key.items() if k.startswith("RAM share")][0]
    macro("vCCgpuRatio", num(gpu, 4))
    macro("vCCcpuRatio", num(cpu, 4))
    macro("vCCmeasRatio", num(meas, 4))
    macro("vCCmeasDisagreementPct", num(abs(meas - 1) * 100, 1))
    macro("vCCtotalRatio", num(total, 3))
    macro("vCCramSharePct", num(ram, 1))

    floor = pd.read_csv(TABLES / "v2_instrument_duration_floor.csv")
    fv = dict(zip(floor.quantity, floor.value))
    macro("vWindowFloorS", num(fv["floor on the reported window (s)"], 2))
    macro("vWindowExcessLongMs",
          num(fv["median window excess, phases longer than 6 s (s)"] * 1000, 0))

    # What the disagreement actually is. The GPU terms are the same register
    # read twice; the CPU terms differ by a near-constant offset that comes from
    # nesting the counter reads inside the tracker's window, not from any
    # sampling regime. Reporting it as a percentage made a fixed offset look
    # like a length-dependent accuracy problem.
    # What the reported boundary contains, and how much signal each half carries.
    cpu_w = epochs.hw_cpu_j / epochs.duration_hw_s
    gpu_w = epochs.hw_gpu_j / epochs.duration_hw_s
    macro("vCpuPowerMeanW", num(cpu_w.mean(), 0))
    macro("vCpuPowerSdW", num(cpu_w.std(), 1))
    macro("vCpuPowerCVPct", num(100 * cpu_w.std() / cpu_w.mean(), 0))
    macro("vGpuPowerCVPct", num(100 * gpu_w.std() / gpu_w.mean(), 0))
    macro("vCpuShareOfTotalPct",
          num(100 * epochs.hw_cpu_j.sum() / epochs.hw_total_j.sum(), 0))

    # The machine's static draw at the same boundary, measured on an idle host
    # (scripts/measure_idle.py). Every figure in this study is whole-machine and
    # un-baselined, so how much of a block is simply the machine being on is a
    # question the study owes an answer to.
    idle_path = TABLES / "v2_idle_baseline.json"
    if idle_path.exists():
        idle = json.loads(idle_path.read_text())
        macro("vIdleTotalW", num(idle["total_w"], 0))
        macro("vIdleGpuW", num(idle["gpu_w"], 0))
        macro("vIdleCpuW", num(idle["cpu_package_w"], 0))
        macro("vIdleCpuSharePct", num(100 * idle["cpu_package_w"] / cpu_w.mean(), 0))
        macro("vIdleGpuSharePct", num(100 * idle["gpu_w"] / gpu_w.mean(), 0))
        block_w = (epochs.hw_total_j / epochs.duration_hw_s)
        macro("vIdleShareOfBlockPct", num(100 * idle["total_w"] / block_w.mean(), 0))

    # The paper's own CPU-fallback signature, applied to the paper's own
    # campaign. The lowest mean GPU power is named in the defect catalogue as
    # the signature of a stack that silently ran on the host; one stack here
    # trips it, and the CPU term is what clears it.
    pw = epochs.assign(gpu_w=epochs.hw_gpu_j / epochs.duration_hw_s,
                       cpu_w=epochs.hw_cpu_j / epochs.duration_hw_s)
    by_eco = pw.groupby("ecosystem")[["gpu_w", "cpu_w"]].mean()
    lowest = by_eco.gpu_w.idxmin()
    macro("vLowestGpuEco", SHORT[lowest])
    macro("vLowestGpuW", num(by_eco.gpu_w.min(), 0))
    macro("vHighestGpuW", num(by_eco.gpu_w.max(), 0))
    macro("vLowestGpuEcoCpuW", num(by_eco.loc[lowest, "cpu_w"], 0))
    macro("vCpuPowerAcrossMinW", num(by_eco.cpu_w.min(), 0))
    macro("vCpuPowerAcrossMaxW", num(by_eco.cpu_w.max(), 0))

    diff = (epochs.cc_meas_j - epochs.hw_meas_j).abs()
    macro("vOffsetMedianJ", num(diff.median(), 2))
    macro("vOffsetGpuMedianMJ", num((epochs.cc_gpu_j - epochs.hw_gpu_j).abs().median() * 1000, 1))
    macro("vOffsetCpuMedianJ", num((epochs.cc_cpu_j - epochs.hw_cpu_j).abs().median(), 2))
    macro("vOffsetDurationCorr", num(diff.corr(epochs.duration_hw_s), 2))
    macro("vCCmeasWeightedRatio", num(epochs.cc_meas_j.sum() / epochs.hw_meas_j.sum(), 5))
    macro("vCCmeasWeightedPct",
          num(abs(epochs.cc_meas_j.sum() / epochs.hw_meas_j.sum() - 1) * 100, 2))
    macro("vCCtotalWeightedRatio", num(epochs.cc_total_j.sum() / epochs.hw_total_j.sum(), 3))

    abl = pd.read_csv(TABLES / "v2_instrument_agreement_by_length.csv").set_index("phase_length")
    macro("vAgreeShortPct", num(abl.loc["<1 s", "mean_error_pct"], 1))
    macro("vAgreeShortWorstPct", num(abl.loc["<1 s", "worst_error_pct"], 0))
    macro("vAgreeLongPct", num(abl.loc[">30 s", "mean_error_pct"], 2))
    macro("vSamplingIntervalS", "1")
    cov = pd.read_csv(TABLES / "v2_coverage_by_ecosystem.csv")
    macro("vCoverageMin", num(cov.coverage_pct.min(), 0))
    macro("vCoverageMax", num(cov.coverage_pct.max(), 0))
    macro("vCoverageWorstEco", SHORT[cov.loc[cov.coverage_pct.idxmin(), "ecosystem"]])
    macro("vCoverageBestEco", SHORT[cov.loc[cov.coverage_pct.idxmax(), "ecosystem"]])

    # What the untracked time costs, at the measured idle power. The paper says
    # this energy is "attributed to nobody"; it should say how much.
    per_block = pd.read_csv(TABLES / "v2_coverage_per_block.csv", usecols=["gap_s"])
    untracked_s = float(per_block.gap_s.sum())
    macro("vUntrackedHours", num(untracked_s / 3600, 1))
    if idle_path.exists():
        untracked_j = untracked_s * idle["total_w"]
        macro("vUntrackedMJ", num(untracked_j / 1e6, 1))
        macro("vUntrackedPct", num(100 * untracked_j / epochs.hw_meas_j.sum(), 0))

    gi = pd.read_csv(TABLES / "v2_coverage_gap_attribution.csv").iloc[0]
    macro("vGapInstrumentR", num(gi.pearson_r, 2))
    macro("vGapInstrumentRPadded", num(gi.pearson_r_padded_only, 2))
    macro("vGapInstrumentRUnpadded", num(gi.pearson_r_unpadded_only, 2))
    macro("vGapTimestampResolutionS", int(gi.timestamp_resolution_s))
    macro("vGapNegativePct", num(gi.negative_gap_pct, 0))
    macro("vGapInstrumentPct", num(gi.share_of_gap_explained_pct, 0))
    macro("vGapMedianS", num(gi.median_gap_s, 2))
    macro("vGapVsExcessS", num(gi.median_abs_difference_s, 2))

    # The reported-duration excess: three modes, not a function of block length.
    wm = pd.read_csv(TABLES / "v2_coverage_window_model.csv").sort_values("mode_excess_s")
    macro("vWindowModes", len(wm))
    for i, (_, r) in enumerate(wm.iterrows(), start=1):
        tag = ["One", "Two", "Three", "Four"][i - 1]
        macro(f"vWindowMode{tag}S", num(r.mode_excess_s, 2))
        macro(f"vWindowMode{tag}Pct", num(r.share_pct, 0))
        macro(f"vWindowMode{tag}Blocks", int(r.n_blocks))
    macro("vWindowModeWidestS", num(wm.spread_s.max(), 2))
    macro("vWindowPaddedPct", num(wm[wm.mode_excess_s > 0.5].share_pct.sum(), 0))
    macro("vWindowConstS", num(wm.mode_excess_s.iloc[1], 2))

    rf = pd.read_csv(TABLES / "v2_coverage_window_rejected_fits.csv")
    naive = rf[rf.model.str.startswith("max(")].iloc[0]
    thresh = rf[~rf.model.str.startswith("max(")].iloc[0]
    macro("vWindowModelNaiveRsq", num(naive.r_squared, 4))
    macro("vWindowModelNaiveMAE", num(naive.mean_abs_error_s, 2))
    macro("vWindowModelBestRsq", num(thresh.r_squared, 4))
    macro("vWindowModelBestMAE", num(thresh.mean_abs_error_s, 2))
    macro("vWindowThresholdS", re.findall(r"[0-9.]+", thresh.model)[1])
    macro("vWindowPaddedAboveThreshold", int(naive.padded_above_threshold))
    macro("vWindowUnpaddedBelowThreshold", int(naive.unpadded_below_threshold))

    # The mechanism, from the probe. Three modes = how many of the tracker's two
    # blocking network lookups are still outstanding when stop() runs.
    mech_path = TABLES / "v2_window_mechanism.csv"
    if mech_path.exists():
        mech = pd.read_csv(mech_path)
        macro("vMechBothS", num(mech[mech.lookups_outstanding == 2].stop_cost_s.median(), 2))
        macro("vMechOneS", num(mech[mech.lookups_outstanding == 1].stop_cost_s.median(), 2))
        macro("vMechNoneS", num(mech[mech.lookups_outstanding == 0].stop_cost_s.median(), 3))
        macro("vMechThresholdS", num(
            mech[mech.lookups_outstanding == 0].work_s.min(), 0))
        macro("vMechApiCallInterval", 8)
        lines = [
            r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
            r"\begin{tabular}{rrl}",
            r"\toprule",
            r"\textbf{Block held open} & \textbf{Cost of \texttt{stop()}} "
            r"& \textbf{Lookups outstanding} \\",
            r"\midrule",
        ]
        for _, r in mech.iterrows():
            lines.append(f"{r.work_s:.0f}\\,s & {r.stop_cost_s:.3f}\\,s & "
                         f"{int(r.lookups_outstanding)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "tab_mechanism.tex").write_text("\n".join(lines) + "\n")
        print(f"  wrote {(OUT / 'tab_mechanism.tex').relative_to(REPO_ROOT)}")

    # A single length threshold, and how well it does.
    blk_thr = pd.read_csv(TABLES / "v2_instrument_epochs.csv",
                          usecols=["ecosystem", "phase", "duration_hw_s",
                                   "duration_cc_s"])
    blk_thr["padded"] = (blk_thr.duration_cc_s - blk_thr.duration_hw_s) > 0.5
    pred = blk_thr.duration_hw_s < 11.0
    macro("vThresholdAccuracyPct", num(100 * (pred == blk_thr.padded).mean(), 1))
    fn = blk_thr[~pred & blk_thr.padded]
    macro("vThresholdMisses", len(fn))
    macro("vThresholdMissEco", SHORT[fn.ecosystem.mode().iat[0]] if len(fn) else "none")
    macro("vThresholdFalsePos", int((pred & ~blk_thr.padded).sum()))
    short = blk_thr[blk_thr.duration_hw_s < 9]
    macro("vShortBlocksPaddedPct", num(100 * short.padded.mean(), 0))
    macro("vShortBlocks", len(short))

    # Where the padding falls, by stack and phase. The manuscript claimed block
    # length does not decide the mode and offered R as the counterexample; R's
    # shortest block is longer than the threshold, so R is consistent with the
    # length model and proves nothing. This table is the honest picture: length
    # is the dominant factor, and there is a stack-specific offset on top of it.
    blk_all = pd.read_csv(TABLES / "v2_instrument_epochs.csv",
                          usecols=["ecosystem", "phase", "duration_hw_s",
                                   "duration_cc_s"])
    blk_all["padded"] = (blk_all.duration_cc_s - blk_all.duration_hw_s) > 0.5
    piv = (100 * blk_all.pivot_table(index="ecosystem", columns="phase",
                                     values="padded", aggfunc="mean"))
    # Per phase: a range over both phases beside a per-phase rate reads as a
    # counterexample to the threshold when it is nothing of the kind.
    length = blk_all.groupby(["ecosystem", "phase"]).duration_hw_s.agg(["min", "max"])
    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Training}} & \multicolumn{2}{c}{\textbf{Inference}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Ecosystem} & \textbf{block length} & \textbf{padded} "
        r"& \textbf{block length} & \textbf{padded} \\",
        r"\midrule",
    ]
    for eco in piv.sort_values("Training").index:
        cells = []
        for ph in ("Training", "Inference"):
            lo, hi = length.loc[(eco, ph), "min"], length.loc[(eco, ph), "max"]
            cells.append(span(lo, hi, 1, r"\,s"))
            cells.append(f"{piv.loc[eco, ph]:.0f}\\%")
        lines.append(f"{SHORT[eco]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_padding.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_padding.tex').relative_to(REPO_ROOT)}")
    infer_padded = blk_all[blk_all.phase == "Inference"]
    non_r = infer_padded[infer_padded.ecosystem != "R/torch"]
    macro("vPaddedInferPct", num(100 * non_r.padded.mean(), 0))
    macro("vPaddedShortestUnpaddedS", num(
        blk_all[~blk_all.padded].duration_hw_s.min(), 1))
    macro("vPaddedLongestPaddedS", num(
        blk_all[blk_all.padded].duration_hw_s.max(), 1))

    # The wide excess mode is two levels and the hour of day decides which one a
    # block gets: the estimator's reported duration carries the round-trip time
    # of a geolocation lookup, so it is worse when the network is busy. A
    # measured, reproducible source of variation in a number the tool reports as
    # a duration.
    macro("vLongestBlockS", num(blk_all.duration_hw_s.max(), 1))

    # Ecosystems where the instrument's own window accounts for more than all of
    # the untracked time -- the excess cannot be attributed to anything else,
    # and a share above 100% is the estimator overlapping its own gap.
    cov = pd.read_csv(TABLES / "v2_coverage_by_ecosystem.csv")
    macro("vCoverageOverAttributedEcos",
          int((cov.untracked_that_is_instrument_pct > 100).sum()))

    dl = pd.read_csv(TABLES / "v2_coverage_window_diurnal.csv")
    if not dl.empty:
        night = dl[dl.window == "night"].iloc[0]
        day = dl[dl.window == "day"].iloc[0]
        macro("vWindowNightMedianS", num(night.median_excess_s, 2))
        macro("vWindowDayMedianS", num(day.median_excess_s, 2))
        macro("vWindowNightHours", night.hours)
        macro("vWindowNightShare", num(night.share_in_level_pct, 0))
        macro("vWindowDayShare", num(day.share_in_level_pct, 0))

    # The two ecosystems that rule out any length-based model, named by the data.
    blk = pd.read_csv(TABLES / "v2_coverage_per_block.csv",
                      usecols=["ecosystem", "window_excess_s"])
    padded_share = blk.assign(p=blk.window_excess_s > 0.5).groupby("ecosystem").p.mean()
    # "The ecosystem that is never padded" is a claim about a set, and idxmin
    # returns one name whether the set has one member, none, or three. Name it
    # only when it is one; otherwise give the list, and let the sentence quoting
    # a singular macro fail rather than silently name an arbitrary member.
    unpadded = sorted(padded_share[padded_share == 0].index)
    always = padded_share.idxmax()
    if len(unpadded) == 1:
        never = unpadded[0]
        macro("vWindowUnpaddedEco", SHORT[never])
        macro("vWindowUnpaddedEcoBlocks", int((blk.ecosystem == never).sum()))
    else:
        macro("vWindowUnpaddedEcos",
              ", ".join(SHORT[e] for e in unpadded) if unpadded else "none")
    macro("vWindowAlwaysPaddedEco", SHORT[always])
    macro("vWindowAlwaysPaddedEcoBlocks", int((blk.ecosystem == always).sum()))

    # The consequence, stated without a model.
    pdist = pd.read_csv(TABLES / "v2_coverage_power_distortion.csv")
    worst = pdist.loc[pdist.understated_by.idxmax()]
    macro("vPowerUnderstatedWorst", num(worst.understated_by, 1))
    macro("vPowerUnderstatedWorstBin", str(worst.bin).replace("<", "$<$"))
    macro("vPowerReportedWorstW", num(worst.reported_power_w, 0))
    macro("vPowerMeasuredWorstW", num(worst.measured_power_w, 0))
    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        # tabularx rather than tabular: the three right-hand headers are what
        # made this table half again as wide as a column, and \resizebox then
        # set it at 5.3 pt. As X columns the headers wrap and the body stays at
        # the surrounding \footnotesize.
        r"\begin{tabularx}{\linewidth}{@{}lrRRR@{}}",
        r"\toprule",
        r"\textbf{Phase length} & \textbf{Blocks} & "
        r"\textbf{Power from the reported fields} & \textbf{Measured power} & "
        r"\textbf{Understated by} \\",
        r"\midrule",
    ]
    for _, r in pdist.iterrows():
        label = str(r["bin"]).replace("<", "$<$").replace(">", "$>$").replace("-", "--")
        lines.append(f"{label} & {int(r.n_blocks)} & {r.reported_power_w:.0f}\\,W & "
                     f"{r.measured_power_w:.0f}\\,W & {r.understated_by:.2f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (OUT / "tab_power_distortion.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_power_distortion.tex').relative_to(REPO_ROOT)}")

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabularx}{\linewidth}{@{}lrrR@{}}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Median block} & \textbf{Tracked share} "
        r"& \textbf{Of the untracked rest, the instrument} \\",
        r"\midrule",
    ]
    for _, row in cov.sort_values("coverage_pct").iterrows():
        # Unclamped: the manuscript says this column prints the above-100%
        # cell rather than rounding it away, and it has to be true.
        share = row.untracked_that_is_instrument_pct
        lines.append(
            f"{SHORT[row.ecosystem]} & {row.median_block_s:.1f}\\,s & "
            f"{row.coverage_pct:.0f}\\% & {share:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (OUT / "tab_coverage.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_coverage.tex').relative_to(REPO_ROOT)}")

    dist = pd.read_csv(TABLES / "v2_instrument_power_distortion.csv")
    sub = dist.set_index("phase_length")
    macro("vPowerUnderstatedSubSecond", num(sub.loc["<1 s", "understated_by"], 1))
    macro("vPowerCountersSubSecond", num(sub.loc["<1 s", "power_counters_w"], 0))
    macro("vPowerCodeCarbonSubSecond", num(sub.loc["<1 s", "power_codecarbon_w"], 0))
    macro("vPowerAgreeLongPct",
          num(abs(sub.loc["10-30 s", "understated_by"] - 1) * 100, 0))

    # the shortest phase in the campaign, i.e. the worst case for the floor
    worst = epochs.loc[epochs.duration_hw_s.idxmin()]
    macro("vShortestPhaseS", num(worst.duration_hw_s, 2))
    macro("vShortestPhaseReportedS", num(worst.duration_cc_s, 2))
    macro("vShortestPhaseFactor", num(worst.duration_cc_s / worst.duration_hw_s, 0))


# ----------------------------------------------------------- repeatability --
def repeatability_facts() -> pd.DataFrame:
    stats = pd.read_csv(TABLES / "v2_between_run_statistics.csv")
    for phase, tag in (("Training", "Train"), ("Inference", "Infer")):
        g = stats[stats.phase == phase]
        macro(f"vCV{tag}Median", num(g.cv_pct.median(), 2))
        macro(f"vCV{tag}Max", num(g.cv_pct.max(), 2))
    return stats


# ----------------------------------------------------------------- energy --
def energy_facts(stats: pd.DataFrame) -> None:
    """Per-block spread between the cheapest and most expensive ecosystem."""
    train = stats[stats.phase == "Training"]
    ratios = []
    for (model, dataset), g in train.groupby(["model", "dataset"]):
        g = g.sort_values("mean_energy_J")
        lo, hi = g.iloc[0], g.iloc[-1]
        ratios.append(
            {
                "model": MODEL[model],
                "dataset": DATASET[dataset],
                "n": len(g),
                "best": SHORT[lo.ecosystem],
                "best_j": lo.mean_energy_J,
                "worst": SHORT[hi.ecosystem],
                "worst_j": hi.mean_energy_J,
                "ratio": hi.mean_energy_J / lo.mean_energy_J,
            }
        )
    spread = pd.DataFrame(ratios)
    macro("vSpreadTrainMin", num(spread.ratio.min(), 1))
    macro("vSpreadTrainMax", num(spread.ratio.max(), 1))
    macro("vSpreadTrainBest", spread.best.mode().iat[0])
    macro("vSpreadTrainWorstCount", int((spread.worst == spread.worst.mode().iat[0]).sum()))

    infer = stats[stats.phase == "Inference"]
    iratios = []
    for (model, dataset), g in infer.groupby(["model", "dataset"]):
        g = g.sort_values("mean_energy_J")
        iratios.append(g.iloc[-1].mean_energy_J / g.iloc[0].mean_energy_J)
    macro("vSpreadInferMin", num(min(iratios), 1))
    macro("vSpreadInferMax", num(max(iratios), 1))

    # Table: training energy per epoch, every ecosystem x block, with CI
    write_energy_table(stats, "Training", "tab_train_energy")
    write_energy_table(stats, "Inference", "tab_infer_energy")
    write_spread_table(spread)


def write_energy_table(stats: pd.DataFrame, phase: str, stem: str) -> None:
    """Six blocks across the page, as a two-level header.

    Written flat --- one "ResNet-18 / Fashion-MNIST" per column --- the header
    alone made the table 276\\,pt wider than the text block, and it ran off the
    page. The model spans its three datasets instead, and the dataset names are
    abbreviated in the second row.
    """
    short_dataset = {"Fashion-MNIST": "F-MNIST", "CIFAR-100": "CIFAR-100",
                     "Tiny ImageNet": "Tiny-IN"}
    g = stats[stats.phase == phase].copy()
    g["Ecosystem"] = g.ecosystem.map(SHORT)
    g["model_label"] = g.model.map(MODEL)
    g["dataset_label"] = g.dataset.map(DATASET)
    pivot = g.pivot_table(index="Ecosystem", columns=["model_label", "dataset_label"],
                          values="mean_energy_J")
    pivot = pivot.sort_values(pivot.columns[0])

    models = list(dict.fromkeys(m for m, _ in pivot.columns))
    spans, rule, cursor = [], [], 2
    for model in models:
        width = sum(1 for m, _ in pivot.columns if m == model)
        spans.append(f"\\multicolumn{{{width}}}{{c}}{{\\textbf{{{model}}}}}")
        rule.append(f"\\cmidrule(lr){{{cursor}-{cursor + width - 1}}}")
        cursor += width

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{l" + "r" * len(pivot.columns) + "}",
        r"\toprule",
        r"\textbf{Ecosystem} & " + " & ".join(spans) + r" \\",
        "".join(rule),
        " & " + " & ".join(short_dataset.get(d, d) for _, d in pivot.columns) + r" \\",
        r"\midrule",
    ]
    for eco, row in pivot.iterrows():
        cells = [f"{v:,.0f}".replace(",", "\\,") if pd.notna(v) else "--" for v in row]
        lines.append(f"{eco} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{stem}.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / f'{stem}.tex').relative_to(REPO_ROOT)}")


def write_spread_table(spread: pd.DataFrame) -> None:
    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{llrlrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Dataset} & \multicolumn{2}{c}{\textbf{Lowest}} "
        r"& \multicolumn{2}{c}{\textbf{Highest}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & {J} & & {J} & {Ratio} \\",
        r"\midrule",
    ]
    for _, r in spread.iterrows():
        cells = [r.model, r.dataset, r.best, f"{r.best_j:,.0f}", r.worst,
                 f"{r.worst_j:,.0f} ({r.ratio:.1f}$\\times$)"]
        lines.append(" & ".join(c.replace(",", "\\,") for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_spread.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_spread.tex').relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------- quality --
# A run counts as collapsed if it never exceeded 1.5x chance; see
# 15_convergence.py, which owns the definition.
CHANCE_PCT = {"fashionmnist": 10.0, "cifar100": 1.0, "tinyimagenet": 0.5}
COLLAPSE_FACTOR = 1.5


def quality_facts() -> None:
    q = pd.read_csv(TABLES / "v2_quality_normalised.csv")
    # The accuracy table averaged the collapsed runs in, so the manuscript used
    # a collapse-contaminated spread as evidence in one section and argued in
    # another that collapses explain nearly all of it. Both versions now exist,
    # and each claim says which it uses.
    q["collapsed"] = q.final_test_acc_pct <= q.dataset.map(CHANCE_PCT) * COLLAPSE_FACTOR
    macro("vCollapseFactor", num(COLLAPSE_FACTOR, 1))
    trained = q[~q.collapsed]

    summ = q.groupby(["ecosystem", "dataset"]).final_test_acc_pct.mean().reset_index()
    summ_ok = (trained.groupby(["ecosystem", "dataset"])
               .final_test_acc_pct.mean().reset_index())
    for ds, tag in (("fashionmnist", "Fashion"), ("cifar100", "Cifar"),
                    ("tinyimagenet", "Tiny")):
        g = summ[summ.dataset == ds]
        if g.empty:
            continue
        macro(f"vAccMin{tag}", num(g.final_test_acc_pct.min(), 1))
        macro(f"vAccMax{tag}", num(g.final_test_acc_pct.max(), 1))
        macro(f"vAccSpread{tag}",
              num(g.final_test_acc_pct.max() - g.final_test_acc_pct.min(), 1))
        h = summ_ok[summ_ok.dataset == ds]
        macro(f"vAccTrainedMin{tag}", num(h.final_test_acc_pct.min(), 1))
        macro(f"vAccTrainedMax{tag}", num(h.final_test_acc_pct.max(), 1))
        macro(f"vAccTrainedSpread{tag}",
              num(h.final_test_acc_pct.max() - h.final_test_acc_pct.min(), 1))

    # Pooling the two architectures cancels most of the spread, because they
    # reach different accuracies in opposite per-ecosystem directions. A claim
    # about how closely the stacks agree has to be made per block.
    per_block = (q.groupby(["model", "dataset", "ecosystem"])
                 .final_test_acc_pct.mean().reset_index())
    blk = (per_block.groupby(["model", "dataset"]).final_test_acc_pct
           .agg(lambda x: x.max() - x.min()).reset_index(name="spread"))
    fashion = blk[blk.dataset == "fashionmnist"]
    macro("vAccBlockSpreadFashionMax", num(fashion.spread.max(), 1))
    ok_block = (trained.groupby(["model", "dataset", "ecosystem"])
                .final_test_acc_pct.mean().reset_index())
    ok_blk = (ok_block.groupby(["model", "dataset"]).final_test_acc_pct
              .agg(lambda x: x.max() - x.min()).reset_index(name="spread"))
    macro("vAccBlockSpreadTrainedMax", num(ok_blk.spread.max(), 1))
    macro("vAccRunsPerCell", int(q.groupby(["ecosystem", "dataset"]).size().max()))

    # accuracy per kJ: the quality-normalised ranking the fixed-budget one hides.
    # Pooling across datasets averages values two orders of magnitude apart
    # (~2.8 on Fashion-MNIST against ~0.03 on Tiny ImageNet), so the pooled mean
    # is dominated by the easiest dataset. Report the per-dataset ratios too.
    for ds, tag in (("fashionmnist", "Fashion"), ("cifar100", "Cifar"),
                    ("tinyimagenet", "Tiny")):
        g = q[q.dataset == ds].groupby("ecosystem").acc_per_kJ.mean().sort_values()
        if len(g) > 1:
            macro(f"vEffRatio{tag}", num(g.iloc[-1] / g.iloc[0], 1))
            macro(f"vEffBest{tag}", SHORT[g.index[-1]])
    eff = q.groupby("ecosystem").acc_per_kJ.mean().sort_values(ascending=False)
    macro("vEffBest", SHORT[eff.index[0]])
    macro("vEffWorst", SHORT[eff.index[-1]])
    macro("vEffRatio", num(eff.iloc[0] / eff.iloc[-1], 1))
    eff_ok = trained.groupby("ecosystem").acc_per_kJ.mean().sort_values(ascending=False)
    macro("vEffRatioTrained", num(eff_ok.iloc[0] / eff_ok.iloc[-1], 1))

    # Energy to first reach a fixed target accuracy: the Methods call this the
    # closest thing to a practitioner's question, and the manuscript never
    # reported it. It is reported now, NaNs and all -- a stack that never
    # reaches the target is the most informative cell in the table.
    tgt = (q.groupby(["ecosystem", "dataset"])
           .agg(n=("energy_to_target_J", "size"),
                reached=("energy_to_target_J", "count"),
                median_kJ=("energy_to_target_J", "median"))
           .reset_index())
    tlines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Fashion-MNIST} & \textbf{CIFAR-100} "
        r"& \textbf{Tiny ImageNet} \\",
        r" & \emph{85\%} & \emph{30\%} & \emph{20\%} \\",
        r"\midrule",
    ]
    tp = tgt.pivot(index="ecosystem", columns="dataset")
    for eco in tp.index:
        cells = []
        for d in ("fashionmnist", "cifar100", "tinyimagenet"):
            reached = tp.loc[eco, ("reached", d)]
            n = tp.loc[eco, ("n", d)]
            med = tp.loc[eco, ("median_kJ", d)]
            if reached == 0:
                cells.append("never")
            elif reached < n:
                cells.append(f"{med / 1000:.1f} ({int(reached)}/{int(n)})")
            else:
                cells.append(f"{med / 1000:.1f}")
        tlines.append(f"{SHORT[eco]} & " + " & ".join(cells) + r" \\")
    tlines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_energy_to_target.tex").write_text("\n".join(tlines) + "\n")
    print(f"  wrote {(OUT / 'tab_energy_to_target.tex').relative_to(REPO_ROOT)}")
    macro("vTargetFashion", num(q[q.dataset == "fashionmnist"].target_acc_pct.iat[0], 0))
    macro("vTargetCifar", num(q[q.dataset == "cifar100"].target_acc_pct.iat[0], 0))
    macro("vTargetTiny", num(q[q.dataset == "tinyimagenet"].target_acc_pct.iat[0], 0))
    fash = tgt[tgt.dataset == "fashionmnist"]
    macro("vTargetFashionSpread",
          num(fash.median_kJ.max() / fash.median_kJ.min(), 1))
    macro("vTargetNeverStacks",
          int((tgt[tgt.dataset == "tinyimagenet"].reached == 0).sum()))

    # Per (architecture, dataset), not pooled. The manuscript's most quoted
    # accuracy figures -- \vAccBlockSpreadFashionMax and
    # \vAccBlockSpreadTrainedMax -- are per-block quantities, and the pooled
    # table that used to stand here could not be read for either of them: a
    # referee checking 0.5 against it found 0.3 and no way to get from one to
    # the other. The block spreads are printed as their own row, so both
    # numbers are read off the table that produces them.
    ARCH = {"resnet18": "R-18", "vgg16": "VGG-16"}
    models = [m for m in ("resnet18", "vgg16") if m in set(per_block.model)]
    datasets = [d for d in ("fashionmnist", "cifar100", "tinyimagenet")
                if d in set(per_block.dataset)]
    group, sub, rules = [], [], []
    for i, d in enumerate(datasets):
        span = len(models)
        group.append(r"\multicolumn{%d}{c}{\textbf{%s}}" % (span, DATASET[d]))
        sub.extend(ARCH[m] for m in models)
        first = 2 + i * span
        rules.append(r"\cmidrule(lr){%d-%d}" % (first, first + span - 1))
    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{@{}l" + ("r" * (len(models) * len(datasets))) + r"@{}}",
        r"\toprule",
        " & ".join([""] + group) + r" \\",
        "".join(rules),
        " & ".join([r"\textbf{Ecosystem}"] + sub) + r" \\",
        r"\midrule",
    ]
    blk_ok = ok_block.set_index(["model", "dataset", "ecosystem"]).final_test_acc_pct
    blk_all = per_block.set_index(["model", "dataset", "ecosystem"]).final_test_acc_pct
    for eco in summ.ecosystem.drop_duplicates():
        cells = []
        for d in datasets:
            for m in models:
                v = blk_all.get((m, d, eco), np.nan)
                w = blk_ok.get((m, d, eco), np.nan)
                if pd.isna(v):
                    cells.append("--")
                elif pd.isna(w) or abs(v - w) < 0.005:
                    cells.append(f"{v:.2f}")
                else:  # collapsed runs excluded, in parentheses
                    cells.append(f"{v:.2f} ({w:.2f})")
        lines.append(f"{SHORT[eco]} & " + " & ".join(cells) + r" \\")
    spread = blk.set_index(["model", "dataset"]).spread
    spread_ok = ok_blk.set_index(["model", "dataset"]).spread

    def spread_cell(m: str, d: str) -> str:
        # One decimal, because the manuscript quotes these two cells through
        # \vAccBlockSpreadFashionMax and \vAccBlockSpreadTrainedMax, which are
        # one-decimal macros. A two-decimal row here would print 0.52 beside a
        # sentence saying 0.5 and invite the reader to wonder which is wrong.
        v, w = spread.get((m, d), np.nan), spread_ok.get((m, d), np.nan)
        if pd.isna(v):
            return "--"
        if pd.isna(w) or abs(v - w) < 0.05:
            return f"{v:.1f}"
        return f"{v:.1f} ({w:.1f})"

    lines.append(r"\midrule")
    lines.append(r"\textbf{Spread} & " + " & ".join(
        spread_cell(m, d) for d in datasets for m in models) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_accuracy.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_accuracy.tex').relative_to(REPO_ROOT)}")


# ------------------------------------------------------------- statistics --
def statistics_facts() -> None:
    """Run-level tests, written by 14_v2_statistics.py."""
    et = pd.read_csv(TABLES / "v2_stats_energy_time.csv").set_index("phase")
    for phase, tag in (("Training", "Train"), ("Inference", "Infer")):
        if phase not in et.index:
            continue
        macro(f"vRho{tag}", num(et.loc[phase, "spearman_rho"], 2))
        macro(f"vDiscordant{tag}Pct", num(et.loc[phase, "discordant_pct"], 1))

    pw = pd.read_csv(TABLES / "v2_stats_pairwise.csv")
    macro("vPairsTotal", len(pw))
    macro("vPairsSignificant", int(pw.significant.sum()))
    # Why that count is zero, and must be: with five runs against five the
    # smallest attainable exact two-sided p is 2/C(10,5), and Holm over the
    # comparisons within a block multiplies it past 0.05 whatever the effect.
    per_block = int(pw.groupby(["model", "dataset", "phase"]).size().max())
    macro("vPairsPerBlock", per_block)
    macro("vPairsMinRawP", num(pw.p_raw.min(), 4))
    macro("vPairsMinHolmP", num(pw.p_holm.min(), 3))
    om = pd.read_csv(TABLES / "v2_stats_omnibus.csv")
    macro("vOmnibusEpsMin", num(om.epsilon_sq.min(), 2))
    macro("vOmnibusBlocks", len(om))
    macro("vOmnibusMaxP", sci_upper(om.p.max()))
    macro("vPairsLargePct",
          num(100 * (pw.magnitude == "large").mean(), 0))

    pc = pd.read_csv(TABLES / "v2_stats_phase_consistency.csv")
    macro("vPhaseRhoMin", num(pc.spearman_rho.min(), 2))
    macro("vPhaseRhoMax", num(pc.spearman_rho.max(), 2))
    macro("vPhaseSameBest", int(pc.same_best.sum()))
    # same_best is only "the cheapest is the same"; the full ordering is a
    # different and weaker claim, and the manuscript was making the stronger one.
    macro("vPhaseIdenticalOrder", int(pc.identical_order.sum()))
    macro("vPhaseBlocks", len(pc))

    # The pooled correlation is mostly the workload: a VGG-16/Tiny ImageNet run
    # outranks a ResNet-18/Fashion-MNIST run on every stack. Within a cell only
    # the ecosystem varies, which is the comparison the recommendation rests on.
    cr = pd.read_csv(TABLES / "v2_stats_cell_rho.csv")
    if not cr.empty:
        macro("vCellRhoMin", num(cr.spearman_rho.min(), 2))
        macro("vCellRhoMax", num(cr.spearman_rho.max(), 2))
        macro("vCellRhoCells", len(cr))

    ct = pd.read_csv(TABLES / "v2_stats_libtorch_control.csv")
    macro("vCtrlSpreadMin", num(ct.spread_shared_module.min(), 1))
    macro("vCtrlSpreadMax", num(ct.spread_shared_module.max(), 1))
    macro("vCtrlShareMin", num(ct.share_of_log_spread_pct.min(), 0))
    macro("vCtrlShareMax", num(ct.share_of_log_spread_pct.max(), 0))
    macro("vCtrlStacks", int(ct.n_shared_module.max()))
    macro("vFamilySpreadMin", num(ct.spread_libtorch_family.min(), 1))
    macro("vFamilySpreadMax", num(ct.spread_libtorch_family.max(), 1))
    resnet = ct[ct.model == "resnet18"]
    vgg = ct[ct.model == "vgg16"]
    macro("vCtrlSpreadResnetMin", num(resnet.spread_shared_module.min(), 1))
    macro("vCtrlSpreadResnetMax", num(resnet.spread_shared_module.max(), 1))
    macro("vCtrlSpreadVggMin", num(vgg.spread_shared_module.min(), 1))
    macro("vCtrlSpreadVggMax", num(vgg.spread_shared_module.max(), 1))
    # \vCtrlShareMin/Max are extrema over all six blocks, so a sentence about
    # what the control group accounts for *on ResNet-18* cannot be built from
    # them: the two architectures sit at opposite ends of this quantity.
    macro("vCtrlShareResnetMax", num(resnet.share_of_log_spread_pct.max(), 0))
    macro("vCtrlShareVggMin", num(vgg.share_of_log_spread_pct.min(), 0))

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabularx}{\linewidth}{@{}llRRRR@{}}",
        r"\toprule",
        # Both group sizes come from 14's table, which builds them from
        # SHARED_MODULE and LIBTORCH_FAMILY. The family size was written here as
        # "shared module + 1", which is a literal wearing the shape of a
        # derivation. The header says "module & build" because that is what the
        # group holds fixed: the same exported TorchScript module on the same
        # pinned LibTorch build, which is the pair R shares neither of.
        r"\textbf{Model} & \textbf{Dataset} & \textbf{All " + str(int(ct.n_all.max()))
        + r"} & \textbf{Shared module \& build (" + str(int(ct.n_shared_module.max()))
        + r")} & \textbf{LibTorch family (" + str(int(ct.n_libtorch_family.max()))
        + r")} & \textbf{Share of log spread} \\",
        r"\midrule",
    ]
    for _, r in ct.iterrows():
        lines.append(
            f"{MODEL[r.model]} & {DATASET[r.dataset]} & {r.spread_all:.1f}$\\times$ & "
            f"{r.spread_shared_module:.1f}$\\times$ & "
            f"{r.spread_libtorch_family:.1f}$\\times$ & "
            f"{r.share_of_log_spread_pct:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (OUT / "tab_control.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_control.tex').relative_to(REPO_ROOT)}")


# ------------------------------------------------------------- instrument --
def instrument_table() -> None:
    summ = pd.read_csv(TABLES / "v2_instrument_summary.csv")
    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Quantity} & \textbf{Mean per block} & \textbf{5th--95th pct} "
        r"& \textbf{Campaign-weighted} \\",
        r"\midrule",
    ]
    for _, r in summ.iterrows():
        # These labels come from a CSV and carry identifiers: escape what TeX
        # would otherwise read as markup.
        q = (r.quantity.replace("&", r"\&").replace("%", r"\%")
             .replace("_", r"\_"))
        lines.append(f"{q} & {r.mean_ratio_or_share:.3f} & "
                     f"{r.p05:.3f}--{r.p95:.3f} & {r.campaign_weighted:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_instrument.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_instrument.tex').relative_to(REPO_ROOT)}")


# --------------------------------------------------------------- scenario --
def industrial_scenario() -> None:
    """Relative inference multipliers, PyTorch as the baseline.

    Reviewer #1 (c11) and Reviewer #3 (M6) both caught a boundary mismatch in
    the submitted version of this scenario: it multiplied an assumed *per-GPU
    board power* by a ratio derived from CPU + GPU + RAM energy. Those are
    different boundaries, and the product is not a quantity.

    The fix is to take the ratio at the same boundary as the thing it multiplies.
    The scenario assumes a per-accelerator power budget, so the multiplier here
    is computed from **accelerator energy alone** -- the NVML counter, with the
    CPU package term excluded -- rather than from the total. That makes the
    arithmetic dimensionally honest; it does not make it a forecast, and the
    manuscript says so.

    The scenario is also stated in accelerator-hours rather than as energy at a
    fixed fleet. Multiplying energy by 6x at fixed N and fixed hours implies 6x
    the board power, which is seven times the card's limit and not a thing that
    can happen. What a less efficient stack actually costs at a fixed service
    target is more accelerators, or the same accelerators for longer. The euro
    figures are unchanged -- energy is accelerator-hours times a fixed power --
    but the quantity now corresponds to something physical.
    """
    runs = pd.read_csv(TABLES / "v2_run_totals.csv")
    infer = runs[runs.phase == "Inference"]
    per_eco = infer.groupby("ecosystem").agg(gpu_j=("gpu_j", "mean"),
                                             duration_s=("duration_s", "mean"))
    per_eco["gpu_w"] = per_eco.gpu_j / per_eco.duration_s
    base = "Python/PyTorch"
    # Two multipliers, and the scenario needs both. A less efficient stack takes
    # longer per unit of work (accelerator-hours) AND may draw different power
    # while doing it. Scaling the fleet by the ENERGY multiplier while holding
    # per-card draw fixed asserts a draw the data contradict: R needs 11.7x the
    # accelerator-hours but only 6.2x the energy, because it draws half the
    # power per card. Fleet follows time; energy follows fleet times measured
    # power; the energy ratio then comes out right by construction.
    per_eco["time_mult"] = per_eco.duration_s / per_eco.loc[base, "duration_s"]
    per_eco["energy_mult"] = per_eco.gpu_j / per_eco.loc[base, "gpu_j"]
    mult = per_eco.energy_mult.sort_values()

    N_GPU, HOURS, EUR_PER_MWH = 100_000, 24, 100
    base_w = float(per_eco.loc[base, "gpu_w"])
    baseline_mwh = N_GPU * (base_w / 1000.0) * HOURS / 1000.0
    macro("vScenarioBaselineW", num(base_w, 0))
    macro("vScenarioBaselineMWh", num(baseline_mwh, 0))
    macro("vScenarioBaselineCost", f"{baseline_mwh * EUR_PER_MWH:,.0f}".replace(",", "\\,"))
    macro("vScenarioBest", SHORT[mult.index[0]])
    macro("vScenarioBestMult", num(mult.iloc[0], 2))
    macro("vScenarioWorst", SHORT[mult.index[-1]])
    macro("vScenarioWorstMult", num(mult.iloc[-1], 2))
    macro("vScenarioWorstTimeMult", num(per_eco.loc[mult.index[-1], "time_mult"], 1))
    macro("vScenarioWorstW", num(per_eco.loc[mult.index[-1], "gpu_w"], 0))
    macro("vScenarioSpreadGpu", num(mult.max() / mult.min(), 1))
    macro("vScenarioSpreadTime",
          num(per_eco.time_mult.max() / per_eco.time_mult.min(), 1))
    total_mult = (infer.groupby("ecosystem").energy_j.mean()
                  / infer.groupby("ecosystem").energy_j.mean()[base])
    macro("vScenarioSpreadTotal", num(total_mult.max() / total_mult.min(), 1))

    # The training-budget arithmetic, generated because the manuscript had it
    # wrong by an order of magnitude: a 1 GWh budget costs EUR 100,000 in total,
    # so no spread inside it can be "hundreds of thousands".
    spreads = pd.read_csv(TABLES / "v2_between_run_statistics.csv")
    tr = spreads[spreads.phase == "Training"]
    ratios = (tr.groupby(["model", "dataset"]).mean_energy_J
              .agg(lambda x: x.max() / x.min()))
    budget_eur = 1000 * EUR_PER_MWH
    macro("vTrainBudgetEur", f"{budget_eur:,.0f}".replace(",", "\\,"))
    lo = budget_eur - budget_eur / ratios.min()
    hi = budget_eur - budget_eur / ratios.max()
    macro("vTrainBudgetSavingMinEur", f"{min(lo, hi):,.0f}".replace(",", "\\,"))
    macro("vTrainBudgetSavingMaxEur", f"{max(lo, hi):,.0f}".replace(",", "\\,"))

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Accelerator-hours} & \textbf{Fleet} "
        r"& \textbf{Measured draw} & \textbf{Energy (MWh/day)} "
        r"& \textbf{Cost (\euro/day)} \\",
        r"\midrule",
    ]
    for eco in mult.index:
        t = float(per_eco.loc[eco, "time_mult"])
        w = float(per_eco.loc[eco, "gpu_w"])
        fleet = N_GPU * t
        mwh = fleet * (w / 1000.0) * HOURS / 1000.0
        name = SHORT[eco] + (" (baseline)" if eco == base else "")
        # Thousands separators inserted per cell: a blanket comma replacement
        # also hits the comma inside a LaTeX thin space and doubles it.
        def thin(x):
            return f"{x:,.0f}".replace(",", "\\,")

        cells = [name, f"{t:.2f}$\\times$", thin(fleet), f"{w:.0f}\\,W",
                 thin(mwh), thin(mwh * EUR_PER_MWH)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_industrial.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_industrial.tex').relative_to(REPO_ROOT)}")


# --------------------------------------------------- the audited campaign ----
def audited_campaign_facts() -> None:
    """Values the defect catalogue quotes about the earlier campaign.

    These come from 01_data_audit.py rather than from the replication, and they
    are generated here for the same reason as everything else: the catalogue is
    an evidentiary table, and a number in it that cannot be traced to a file is
    exactly the failure the paper is about.
    """
    path = TABLES / "audit_measurement_boundary.csv"
    if not path.exists():
        return
    b = pd.read_csv(path)
    versions = sorted(b.codecarbon.astype(str).unique())
    macro("vAuditCCVersions", " and ".join(versions))
    par = TABLES / "impl_parallelism_vs_ranking.csv"
    if par.exists():
        from scipy import stats as _st
        d = pd.read_csv(par)
        rho, pv = _st.spearmanr(d.loader_threads, d.mean_duration_s)
        macro("vAuditLoaderRho", num(rho, 2))
        macro("vAuditLoaderP", num(pv, 3))
        macro("vAuditLoaderThreadsMin", int(d.loader_threads.min()))
        macro("vAuditLoaderThreadsMax", int(d.loader_threads.max()))
        macro("vAuditDurationSpread",
              num(d.mean_duration_s.max() / d.mean_duration_s.min(), 1))
        macro("vAuditEnergySpread",
              num(d.mean_energy_J.max() / d.mean_energy_J.min(), 1))
        macro("vAuditLearningRates",
              len(d.learning_rate.unique()))

    machine = b[b.tracking_mode == "machine"]
    for v in versions:
        # Averaged over machine-mode stacks only: the one process-mode stack
        # reports essentially no RAM term and would drag its version's mean down
        # for a reason that has nothing to do with the version.
        g = machine[machine.codecarbon.astype(str) == v]
        if not len(g):
            continue
        host = float(g.mean_cpu_power_w.mean() + g.mean_ram_power_w.mean())
        tag = "Old" if v == min(versions) else "New"
        macro(f"vAuditHost{tag}W", num(host, 0))

    process = b[b.tracking_mode == "process"]
    macro("vAuditRamShareMin", num(machine.ram_energy_pct.min(), 0))
    macro("vAuditRamShareMax", num(machine.ram_energy_pct.max(), 0))
    if len(process):
        macro("vAuditProcessRamSharePct", num(process.ram_energy_pct.iat[0], 3))
        macro("vAuditProcessStacks", len(process))
    # a constant CPU power is the tell that the tool fell back to a model
    macro("vAuditConstantCpuStacks", int((b.cpu_power_distinct == 1).sum()))
    macro("vAuditStacks", len(b))


# -------------------------------------------------------------- convergence --
def v1_boundary_facts() -> None:
    """The boundary argument, made inside the earlier campaign's own data.

    The manuscript compared this campaign's 18.2x against the earlier one's
    4.6x, on a different machine, two paragraphs after saying that absolute
    figures are not comparable between the two machines. The comparison it
    needs is available without leaving the old campaign: the same runs give
    4.58x under CodeCarbon's total and 8.54x at a GPU-only boundary.
    """
    h = pd.read_csv(TABLES / "table_headline_spreads.csv")
    tr = h[h.phase == "Training"]
    conf = tr[tr.energy_definition.str.startswith("as measured")].iloc[0]
    gpu = tr[tr.energy_definition.str.startswith("GPU only")].iloc[0]
    macro("vAuditSpreadConfounded", num(conf.spread_x, 1))
    macro("vAuditSpreadGpuOnly", num(gpu.spread_x, 1))


def convergence_facts() -> None:
    bm = pd.read_csv(TABLES / "v2_convergence_by_model.csv")
    vgg = bm[bm.model == "vgg16"]
    res = bm[bm.model == "resnet18"]
    macro("vCollapsedVgg", int(vgg.n_collapsed.sum()))
    macro("vCollapsedVggRuns", int(vgg.n_runs.sum()))
    macro("vCollapsedResnet", int(res.n_collapsed.sum()))
    macro("vCollapsedResnetRuns", int(res.n_runs.sum()))
    for ds, tag in (("cifar100", "Cifar"), ("tinyimagenet", "Tiny")):
        row = vgg[vgg.dataset == ds]
        if len(row):
            macro(f"vCollapsePct{tag}", num(row.collapse_pct.iat[0], 0))

    be = pd.read_csv(TABLES / "v2_convergence_by_ecosystem.csv")
    macro("vCollapseEcosystems",
          int(be[be.n_collapsed > 0].ecosystem.nunique()))

    cond = pd.read_csv(TABLES / "v2_convergence_conditional.csv")
    if cond.empty:
        raise ValueError("v2_convergence_conditional has no rows, so "
                         "\\vCondBlockName and the spread macros beside it are "
                         "not defined for this campaign")
    worst = cond.loc[cond.raw_spread_pp.idxmax()]
    macro("vCondBlockRaw", num(worst.raw_spread_pp, 1))
    macro("vCondBlockConverged", num(worst.converged_spread_pp, 1))
    macro("vCondBlockName", f"{MODEL[worst.model]} on {DATASET[worst.dataset]}")
    macro("vCondSpreadMax", num(cond.converged_spread_pp.max(), 1))

    sig_path = TABLES / "v2_convergence_signature.csv"
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        coll = sig[sig.diagnosis.str.startswith("optimisation")]
        pipe = sig[sig.diagnosis.str.startswith("pipeline")]
        macro("vSigCollapseN", len(coll))
        macro("vSigPipelineN", len(pipe))
        if len(coll):
            macro("vSigCollapseLossDropMax",
                  num(coll.train_loss_drop_pct.abs().max(), 1))
        if len(pipe):
            macro("vSigPipelineLossDropMin",
                  num(pipe.train_loss_drop_pct.min(), 0))

    # The pipeline defect the discriminator found has been fixed and its runs
    # re-executed, so it is no longer in the campaign. The manuscript still
    # reports it, and a claim about a defect should be as traceable as a claim
    # about a result, so the discarded evidence is preserved as a record.
    rec = REPO_ROOT / "results" / "revision" / "record" / "vgg_fashion_pipeline_defect.csv"
    if rec.exists():
        r = pd.read_csv(rec, comment="#")
        macro("vDefectRuns", len(r))
        macro("vDefectLossDropMin", num(r.train_loss_drop_pct.min(), 0))
        macro("vDefectLossDropMax", num(r.train_loss_drop_pct.max(), 0))
        macro("vDefectAccMax", num(r.final_test_acc_pct.max(), 1))

    initialiser_facts()
    campaign_record_facts()
    utilisation_facts()
    collapse_mechanism_facts()

    ht = pd.read_csv(TABLES / "v2_convergence_homogeneity.csv")
    # 15 writes this with a header and no rows when no cell is susceptible at
    # all; there is then no rate, no p and nothing to say about homogeneity.
    only_row(ht, "v2_convergence_homogeneity", "vCollapseHomogeneityP")
    macro("vCollapseRateMin", num(ht.collapse_pct.min(), 0))
    macro("vCollapseRateMax", num(ht.collapse_pct.max(), 0))
    macro("vCollapseRateOverall", num(ht.overall_pct.iat[0], 0))
    macro("vCollapseHomogeneityP", num(ht.permutation_p.iat[0], 2))
    macro("vCollapseNeverEcosystems",
          int((ht.n_collapsed == 0).sum()))
    # The exact answer, and the one the manuscript should be quoting: 15 has
    # computed Freeman-Halton since commit 5622c7e and nothing carried it out of
    # the table, so the paper argued about a permutation p and a chi-square p
    # while the test that needs neither an approximation nor a seed sat unread
    # in a column beside them.
    macro("vCollapseExactP", num(ht.exact_p.iat[0], 4))
    # The manuscript contrasts the permutation p with the chi-square one it
    # declines to use. It quoted 0.017 for the latter, which belonged to an
    # earlier version of this table; the value is 0.0065. Both are conditional
    # on chi-square being defined at all: with no collapses anywhere the
    # expected frequencies are zero, scipy raises, 15 records NaN, and there is
    # no approximate answer to contrast the exact one with. Emitting nothing
    # here fails the LaTeX build on \vCollapseChiSqP, which is the correct
    # outcome -- the passage that quotes it is arguing about a table that has
    # become all zeros and has to be rewritten, not filled in.
    if pd.notna(ht.chi_square_p.iat[0]):
        macro("vCollapseChiSqP", num(ht.chi_square_p.iat[0], 4))
        macro("vCollapseChiSqMinExpected", num(ht.chi_square_min_expected.iat[0], 1))

    w = only_row(pd.read_csv(TABLES / "v2_convergence_waste.csv"),
                 "v2_convergence_waste", "vWastedPct")
    macro("vWastedPct", num(w.wasted_pct, 1))
    macro("vWastedMJ", num(w.collapsed_energy_MJ, 1))
    macro("vTrainingEnergyMJ", num(w.training_energy_MJ, 1))
    reexecution_facts()


def collapse_mechanism_facts() -> None:
    """What the collapsed runs actually are, and what decides them.

    The manuscript said the network "settles into predicting a single class"
    and that "whether it does is decided by the initialisation". The traces say
    otherwise on both counts. Every collapsed run sits at a loss of exactly
    ln(N) for N classes, which is a uniform output distribution -- a dead
    network, not a confident wrong one. And the three stacks that load the
    byte-identical exported module, from identical weights, disagree about
    which repetitions collapse: so what decides it is the stochastic path
    through training, not the starting point.

    The per-epoch traces come from common.read_campaign_metrics, the one reader
    15_convergence and check_consistency also use, so the collapse count here
    and \\vSigCollapseN there cannot describe different populations of the same
    campaign. They used to come from results/replication/metrics.csv.gz
    -- the replication package, which nothing in the pipeline rebuilds, and
    which held the first campaign. That is worse than staleness here: in the
    first campaign Python/PyTorch built torchvision fresh and loaded no shared
    module, so the discordance below was measured between stacks that did not
    share weights, under a sentence asserting that they did.
    """
    facts = collapse_mechanism(read_campaign_metrics())
    macro("vCollapseRuns", facts["n_collapsed"])
    if facts["loss_deviation_max"] is not None:
        macro("vCollapseLossDeviationMax", f"{facts['loss_deviation_max']:.4f}")
    # With no collapsed run there is no worst deviation from ln(N), and no
    # defensible value to invent for one: the macro is left undefined and the
    # sentence quoting it has to go, which is the point of \vCollapseRuns above.

    macro("vCollapseDiscordantCells", facts["discordant_cells"])
    macro("vCollapseSharedStacks", facts["shared_stacks"])
    macro("vSigCollapseJava", facts["java_collapses"])


def campaign_record_facts() -> None:
    """Two facts about what the stacks record, from what they recorded.

    train_acc is optional by design -- the spec says "where the stack computes
    it" -- and no analysis reads it, so the manuscript's claim about how many
    stacks provide it can only be checked against the files. The fingerprint
    count is the size of the sample every stack's data-parity claim rests on,
    and it has to be the same number everywhere or the claim is not about the
    same thing; that is asserted here rather than assumed.
    """
    m = read_campaign_metrics()
    if "train_acc" in m:
        have = m.groupby("ecosystem").train_acc.apply(lambda s: s.notna().any())
        macro("vTrainAccStacks", int(have.sum()))

    counts = set()
    for run_dir in sorted(CAMPAIGN_DIR.glob("*")):
        path = run_dir / "data_fingerprint.csv"
        if not path.is_dir() and path.exists() and path.stat().st_size:
            fp = pd.read_csv(path)
            # The two writers name the column differently: the Python harness
            # writes n_values, the shared bridge writes n.
            column = "n_values" if "n_values" in fp else "n"
            if column in fp:
                counts.update(int(v) for v in fp[column].dropna())
    if len(counts) == 1:
        macro("vDataFpValues", counts.pop())
    elif counts:
        raise ValueError(
            "the stacks fingerprinted different numbers of values, so "
            "\\vDataFpValues is not one number and the data-parity claim is "
            f"not about one sample: {sorted(counts)}")


def utilisation_facts() -> None:
    """Utilisation beside energy, for the stack the energy tables call cheapest.

    R draws about half the power of the others and takes far longer. Energy
    alone cannot say whether that is a slow kernel or a card waiting on a host,
    and the difference matters to every conclusion drawn from R's numbers. The
    1 Hz record answers it directly -- see 19_gpu_utilisation, including which
    runs the record covers, since it began after the campaign did.
    """
    path = TABLES / "v2_gpu_utilisation_by_ecosystem.csv"
    if not path.exists():
        return
    u = pd.read_csv(path)
    if u.empty:
        return
    # The stack with the lowest utilisation, found rather than named.
    worst = u.groupby("ecosystem").util_mean_pct.mean().idxmin()
    macro("vLowestGpuUtilEco", SHORT[worst])
    e = u[u.ecosystem == worst]
    for model, tag in (("resnet18", "Resnet"), ("vgg16", "Vgg")):
        row = e[e.model == model]
        if not row.empty:
            macro(f"vLowestGpuUtil{tag}Pct",
                  span(row.util_min_pct.iat[0], row.util_max_pct.iat[0], 1))
    macro("vLowestGpuMemMinMiB", num(e.mem_min_mib.min(), 0))
    macro("vLowestGpuMemMaxMiB", num(e.mem_max_mib.max(), 0))
    macro("vLowestGpuRunMinW", num(e.power_min_w.min(), 0))
    macro("vLowestGpuRunMaxW", num(e.power_max_w.max(), 0))
    runs = pd.read_csv(TABLES / "v2_gpu_utilisation_by_run.csv")
    # How much of the campaign the record saw, so no reader has to assume all.
    macro("vGpuUtilCoveredRuns", len(runs))


def initialiser_facts() -> None:
    """What actually decides whether VGG-16 collapses.

    Held framework, optimiser, learning rate and data order fixed and varied
    only the initialiser. These runs were not part of either campaign and no
    script here can regenerate them, so they live in results/revision/record/
    beside the pipeline defect -- but they are read from that file rather than
    typed in, because a number the manuscript quotes should have exactly one
    place it can be corrected.
    """
    path = REPO_ROOT / "results" / "revision" / "record" / "initialiser_trials.csv"
    if not path.exists():
        return
    r = pd.read_csv(path, comment="#")
    trials = r[r.quantity == "collapse_trials"].set_index("initialiser")
    counts = trials.n_trials.unique()
    assert len(counts) == 1, f"initialisers were not tried equally often: {counts}"
    macro("vInitCollapseTrials", int(counts[0]))
    for name in ("He", "Glorot", "Xavier"):
        macro(f"vInitCollapse{name}", int(trials.n_collapsed.loc[name]))
    width = r[r.quantity == "stem_width_ratio"]
    macro("vInitStemWidthRatioJava", num(width.value.iat[0], 1))
    # flax's lecun_normal stem, the same comparison for JAX (REVISION_LOG section 14).
    jax = r[r.quantity == "stem_width_ratio_jax"]
    if len(jax):
        macro("vInitStemWidthRatioJax", num(jax.value.iat[0], 1))

    # Two more first-campaign counts with no table behind them, for the same
    # reason: the code that produced them has been corrected, so nothing in the
    # analysis can rederive them. Read, not typed.
    div = REPO_ROOT / "results" / "revision" / "record" / "first_campaign_divergences.csv"
    if div.exists():
        d = pd.read_csv(div, comment="#").set_index("quantity")
        macro("vSeedDivergentStacks", int(d.n_stacks.loc["seed_divergent"]))
        macro("vPixelDivergentStacks", int(d.n_stacks.loc["pixel_divergent"]))


def collapse_mechanism(m: pd.DataFrame) -> dict:
    """The collapse quantities derivable from one campaign's per-epoch traces.

    Shared by the current campaign and the superseded one rather than written
    twice: the manuscript now reports both, this campaign's zero beside the
    first campaign's twelve, and two copies of this arithmetic would be two
    definitions of what a collapse is.
    """
    chance = {"fashionmnist": 10.0, "cifar100": 1.0, "tinyimagenet": 0.5}
    n_classes = {"fashionmnist": 10, "cifar100": 100, "tinyimagenet": 200}
    final = m.sort_values("epoch").groupby("run").tail(1).copy()
    final["collapsed"] = final.test_acc <= final.dataset.map(chance) * 1.5
    collapsed_runs = set(final[final.collapsed].run)
    traces = m[m.run.isin(collapsed_runs)].copy()

    deviation = None
    if collapsed_runs:
        traces["ln_n"] = np.log(traces.dataset.map(n_classes))
        deviation = float((traces.train_loss - traces.ln_n).abs().max())

    # Do the byte-identical-module stacks agree about which runs collapse?
    # 14_v2_statistics defines the control group; a second list here is a second
    # opinion about which stacks load the shared module, and the two would only
    # ever be found to disagree by a reader.
    shared = sibling("14_v2_statistics").SHARED_MODULE
    f = final[final.ecosystem.isin(shared)]
    per_cell = f.groupby(["model", "dataset", "repetition"]).collapsed
    vgg = final[final.model == "vgg16"]
    return {
        "n_collapsed": len(collapsed_runs),
        "loss_deviation_max": deviation,
        "discordant_cells": int(per_cell.apply(
            lambda x: 0 < x.sum() < len(x)).sum()),
        # Counted from the campaign, not from the length of the list:
        # SHARED_MODULE carries both spellings of C++/LibTorch, so it has four
        # entries for three stacks.
        "shared_stacks": int(f.ecosystem.nunique()),
        "java_collapses": int((final.collapsed
                               & (final.ecosystem == "Java/DL4J")).sum()),
        "vgg_collapses": int(vgg.collapsed.sum()),
        "vgg_runs": int(len(vgg)),
    }


def first_campaign_collapse_facts() -> None:
    """The collapse finding, which belongs to the campaign that has it.

    The second campaign has no collapses at all -- every stack now initialises
    from the same exported weights, which is what log 3 predicted -- so the
    manuscript reports the phenomenon as the first campaign's result and this
    campaign's zero as the evidence that the alignment closed it. Both sets of
    numbers therefore have to be derived, and the first campaign's cannot be
    typed in from the superseded numbers.tex.

    results/campaign_v2_first_campaign/ is read here and in 18_precision_ablation
    and nowhere else. The tables come from 15_convergence --campaign v1.
    """
    ht = TABLES / "v1_convergence_homogeneity.csv"
    if not ht.exists():
        raise ValueError(
            "v1_convergence_homogeneity.csv is missing, so the "
            "\\vCollapseFirst* macros cannot be derived; run "
            "15_convergence.py --campaign v1")
    ht = only_row(pd.read_csv(ht), "v1_convergence_homogeneity",
                  "vCollapseFirstRateOverall")
    macro("vCollapseFirstRateOverall", num(ht.overall_pct, 0))
    macro("vCollapseFirstExactP", num(ht.exact_p, 4))
    macro("vCollapseFirstChiSqMinExpected", num(ht.chi_square_min_expected, 1))

    w = only_row(pd.read_csv(TABLES / "v1_convergence_waste.csv"),
                 "v1_convergence_waste", "vCollapseFirstWastedPct")
    macro("vCollapseFirstWastedPct", num(w.wasted_pct, 1))

    sig = pd.read_csv(TABLES / "v1_convergence_signature.csv")
    macro("vCollapseFirstSigCollapseN",
          int(sig.diagnosis.str.startswith("optimisation").sum()))
    macro("vCollapseFirstSigPipelineN",
          int(sig.diagnosis.str.startswith("pipeline").sum()))

    facts = collapse_mechanism(read_campaign_metrics(root=FIRST_CAMPAIGN))
    macro("vCollapseFirstVgg", facts["vgg_collapses"])
    macro("vCollapseFirstVggRuns", facts["vgg_runs"])
    macro("vCollapseFirstJava", facts["java_collapses"])
    macro("vCollapseFirstDiscordantCells", facts["discordant_cells"])
    macro("vCollapseFirstLossDeviationMax",
          f"{facts['loss_deviation_max']:.4f}")

    # What this campaign's zero can and cannot say. With no events in n trials
    # the exact one-sided 95% upper bound on the rate is 1 - 0.05^(1/n): the
    # largest rate that would still produce zero collapses one time in twenty.
    # It is the honest form of "the collapses are gone" -- log 25 wrote the
    # same bound at 20 runs, where it was 13.9%.
    hv2 = pd.read_csv(TABLES / "v2_convergence_homogeneity.csv")
    n = int(hv2.n_runs.sum())
    macro("vCollapseSusceptibleRuns", n)
    macro("vCollapseRateUpperNinetyFive", num(100 * (1 - 0.05 ** (1 / n)), 1))


def reexecution_facts() -> None:
    """Which runs were executed outside the main interleaved window.

    The threats section owes the reader a count and a scope, and had both wrong
    -- "the twenty Rust runs on Fashion-MNIST and Tiny ImageNet" against 25 runs
    over five configurations including VGG-16 on CIFAR-100. Read the timestamps.
    A run belongs to the later window if it started more than a day after the
    campaign's last first-window start; the largest gap inside a window is under
    an hour and the gap to the re-execution is days, so the threshold is not
    delicate.

    The timestamps are 11's consolidated per-block table, which is this
    campaign. They were read from results/replication/codecarbon.csv.gz, which
    is not: it is a package the pipeline writes and never rebuilds, and it still
    held the first campaign's eight-day window, so this function reported a late
    window that belonged to a campaign the manuscript no longer describes.
    """
    cc = pd.read_csv(TABLES / "v2_instrument_epochs.csv",
                     usecols=["ecosystem", "model", "dataset", "repetition",
                              "timestamp"])
    cc["t"] = pd.to_datetime(cc.timestamp)
    starts = cc.groupby(["ecosystem", "model", "dataset", "repetition"]).t.min()
    gap = starts.sort_values().diff()
    boundary = gap.idxmax() if gap.max() > pd.Timedelta("1D") else None
    if boundary is None:  # one contiguous window: nothing to declare
        return
    cutoff = starts[boundary]
    later = starts[starts >= cutoff].reset_index()

    macro("vLateRuns", len(later))
    macro("vLateConfigurations",
          later.groupby(["ecosystem", "model", "dataset"]).ngroups)
    macro("vLateEcosystems",
          ", ".join(SHORT[e] for e in sorted(later.ecosystem.unique())))
    macro("vLateGapDays", num((cutoff - starts[starts < cutoff].max()).days, 0))
    macro("vLateBlocks",
          ", ".join(sorted({f"{MODEL[m]}/{DATASET[d]}"
                            for m, d in zip(later.model, later.dataset)})))
    macro("vInterleavedRuns", len(starts) - len(later))

    # The between-window drift, measured rather than asserted: one configuration
    # re-executed in a third window and compared with its original runs.
    # An empty table means 17 refused the comparison -- see its
    # comparability_objection. The \vCalib* macros are then simply absent and the
    # LaTeX build fails on them, which is what should happen: the alternative is
    # the manuscript quoting a drift figure that nobody was willing to compute.
    cal_path = TABLES / "v2_window_calibration.csv"
    cal = pd.read_csv(cal_path) if cal_path.exists() else pd.DataFrame()
    if not cal.empty:
        macro("vCalibConfig", f"{SHORT[cal.ecosystem.iat[0]]}, "
                              f"{MODEL[cal.model.iat[0]]} on {DATASET[cal.dataset.iat[0]]}")
        for phase, tag in (("Training", "Train"), ("Inference", "Infer")):
            row = cal[cal.phase == phase]
            if row.empty:
                continue
            r = row.iloc[0]
            macro(f"vCalib{tag}DiffPct", num(abs(r.difference_pct), 1))
            # The same difference in the unit the instrument reads, because a
            # percentage of an inference block and of a training block are not
            # comparable quantities.
            macro(f"vCalib{tag}DiffJ", num(abs(r.original_J - r.recheck_J), 1))
            macro(f"vCalib{tag}SD", num(r.difference_in_sd, 1))
            macro(f"vCalib{tag}CVPct", num(r.within_window_cv_pct, 1))


# ----------------------------------------------------------------- ranking --
def ranking_facts() -> None:
    r = pd.read_csv(TABLES / "v2_instrument_ranking.csv")
    macro("vRankBlocks", len(r))
    macro("vRankIdentical", int(r.identical.sum()))
    # Reported over training only while the paper argued inference is where the
    # apparatus bites. Both phases, and each separately.
    for phase, tag in (("Training", "Train"), ("Inference", "Infer")):
        g = r[r.phase == phase]
        macro(f"vRank{tag}Identical", int(g.identical.sum()))
        macro(f"vRank{tag}Cells", len(g))


#: The comparability notes 18 writes are paragraphs, because they have to carry
#: their own evidence. A table column cannot, so each is abbreviated to a key
#: here and the note itself is printed under the table. Matched on a
#: distinctive fragment rather than in full, so a reworded note fails loudly at
#: the assertion below rather than silently printing the wrong reason.
#:
#: The previous version of this table put the whole note in the last column and
#: wrapped the result in \resizebox{\columnwidth}, which set the paper's most
#: important new table at 3.9 pt -- unreadable in print and below any journal's
#: floor. Keys plus a footnote list carry the same information at the
#: surrounding \footnotesize.
#: fragment -> (key printed in the column, footnote text, comparable?)
NOTE_SHORT = {
    "no known between-campaign difference": ("yes", None, True),
    "TorchScript module": (
        "yes",
        "PyTorch's harness also moved from eager torchvision to the exported "
        "TorchScript module between the executions, so this ratio bounds the "
        "precision effect from above.", True),
    "projection shortcut": (
        "network",
        "Python/TensorFlow ran a Model Garden ResNet-18 with a projection "
        "shortcut the other stacks do not have, and dropped its last partial "
        "batch.", False),
    "ConvolutionMode.Truncate": (
        "network",
        "Java/DL4J built its graph by hand, with truncated convolution padding "
        "and a bias on every convolution.", False),
    "block_until_ready": (
        "window",
        "Python/JAX was measured through a window that did not force "
        "completion in the first execution and does in this one.", False),
    "28x28 -> 32x32": (
        "data",
        "Fashion-MNIST training images were re-encoded $28\\to32$ on disk "
        "between the executions, so every stack's loader does different work.",
        False),
    "64x64 -> 32x32": (
        "data",
        "Tiny ImageNet training images were re-encoded $64\\to32$ on disk "
        "between the executions, so every stack's loader does different work.",
        False),
}


def precision_table(ct: pd.DataFrame) -> None:
    """Table~\\ref{tab:precision_contrast}: the ablation, cell by cell.

    Every ResNet-18 training cell is listed, comparable or not, because the
    argument is the exclusions as much as the ratio: a reader has to be able to
    see that the one stack which changed precision policy moved 3x while the
    stacks that did not moved by 1%, and that the cells left out were left out
    for a stated reason rather than for their answer.

    The comparable cells come first, because they are the argument; the rest
    follow under a rule, keyed to the notes printed beneath the table. The
    table is sized to fit a column at \\footnotesize rather than scaled down to
    fit it, which is how it came to be set at 3.9 pt.
    """
    rows = ct[(ct.row_kind == "stack contrast") & (ct.phase == "Training")
              & (ct.model == "resnet18")].copy()

    def short(note: str):
        for fragment, entry in NOTE_SHORT.items():
            if fragment in note:
                return entry
        raise ValueError(f"no short form for comparability note {note!r}; "
                         "18_precision_ablation's wording changed and "
                         "NOTE_SHORT has to change with it")

    # One footnote letter per distinct note, in the order the rows use them, so
    # the list under the table reads top to bottom with the rows.
    letters, notes = {}, []
    ordered = pd.concat([rows[rows.comparable == "yes"].sort_values(
                             ["dataset", "ecosystem"]),
                         rows[rows.comparable != "yes"].sort_values(
                             ["dataset", "ecosystem"])])
    body = []
    for _, r in ordered.iterrows():
        key, note, _ok = short(r.comparability_note)
        mark = ""
        if note is not None:
            if note not in letters:
                letters[note] = chr(ord("a") + len(letters))
                notes.append(note)
            mark = "$^{\\mathrm{%s}}$" % letters[note]
        cited = r.ecosystem == "Python/PyTorch" and r.comparable == "yes"
        name = SHORT[r.ecosystem]
        body.append(
            ((r"\textbf{" + name + r"}") if cited else name)
            + f" & {DATASET[r.dataset]} & {r.value_v1:.0f} & {r.value_v2:.0f} & "
            + ((r"\textbf{" + f"{r.ratio_v1_v2:.2f}" + r"$\times$}") if cited
               else f"{r.ratio_v1_v2:.2f}$\\times$")
            + f" & {key}{mark} \\\\")
    split = int((ordered.comparable == "yes").sum())

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean GPU energy per training epoch, ResNet-18, in the "
        r"superseded campaign (TF32 denied for Python/PyTorch alone) against "
        r"the current one (TF32 allowed for all seven). Five repetitions and "
        r"150 epochs per cell on each side. \textbf{Python/PyTorch on "
        r"CIFAR-100 is the contrast the text cites}; the cells above the rule "
        r"are the ones the two executions leave comparable, and each cell "
        r"below it is excluded for the stated reason rather than for its "
        r"value.}",
        r"\label{tab:precision_contrast}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrl@{}}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Dataset} & \textbf{TF32} "
        r"& \textbf{TF32} & \textbf{Ratio} & \textbf{Compar-} \\",
        r" & & \textbf{off (J)} & \textbf{on (J)} & & \textbf{able} \\",
        r"\midrule",
    ]
    lines += body[:split]
    lines += [r"\midrule"] + body[split:]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\par\vspace{2pt}")
        lines.append(r"\begin{minipage}{\columnwidth}\footnotesize\raggedright")
        lines.append(r"\par".join(
            r"$^{\mathrm{%s}}$~%s" % (letters[note], note) for note in notes))
        lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    (OUT / "tab_precision_contrast.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(OUT / 'tab_precision_contrast.tex').relative_to(REPO_ROOT)}")


def precision_ablation_facts() -> None:
    """What TF32 cost, from the campaigns and from the kernel probe.

    Two measurements of one effect, and the manuscript needs both because
    neither is sufficient alone. 18_precision_ablation reads the two campaigns:
    Python/PyTorch ran with TF32 denied in the first and allowed in the second
    while six stacks did not change, which is an ablation with a control group
    at 150 training epochs per cell. scripts/probe_tf32 measures the kernels
    directly, which the campaign contrast cannot isolate because PyTorch also
    moved from eager torchvision to the exported TorchScript module between the
    campaigns.

    Only the cells 18 grades comparable are used. The rest are excluded there
    for stated reasons -- a different network, a re-encoded dataset, a changed
    measurement window -- and pulling a ratio out of one of them would be
    quoting the audit as if it were the flag.
    """
    ct = pd.read_csv(TABLES / "v2_tf32_campaign_contrast.csv")
    train = ct[(ct.phase == "Training") & (ct.model == "resnet18")
               & (ct.comparable == "yes")]

    stacks = train[train.row_kind == "stack contrast"]
    pt = stacks[stacks.ecosystem == "Python/PyTorch"]
    if pt.empty:
        raise ValueError("no comparable Python/PyTorch cell in "
                         "v2_tf32_campaign_contrast, so \\vPrecisionPytorchRatioMin "
                         "and the macros beside it are not defined")
    macro("vPrecisionPytorchRatioMin", num(pt.ratio_v1_v2.min(), 2))
    macro("vPrecisionPytorchRatioMax", num(pt.ratio_v1_v2.max(), 2))
    # The control group is the argument: stacks whose precision policy did not
    # change must not move, or the ratio above is measuring something else.
    others = stacks[stacks.ecosystem != "Python/PyTorch"]
    macro("vPrecisionOthersMaxChangePct",
          num(100 * (others.ratio_v1_v2 - 1).abs().max(), 1))
    macro("vPrecisionControlStacks", len(others))

    precision_table(ct)

    gaps = train[train.row_kind == "PyTorch/C++ gap"]
    if len(gaps) != 1:
        # One comparable cell today (ResNet-18 on CIFAR-100). If 18 ever grades
        # a second one comparable these become a range, and quietly taking the
        # first would put one dataset's number under a sentence about both.
        raise ValueError(
            f"{len(gaps)} comparable PyTorch/C++ gap cells, not 1; "
            f"\\vPrecisionGapBefore and \\vPrecisionGapAfter name a single "
            f"cell and need rewriting as a range: "
            f"{', '.join(sorted(gaps.dataset))}")
    gap = gaps.iloc[0]
    macro("vPrecisionGapBefore", num(gap.value_v1, 2))
    macro("vPrecisionGapAfter", num(gap.value_v2, 2))
    macro("vPrecisionGapDataset", DATASET[gap.dataset])

    # "Precision", not "Tf32": a TeX control sequence is letters only, so
    # \vTf32EnergyRatio parses as \vTf followed by the text "32EnergyRatio" and
    # \newcommand never sees a name. macro()'s assert catches it.
    # The kernel probe. Read by flag rather than by row order: the table gained
    # a fifth cell and a matmul column after these macros were written, and a
    # positional read would have silently picked up the wrong row.
    probe_path = TABLES / "v2_tf32_ablation.csv"
    if not probe_path.exists():
        return
    probe = pd.read_csv(probe_path)

    def cell(column: str, matmul: bool = False, **flags):
        """One row of the probe, chosen by flag rather than by position.

        The table has gained a matmul column, a fifth cell and a cuDNN-disabled
        sixth since these macros were written, and a positional read would have
        silently moved to a different configuration each time. A flag the table
        does not carry is not filtered on, so an older probe output still
        answers for the cells it does have.
        """
        want = pd.Series(True, index=probe.index)
        for name, value in flags.items():
            # A flag the table does not carry means the cell asked for is not in
            # it. Returning None rather than ignoring the flag: ignoring it
            # matched the cuDNN-disabled cell against the default row of a table
            # that has no such cell, and emitted a ratio of exactly 1.00.
            if name not in probe:
                return None
            want &= probe[name].astype(str).str.lower() == str(value).lower()
        # The cells that isolate the v1 defect hold matmul off; the fifth is
        # v2's configuration and is not part of the v1 ratios. Pinned only when
        # the column exists, so an older probe output still answers.
        if "matmul_allow_tf32" in probe:
            want &= (probe.matmul_allow_tf32.astype(str).str.lower()
                     == str(matmul).lower())
        rows = probe[want]
        return None if rows.empty or column not in rows else float(rows[column].iat[0])

    #: The probe's median columns are rounded for display -- 0.0978 s against a
    #: 0.0060 s baseline -- and dividing them gave 16.30 where the probe's own
    #: ratio column, computed from the unrounded medians, says 16.417. A
    #: referee recomputing from the released CSV got a different number from the
    #: one the manuscript printed. So the ratios are read, not recomputed.
    RATIO_COLUMN = {"gpu_j_median": "ratio_gpu_j_vs_%s",
                    "wall_s_median": "ratio_wall_s_vs_%s"}

    def ratio(name: str, column: str, base_kind: str, **flags) -> None:
        """Emit one macro from the probe's own ratio column for that cell.

        `base_kind` is "v1_baseline" or "v2_default" and names the column the
        probe already computed the quotient into, so the macro and the CSV
        cannot drift apart. A probe output predating those columns falls back
        to dividing the medians, which is what the manuscript used to do.
        """
        col = RATIO_COLUMN[column] % base_kind
        value = cell(col, **flags) if col in probe else None
        if value is None:
            base_flags = (dict(matmul=False, cudnn_allow_tf32=True,
                               cudnn_deterministic=False)
                          if base_kind == "v1_baseline"
                          else dict(matmul=True, cudnn_enabled=True,
                                    cudnn_allow_tf32=True,
                                    cudnn_deterministic=False))
            here, base = cell(column, **flags), cell(column, **base_flags)
            if not (here and base):
                return
            value = here / base
        macro(name, num(value, 2))

    for column, tag in (("gpu_j_median", "Energy"), ("wall_s_median", "Time")):
        ratio(f"vPrecisionKernel{tag}Ratio", column, "v1_baseline",
              cudnn_allow_tf32=False, cudnn_deterministic=False)
        if tag == "Energy":
            ratio("vPrecisionKernelDetEnergyRatio", column, "v1_baseline",
                  cudnn_allow_tf32=False, cudnn_deterministic=True)
            ratio("vPrecisionKernelDetOnlyRatio", column, "v1_baseline",
                  cudnn_allow_tf32=True, cudnn_deterministic=True)
        # cuDNN disabled entirely, against v2's own configuration rather than
        # v1's baseline: this one answers "what does the library buy", which is
        # a different question from "what does TF32 buy", and it is the only
        # cell whose matmul flag matches v2 rather than v1. Absent from an older
        # probe output, and then simply not emitted -- the probe is a host
        # measurement and may legitimately be run without the cell.
        ratio(f"vCudnnOff{tag}Ratio", column, "v2_default",
              matmul=True, cudnn_enabled=False, cudnn_allow_tf32=True,
              cudnn_deterministic=False)


def main() -> None:
    epochs = design_facts()
    apparatus_facts()
    instrument_facts(epochs)
    stats = repeatability_facts()
    energy_facts(stats)
    quality_facts()
    audited_campaign_facts()
    v1_boundary_facts()
    convergence_facts()
    statistics_facts()
    instrument_table()
    industrial_scenario()
    ranking_facts()
    precision_ablation_facts()
    first_campaign_collapse_facts()

    body = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"% every value the manuscript quotes, straight from results/revision/tables/",
    ]
    for name in sorted(macros):
        body.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    (OUT / "numbers.tex").write_text("\n".join(body) + "\n")
    print(f"  wrote {(OUT / 'numbers.tex').relative_to(REPO_ROOT)} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
