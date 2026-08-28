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
from common import REPO_ROOT  # noqa: E402

TABLES = REPO_ROOT / "results" / "revision" / "tables"
OUT = REPO_ROOT / "paper" / "generated"
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
DATASET = {"fashionmnist": "Fashion-MNIST", "cifar100": "CIFAR-100",
           "tinyimagenet": "Tiny ImageNet"}
MODEL = {"resnet18": "ResNet-18", "vgg16": "VGG-16"}

macros: dict[str, str] = {}


def macro(name: str, value) -> None:
    """Register one \\newcommand. Names must be letters only (TeX restriction)."""
    assert name.isalpha(), f"macro name {name!r} must be letters only"
    macros[name] = str(value)


def num(x, digits: int = 0) -> str:
    """A number formatted for siunitx, without thousands separators."""
    return f"{x:.{digits}f}"


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

    macro("vConformanceChecks", len(check_consistency.run()))
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
    anchor = tex.index(r"\label{tab:catalogue}")
    start = tex.index(r"\begin{tabularx}", anchor)
    body = tex[start:tex.index(r"\end{tabularx}", start)]

    total = ours = 0
    for row in body.split("\\\\\n"):
        lines = [ln.strip() for ln in row.strip().split("\n") if ln.strip()]
        lines = [ln for ln in lines
                 if not ln.startswith((r"\toprule", r"\midrule", r"\addlinespace",
                                       r"\bottomrule", r"\begin{tabularx}"))]
        if not lines or lines[0].startswith((r"\multicolumn", r"\textbf")):
            continue
        total += 1
        ours += r"$^\ast$" in row
    assert total > 10 and 0 < ours < total, f"catalogue parse looks wrong: {total}/{ours}"
    macro("vDefectCount", total)
    macro("vDefectOurs", ours)


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
    macro("vGapInstrumentPct", num(min(100.0, gi.share_of_gap_explained_pct), 0))
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

    # The two ecosystems that rule out any length-based model, named by the data.
    blk = pd.read_csv(TABLES / "v2_coverage_per_block.csv",
                      usecols=["ecosystem", "window_excess_s"])
    padded_share = blk.assign(p=blk.window_excess_s > 0.5).groupby("ecosystem").p.mean()
    never = padded_share.idxmin()
    always = padded_share.idxmax()
    macro("vWindowUnpaddedEco", SHORT[never])
    macro("vWindowUnpaddedEcoBlocks", int((blk.ecosystem == never).sum()))
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
        r"\begin{tabular}{lrrrr}",
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
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_power_distortion.tex").write_text("\n".join(lines) + "\n")
    print("  wrote paper/generated/tab_power_distortion.tex")

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Median block} & \textbf{Tracked share} "
        r"& \textbf{Of the untracked rest, the instrument} \\",
        r"\midrule",
    ]
    for _, row in cov.sort_values("coverage_pct").iterrows():
        share = min(100.0, row.untracked_that_is_instrument_pct)
        lines.append(
            f"{SHORT[row.ecosystem]} & {row.median_block_s:.1f}\\,s & "
            f"{row.coverage_pct:.0f}\\% & {share:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_coverage.tex").write_text("\n".join(lines) + "\n")
    print("  wrote paper/generated/tab_coverage.tex")

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
    print(f"  wrote paper/generated/{stem}.tex")


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
    print("  wrote paper/generated/tab_spread.tex")


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
                cells.append(f"{med / 1000:.0f} ({int(reached)}/{int(n)})")
            else:
                cells.append(f"{med / 1000:.0f}")
        tlines.append(f"{SHORT[eco]} & " + " & ".join(cells) + r" \\")
    tlines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_energy_to_target.tex").write_text("\n".join(tlines) + "\n")
    print("  wrote paper/generated/tab_energy_to_target.tex")
    macro("vTargetFashion", num(q[q.dataset == "fashionmnist"].target_acc_pct.iat[0], 0))
    macro("vTargetCifar", num(q[q.dataset == "cifar100"].target_acc_pct.iat[0], 0))
    macro("vTargetTiny", num(q[q.dataset == "tinyimagenet"].target_acc_pct.iat[0], 0))
    fash = tgt[tgt.dataset == "fashionmnist"]
    macro("vTargetFashionSpread",
          num(fash.median_kJ.max() / fash.median_kJ.min(), 1))
    macro("vTargetNeverStacks",
          int((tgt[tgt.dataset == "tinyimagenet"].reached == 0).sum()))

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Fashion-MNIST} & \textbf{CIFAR-100} "
        r"& \textbf{Tiny ImageNet} \\",
        r"\midrule",
    ]
    piv = summ.pivot(index="ecosystem", columns="dataset", values="final_test_acc_pct")
    piv_ok = summ_ok.pivot(index="ecosystem", columns="dataset",
                           values="final_test_acc_pct")
    for eco, row in piv.iterrows():
        cells = []
        for d in ("fashionmnist", "cifar100", "tinyimagenet"):
            v, w = row.get(d, np.nan), piv_ok.loc[eco].get(d, np.nan)
            if pd.isna(v):
                cells.append("--")
            elif pd.isna(w) or abs(v - w) < 0.005:
                cells.append(f"{v:.2f}")
            else:  # collapsed runs excluded, in parentheses
                cells.append(f"{v:.2f} ({w:.2f})")
        lines.append(f"{SHORT[eco]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_accuracy.tex").write_text("\n".join(lines) + "\n")
    print("  wrote paper/generated/tab_accuracy.tex")


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
    macro("vOmnibusMaxP", f"{om.p.max():.0e}".replace("e-0", "\\times 10^{-") + "}")
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

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Dataset} & \textbf{All " + str(int(ct.n_all.max()))
        + r"} & \textbf{Shared module (" + str(int(ct.n_shared_module.max()))
        + r")} & \textbf{LibTorch family (" + str(int(ct.n_shared_module.max()) + 1)
        + r")} & \textbf{Share of log spread} \\",
        r"\midrule",
    ]
    for _, r in ct.iterrows():
        lines.append(
            f"{MODEL[r.model]} & {DATASET[r.dataset]} & {r.spread_all:.1f}$\\times$ & "
            f"{r.spread_shared_module:.1f}$\\times$ & "
            f"{r.spread_libtorch_family:.1f}$\\times$ & "
            f"{r.share_of_log_spread_pct:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_control.tex").write_text("\n".join(lines) + "\n")
    print("  wrote paper/generated/tab_control.tex")


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
    print("  wrote paper/generated/tab_instrument.tex")


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
    mean_j = infer.groupby("ecosystem").gpu_j.mean()
    base = mean_j["Python/PyTorch"]
    mult = (mean_j / base).sort_values()

    N_GPU, P_KW, HOURS, EUR_PER_MWH = 100_000, 0.4, 24, 100
    baseline_mwh = N_GPU * P_KW * HOURS / 1000.0
    macro("vScenarioBaselineMWh", num(baseline_mwh, 0))
    macro("vScenarioBaselineCost", f"{baseline_mwh * EUR_PER_MWH:,.0f}".replace(",", "\\,"))
    macro("vScenarioBest", SHORT[mult.index[0]])
    macro("vScenarioBestMult", num(mult.iloc[0], 2))
    macro("vScenarioWorst", SHORT[mult.index[-1]])
    macro("vScenarioWorstMult", num(mult.iloc[-1], 2))

    # How much does the boundary choice move the scenario? Reporting this is
    # the point: it is the size of the error the submitted version made.
    total_mult = (infer.groupby("ecosystem").energy_j.mean()
                  / infer.groupby("ecosystem").energy_j.mean()["Python/PyTorch"])
    spread_gpu = mult.max() / mult.min()
    spread_total = total_mult.max() / total_mult.min()
    macro("vScenarioSpreadGpu", num(spread_gpu, 1))
    macro("vScenarioSpreadTotal", num(spread_total, 1))

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Relative energy} & \textbf{Accelerators} "
        r"& \textbf{Energy (MWh/day)} & \textbf{Cost (\euro/day)} \\",
        r"\midrule",
    ]
    for eco, m in mult.items():
        # Fixed service target: a less efficient stack needs more accelerators
        # for the same work, not more power from the same ones.
        fleet = N_GPU * m
        mwh = baseline_mwh * m
        cost = mwh * EUR_PER_MWH
        name = SHORT[eco] + (" (baseline)" if eco == "Python/PyTorch" else "")
        # Thousands separators are inserted once, at the end, so that a cell
        # already carrying one is not processed twice. Two significant figures:
        # the input is five desktop runs, and the arithmetic cannot add
        # precision the measurement has not got.
        cells = [name, f"{m:.2f}$\\times$", f"{fleet:,.0f}",
                 f"{mwh:,.0f}", f"{cost:,.0f}"]
        lines.append(" & ".join(c.replace(",", "\\,") for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_industrial.tex").write_text("\n".join(lines) + "\n")
    print("  wrote paper/generated/tab_industrial.tex")


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

    collapse_mechanism_facts()

    ht = pd.read_csv(TABLES / "v2_convergence_homogeneity.csv")
    macro("vCollapseRateMin", num(ht.collapse_pct.min(), 0))
    macro("vCollapseRateMax", num(ht.collapse_pct.max(), 0))
    macro("vCollapseRateOverall", num(ht.overall_pct.iat[0], 0))
    macro("vCollapseHomogeneityP", num(ht.permutation_p.iat[0], 2))
    macro("vCollapseNeverEcosystems",
          int((ht.n_collapsed == 0).sum()))
    # The manuscript contrasts the permutation p with the chi-square one it
    # declines to use. It quoted 0.017 for the latter, which belonged to an
    # earlier version of this table; the value is 0.0065.
    macro("vCollapseChiSqP", num(ht.chi_square_p.iat[0], 4))
    macro("vCollapseChiSqMinExpected", num(ht.chi_square_min_expected.iat[0], 1))

    w = pd.read_csv(TABLES / "v2_convergence_waste.csv").iloc[0]
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
    """
    m = pd.read_csv(REPO_ROOT / "results" / "replication" / "metrics.csv.gz")
    chance = {"fashionmnist": 10.0, "cifar100": 1.0, "tinyimagenet": 0.5}
    n_classes = {"fashionmnist": 10, "cifar100": 100, "tinyimagenet": 200}
    final = m.sort_values("epoch").groupby("run").tail(1).copy()
    final["collapsed"] = final.test_acc <= final.dataset.map(chance) * 1.5
    collapsed_runs = set(final[final.collapsed].run)
    traces = m[m.run.isin(collapsed_runs)].copy()
    traces["ln_n"] = np.log(traces.dataset.map(n_classes))
    dev = (traces.train_loss - traces.ln_n).abs().max()
    macro("vCollapseLossDeviationMax", f"{dev:.4f}")

    # Do the byte-identical-module stacks agree about which runs collapse?
    shared = ["C++/LibTorch", "Python/PyTorch", "Rust/tch"]
    f = final[final.ecosystem.isin(shared)]
    per_cell = f.groupby(["model", "dataset", "repetition"]).collapsed
    discordant = int(per_cell.apply(lambda x: 0 < x.sum() < len(x)).sum())
    macro("vCollapseDiscordantCells", discordant)
    macro("vCollapseSharedStacks", len(shared))


def reexecution_facts() -> None:
    """Which runs were executed outside the main interleaved window.

    The threats section owes the reader a count and a scope, and had both wrong
    -- "the twenty Rust runs on Fashion-MNIST and Tiny ImageNet" against 25 runs
    over five configurations including VGG-16 on CIFAR-100. Read the timestamps.
    A run belongs to the later window if it started more than a day after the
    campaign's last first-window start; the two windows here are eight days
    apart, so the threshold is not delicate.
    """
    cc = pd.read_csv(REPO_ROOT / "results" / "replication" / "codecarbon.csv.gz",
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


# ----------------------------------------------------------------- ranking --
def ranking_facts() -> None:
    r = pd.read_csv(TABLES / "v2_instrument_ranking.csv")
    macro("vRankBlocks", len(r))
    macro("vRankIdentical", int(r.identical.sum()))


def main() -> None:
    epochs = design_facts()
    apparatus_facts()
    instrument_facts(epochs)
    stats = repeatability_facts()
    energy_facts(stats)
    quality_facts()
    audited_campaign_facts()
    convergence_facts()
    statistics_facts()
    instrument_table()
    industrial_scenario()
    ranking_facts()

    body = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"% every value the manuscript quotes, straight from results/revision/tables/",
    ]
    for name in sorted(macros):
        body.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    (OUT / "numbers.tex").write_text("\n".join(body) + "\n")
    print(f"  wrote paper/generated/numbers.tex ({len(macros)} macros)")


if __name__ == "__main__":
    main()
