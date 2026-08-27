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


# ------------------------------------------------------------ instrument ----
def instrument_facts(epochs: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLES / "v2_instrument_summary.csv")
    by_key = dict(zip(summary.quantity, summary.mean_ratio_or_share))
    gpu = [v for k, v in by_key.items() if k.startswith("GPU (")][0]
    cpu = [v for k, v in by_key.items() if k.startswith("CPU package")][0]
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

    gi = pd.read_csv(TABLES / "v2_coverage_gap_attribution.csv").iloc[0]
    macro("vGapInstrumentR", num(gi.pearson_r, 2))
    macro("vGapInstrumentPct", num(min(100.0, gi.share_of_gap_explained_pct), 0))
    macro("vGapMedianS", num(gi.median_gap_s, 2))
    macro("vGapVsExcessS", num(gi.median_abs_difference_s, 2))

    wm = pd.read_csv(TABLES / "v2_coverage_window_model.csv")
    best = wm.loc[wm.mean_abs_error_s.idxmin()]
    naive = wm[wm.model.str.startswith("max(")].iloc[0]
    macro("vWindowModelBestRsq", num(best.r_squared, 4))
    macro("vWindowModelBestMAE", num(best.mean_abs_error_s, 2))
    macro("vWindowModelNaiveRsq", num(naive.r_squared, 4))
    macro("vWindowModelNaiveMAE", num(naive.mean_abs_error_s, 2))
    nums = re.findall(r"[0-9.]+", best.model)
    macro("vWindowConstS", nums[0])
    macro("vWindowThresholdS", nums[1])

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
    g = stats[stats.phase == phase].copy()
    g["Ecosystem"] = g.ecosystem.map(SHORT)
    g["block"] = g.model.map(MODEL) + " / " + g.dataset.map(DATASET)
    pivot = g.pivot_table(index="Ecosystem", columns="block", values="mean_energy_J")
    pivot = pivot.sort_values(pivot.columns[0])
    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{l" + "r" * len(pivot.columns) + "}",
        r"\toprule",
        "\\textbf{Ecosystem} & "
        + " & ".join(f"\\textbf{{{c}}}" for c in pivot.columns)
        + r" \\",
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
def quality_facts() -> None:
    q = pd.read_csv(TABLES / "v2_quality_normalised.csv")
    summ = q.groupby(["ecosystem", "dataset"]).final_test_acc_pct.mean().reset_index()
    for ds, tag in (("fashionmnist", "Fashion"), ("cifar100", "Cifar"),
                    ("tinyimagenet", "Tiny")):
        g = summ[summ.dataset == ds]
        if g.empty:
            continue
        macro(f"vAccMin{tag}", num(g.final_test_acc_pct.min(), 1))
        macro(f"vAccMax{tag}", num(g.final_test_acc_pct.max(), 1))
        macro(f"vAccSpread{tag}",
              num(g.final_test_acc_pct.max() - g.final_test_acc_pct.min(), 1))

    # accuracy per kJ: the quality-normalised ranking the fixed-budget one hides
    eff = q.groupby("ecosystem").acc_per_kJ.mean().sort_values(ascending=False)
    macro("vEffBest", SHORT[eff.index[0]])
    macro("vEffWorst", SHORT[eff.index[-1]])
    macro("vEffRatio", num(eff.iloc[0] / eff.iloc[-1], 1))

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Ecosystem} & \textbf{Fashion-MNIST} & \textbf{CIFAR-100} "
        r"& \textbf{Tiny ImageNet} \\",
        r"\midrule",
    ]
    piv = summ.pivot(index="ecosystem", columns="dataset", values="final_test_acc_pct")
    for eco, row in piv.iterrows():
        vals = [row.get(d, np.nan) for d in ("fashionmnist", "cifar100", "tinyimagenet")]
        cells = [f"{v:.2f}" if pd.notna(v) else "--" for v in vals]
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
    macro("vPairsLargePct",
          num(100 * (pw.magnitude == "large").mean(), 0))

    pc = pd.read_csv(TABLES / "v2_stats_phase_consistency.csv")
    macro("vPhaseRhoMin", num(pc.spearman_rho.min(), 2))
    macro("vPhaseRhoMax", num(pc.spearman_rho.max(), 2))
    macro("vPhaseSameBest", int(pc.same_best.sum()))
    macro("vPhaseBlocks", len(pc))

    ct = pd.read_csv(TABLES / "v2_stats_libtorch_control.csv")
    macro("vCtrlSpreadMin", num(ct.spread_libtorch.min(), 1))
    macro("vCtrlSpreadMax", num(ct.spread_libtorch.max(), 1))
    macro("vCtrlShareMin", num(ct.share_of_log_spread_pct.min(), 0))
    macro("vCtrlShareMax", num(ct.share_of_log_spread_pct.max(), 0))

    lines = [
        r"% generated by results/analysis/12_paper_numbers.py -- do not edit",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Dataset} & \textbf{All " + str(int(ct.n_all.max()))
        + r"} & \textbf{LibTorch group} & \textbf{Share of log spread} \\",
        r"\midrule",
    ]
    for _, r in ct.iterrows():
        lines.append(
            f"{MODEL[r.model]} & {DATASET[r.dataset]} & {r.spread_all:.1f}$\\times$ & "
            f"{r.spread_libtorch:.1f}$\\times$ & {r.share_of_log_spread_pct:.0f}\\% \\\\")
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
        r"\textbf{Quantity} & \textbf{Mean} & \textbf{5th pct} & \textbf{95th pct} \\",
        r"\midrule",
    ]
    for _, r in summ.iterrows():
        q = r.quantity.replace("&", r"\&")
        lines.append(f"{q} & {r.mean_ratio_or_share:.3f} & {r.p05:.3f} & {r.p95:.3f} \\\\")
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
        r"\textbf{Ecosystem} & \textbf{Relative energy} & \textbf{Energy (MWh/day)} "
        r"& \textbf{Cost (\euro/day)} & \textbf{Saving vs PyTorch (\euro/day)} \\",
        r"\midrule",
    ]
    for eco, m in mult.items():
        mwh = baseline_mwh * m
        cost = mwh * EUR_PER_MWH
        saving = (baseline_mwh - mwh) * EUR_PER_MWH
        name = SHORT[eco] + (" (baseline)" if eco == "Python/PyTorch" else "")
        d = "--" if eco == "Python/PyTorch" else f"{saving:+,.0f}"
        # Thousands separators are inserted once, at the end, so that a cell
        # already carrying one is not processed twice.
        cells = [name, f"{m:.2f}$\\times$", f"{mwh:,.0f}", f"{cost:,.0f}", d]
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

    ht = pd.read_csv(TABLES / "v2_convergence_homogeneity.csv")
    macro("vCollapseRateMin", num(ht.collapse_pct.min(), 0))
    macro("vCollapseRateMax", num(ht.collapse_pct.max(), 0))
    macro("vCollapseRateOverall", num(ht.overall_pct.iat[0], 0))
    macro("vCollapseHomogeneityP", num(ht.permutation_p.iat[0], 2))
    macro("vCollapseNeverEcosystems",
          int((ht.n_collapsed == 0).sum()))

    w = pd.read_csv(TABLES / "v2_convergence_waste.csv").iloc[0]
    macro("vWastedPct", num(w.wasted_pct, 1))
    macro("vWastedMJ", num(w.collapsed_energy_MJ, 1))


# ----------------------------------------------------------------- ranking --
def ranking_facts() -> None:
    r = pd.read_csv(TABLES / "v2_instrument_ranking.csv")
    macro("vRankBlocks", len(r))
    macro("vRankIdentical", int(r.identical.sum()))


def main() -> None:
    epochs = design_facts()
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
