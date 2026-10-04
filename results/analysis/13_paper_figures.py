#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the revised manuscript, drawn from the replicated campaign.

The submitted version carried four figures: two grouped bar charts of mean
energy with no dispersion shown, a scatter of energy against time, and a
heatmap. Reviewer #1 (c4, c10) and Reviewer #3 (M1, M3, M4) between them asked
for units, uncertainty, quality normalisation and a defensible frontier. None
of the four survives as drawn, so this script replaces them.

  fig_window_floor      the instrument finding: what CodeCarbon calls a
                        duration is the phase plus a constant below a
                        threshold, which is why the submitted
                        energy-versus-time analysis cannot stand. (The name is
                        older than the fit: a floor was our first model of it,
                        and the wrong one.)
  fig_energy_ci         energy per ecosystem with genuine between-run
                        intervals, per block, on a common boundary
  fig_energy_accuracy   what the fixed-budget design hides: energy spent
                        against accuracy reached
  fig_instrument        the two instruments agree on energy and disagree on
                        the window, as a function of phase length
  fig_repeatability     between-run coefficient of variation, the quantity
                        the submitted design could not estimate at all
  fig_saturation        the accelerator-saturation cell: the LibTorch
                        lineage's spread narrows at 224x224, the seven-stack
                        spread widens (from 20_saturation's tables)

Every figure is vector PDF, drawn at its printed size, in the one stack
colour-and-marker scheme of figstyle.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, TABLE_DIR, TABLES_RESOLVER,  # noqa: E402
                    announce_scope, campaign_is_partial)
import figstyle as fs  # noqa: E402
from figstyle import (COLOR, DATASET, DATASETS, MARKER, MODEL,  # noqa: E402
                      MODELS, SHORT, STACKS)

TABLES = TABLES_RESOLVER  # writes divert on a live campaign, reads fall back
# The saturation cell's own tables (20_saturation.py), a sibling directory
# named for its campaign; read only.
SAT_TABLES = TABLE_DIR.with_name("tables_campaign_saturation")
# Diverted with everything else while the campaign is incomplete: a monitoring
# run replaced six of the manuscript's figures before this was noticed, and a
# figure carries no run count on its face to give the substitution away.
FIGDIR = REPO_ROOT / "paper" / ("figures" if not campaign_is_partial()
                                else "figures_partial")
FIGDIR.mkdir(parents=True, exist_ok=True)

# Every figure is drawn at the size it is printed (figstyle), in one stack
# colour-and-marker scheme, and written as vector PDF.
fs.apply()


def save(fig, stem: str) -> None:
    path = fs.save(fig, FIGDIR / stem)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def refuse(stem: str, reason: str) -> None:
    """Draw nothing, and remove any figure of this name left by an earlier run.

    A blank plot is the figure pipeline's version of the stale-table fallback,
    and a worse one. fig_energy_accuracy drew three empty log-scaled panels from
    a zero-row frame with no error and no warning -- an empty groupby body never
    executes, an empty step plot is legal, and the legend handles are
    hard-coded, so matplotlib never complains -- and the result overwrote the
    committed figure looking exactly like the right one.

    Deleting instead makes the LaTeX build fail on \\includegraphics, which is
    the only loud channel a figure has. Only inside FIGDIR, so a partial run
    removes its own copy and never the manuscript's.
    """
    gone = [p for p in (FIGDIR / f"{stem}.pdf", FIGDIR / f"{stem}.png")
            if p.exists()]
    for path in gone:
        path.unlink()
        print(f"  REMOVED {path.relative_to(REPO_ROOT)}: {reason}")
    if not gone:
        print(f"  not drawn: {stem}.pdf -- {reason}")


def stack_rows(ax, invert: bool = True, labels: bool = True) -> dict[str, int]:
    """Stacks on the y axis in table order, first at the top."""
    pos = {s: i for i, s in enumerate(STACKS)}
    ax.set_yticks(range(len(STACKS)))
    ax.set_yticklabels(STACKS)
    if invert:
        ax.set_ylim(len(STACKS) - 0.5, -0.5)
    if not labels:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    return pos


def stack_legend(fig_or_ax, stacks=STACKS, extra=(), **kw):
    handles = [fs.stack_handle(s) for s in stacks] + list(extra)
    return fig_or_ax.legend(handles=handles, **kw)


def nice_log_ticks(lo: float, hi: float,
                   candidates=(1, 2, 5)) -> list[float]:
    out = []
    for decade in range(int(np.floor(np.log10(lo))) - 1,
                        int(np.ceil(np.log10(hi))) + 1):
        for c in candidates:
            v = c * 10.0 ** decade
            if lo <= v <= hi:
                out.append(v)
    return out


# --------------------------------------------------------------------------
def fig_window_floor(epochs: pd.DataFrame) -> None:
    """What CodeCarbon calls a duration, and what that does to derived power.

    Panel (a) used to draw the two-regime fit, complete with a threshold line
    and an annotated "reported = phase + 3.28 s". That model is wrong -- the
    excess is trimodal and block length does not decide the mode -- so drawing
    it was drawing a claim the text withdraws two paragraphs later. It shows
    the modes themselves.

    The modes are read from v2_coverage_window_model.csv, the table the
    manuscript's \\vWindowMode* macros are written from. The panel used to
    find its own, by cutting the excess at 0.5 s, 4 s and 6 s: that split the
    paper's 4.57 s mode into two (3.29 s and 4.58 s), dropped the 13.38 s block
    off the end, and printed three labels that disagreed with the three the
    text quotes for the same figure.

    Panel (b) is v2_coverage_power_distortion.csv as written -- the table
    beside it in the manuscript -- rather than a re-binning of the epochs that
    could drift from it.
    """
    fig = fs.figure(fs.TEXT_WIDTH_IN, 4.15)
    ax, ax2 = fig.subplots(2, 1, height_ratios=[1.25, 1])

    d = epochs.assign(excess=epochs.duration_cc_s - epochs.duration_hw_s,
                      name=epochs.ecosystem.map(SHORT))

    # -- (a) the excess, by ecosystem, against phase length. The point of the
    # panel is that the horizontal bands do not sort by the x axis. 12,600
    # points stay vector: the file is a few hundred kB, under the 2 MB at which
    # figstyle.save asks for the layer to be rasterised.
    for name in STACKS:
        g = d[d.name == name]
        ax.scatter(g.duration_hw_s, g.excess, s=4, marker=MARKER[name],
                   color=COLOR[name], alpha=0.55, linewidths=0,
                   zorder=2)
    modes = pd.read_csv(TABLES / "v2_coverage_window_model.csv"
                        ).sort_values("mode_excess_s")
    for r in modes.itertuples(index=False):
        ax.axhline(r.mode_excess_s, color=fs.INK_2, lw=0.6, zorder=1)
        share = (f"{int(r.n_blocks)} block" if r.n_blocks == 1
                 else f"{r.share_pct:.0f}% of blocks")
        ax.annotate(f"{r.mode_excess_s:.2f} s\n{share}",
                    xy=(1.0, r.mode_excess_s), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=fs.SMALL_PT, color=fs.INK,
                    linespacing=1.0)
    # The threshold the text quotes (\vWindowThresholdS), parsed from the
    # rejected-fits table exactly as 12_paper_numbers parses it.
    fits = pd.read_csv(TABLES / "v2_coverage_window_rejected_fits.csv")
    thresh_model = fits[~fits.model.str.startswith("max(")].model.iloc[0]
    threshold = float(re.findall(r"[0-9.]+", thresh_model)[1])
    ax.axvline(threshold, color=fs.MUTED, lw=0.6, zorder=1)
    ax.annotate(f"{threshold:g} s", xy=(threshold, 1.0),
                xycoords=("data", "axes fraction"), xytext=(2, -1),
                textcoords="offset points", ha="left", va="top",
                fontsize=fs.SMALL_PT, color=fs.INK_2)
    ax.set(xscale="log", yscale="log", ylim=(0.006, 25),
           xlabel="phase duration, counter-bracketed (s)",
           ylabel="reported window − phase (s)")
    fs.log_ticks(ax.xaxis, [0.3, 1, 3, 10, 30, 100])
    fs.log_ticks(ax.yaxis, [0.01, 0.1, 1, 10])
    ax.grid(True, which="major", axis="both")
    # The key sits above the panel, clear of the data, the mode lines and the
    # threshold line (inside the plot it crossed the 11 s line).
    stack_legend(fig, loc="outside upper center", ncol=7,
                 handletextpad=0.15, columnspacing=0.9)
    fs.panel_label(ax, "a")

    # -- (b) what that does to power, on the bins the manuscript tabulates
    t = pd.read_csv(TABLES / "v2_coverage_power_distortion.csv")
    idx = np.arange(len(t))
    w = 0.36
    ax2.bar(idx - w / 2 - 0.01, t.measured_power_w, w, color=fs.INK_2,
            label="hardware counters")
    ax2.bar(idx + w / 2 + 0.01, t.reported_power_w, w, color=fs.LIGHT,
            label="CodeCarbon, over its reported duration")
    for i, (a, b, r) in enumerate(zip(t.measured_power_w, t.reported_power_w,
                                      t.understated_by)):
        ax2.annotate(f"{r:.1f}×", xy=(i, max(a, b)), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=fs.SMALL_PT, color=fs.INK)
    bins = [str(b).replace(" s", "").replace("-", "–") for b in t["bin"]]
    ax2.set_xticks(idx, bins)
    ax2.tick_params(axis="x", length=0)
    top = float(max(t.measured_power_w.max(), t.reported_power_w.max()))
    ax2.set(xlabel="phase duration (s)", ylabel="median power (W)",
            ylim=(0, top * 1.32), xlim=(-0.6, len(t) - 0.4))
    ax2.yaxis.set_major_locator(MultipleLocator(100))
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper left", ncol=1)
    fs.panel_label(ax2, "b")
    save(fig, "fig_window_floor")


# --------------------------------------------------------------------------
def fig_energy_ci(stats: pd.DataFrame) -> None:
    """Training energy per epoch with between-run 95% intervals.

    Dots, not bars: on a logarithmic axis a bar's length from an arbitrary
    left edge means nothing. One x range for all six panels, so a position
    means the same energy in every cell, and one stack order -- the tables'.
    """
    train = stats[stats.phase == "Training"].copy()
    train["name"] = train.ecosystem.map(SHORT)

    lo = train.mean_energy_J.min() / 1.35
    hi = train.mean_energy_J.max() * 1.35
    ticks = nice_log_ticks(lo, hi)

    fig = fs.figure(fs.TEXT_WIDTH_IN, 3.25)
    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    widest = 0.0
    for r, model in enumerate(MODELS):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            pos = stack_rows(ax, labels=(c == 0))
            g = train[(train.model == model) & (train.dataset == dataset)]
            for row in g.itertuples(index=False):
                y = pos[row.name]
                ax.plot(row.mean_energy_J, y, ls="none", marker=MARKER[row.name],
                        ms=fs.ms(row.name), color=COLOR[row.name],
                        markeredgewidth=0, zorder=3)
                # The interval, drawn over the marker so that one wider than
                # the marker would show; none is.
                ax.plot([row.ci95_lo_J, row.ci95_hi_J], [y, y], color=fs.INK,
                        lw=0.8, solid_capstyle="butt", zorder=4)
                widest = max(widest, np.log10(row.ci95_hi_J / row.ci95_lo_J))
            for n in STACKS:
                if n not in set(g.name):
                    ax.annotate("not measured", xy=(0.03, pos[n]),
                                xycoords=("axes fraction", "data"), va="center",
                                fontsize=fs.SMALL_PT, color=fs.MUTED,
                                style="italic")
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            fs.log_ticks(ax.xaxis, ticks)
            ax.grid(True, axis="x")
            ax.set_title(f"{MODEL[model]} · {DATASET[dataset]}",
                         loc="left", fontsize=fs.FONT_PT, color=fs.INK)
    fig.supxlabel("training energy per epoch (J, log scale)",
                  fontsize=fs.FONT_PT)
    # The caption says every interval is narrower than its marker. Check it:
    # a marker spans ms points; the axis spans log10(hi/lo) decades over the
    # panel's width.
    fig.canvas.draw()
    width_pt = axes[0, 0].get_window_extent().width * 72 / fig.dpi
    marker_decades = fs.ms("C++") / width_pt * np.log10(hi / lo)
    verdict = "narrower" if widest < marker_decades else "WIDER"
    print(f"  fig_energy_ci: widest interval {widest:.4f} decades, marker "
          f"{marker_decades:.4f}: every interval {verdict} than its marker"
          + ("" if verdict == "narrower" else
             "  !! the caption of fig:energy_ci says otherwise"))
    save(fig, "fig_energy_ci")


# --------------------------------------------------------------------------
def fig_energy_accuracy(quality: pd.DataFrame) -> None:
    """Energy spent against accuracy reached -- what a fixed budget hides."""
    q = quality.copy()
    q["name"] = q.ecosystem.map(SHORT)
    q["kj"] = q.train_energy_total_J / 1000.0

    lo, hi = q.kj.min() / 1.3, q.kj.max() * 1.3
    fig = fs.figure(fs.TEXT_WIDTH_IN, 2.55)
    axes = fig.subplots(1, 3, sharex=True)
    for ax, ds in zip(axes, DATASETS):
        g = q[q.dataset == ds]
        # Colour and marker carry the ecosystem; fill carries the architecture,
        # so a reader can see at a glance that every collapsed run is a VGG-16.
        for name in STACKS:
            for model in MODELS:
                sub = g[(g.name == name) & (g.model == model)]
                if not len(sub):
                    continue
                filled = model == "resnet18"
                ax.plot(sub.kj, sub.final_test_acc_pct, ls="none",
                        marker=MARKER[name], ms=fs.ms(name, 4.2),
                        markerfacecolor=COLOR[name] if filled else fs.SURFACE,
                        markeredgecolor=fs.SURFACE if filled else COLOR[name],
                        markeredgewidth=0.4 if filled else 0.8, zorder=3)
        # the frontier: nothing to the left of it reaches the same accuracy
        pts = g[g.final_test_acc_pct > {"fashionmnist": 15.0, "cifar100": 1.5,
                                        "tinyimagenet": 0.75}[ds]]
        pts = pts[["kj", "final_test_acc_pct"]].dropna().sort_values("kj")
        best, fx, fy = -np.inf, [], []
        for e, a in pts.itertuples(index=False):
            if a > best:
                best = a
                fx.append(e)
                fy.append(a)
        if fx:
            fx.append(hi)
            fy.append(fy[-1])
        ax.step(fx, fy, where="post", color=fs.INK_2, lw=0.8, ls=(0, (3, 2)),
                zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        fs.log_ticks(ax.xaxis, nice_log_ticks(lo, hi), lambda v, _: f"{v:g}")
        ax.grid(True, axis="both")
        ax.set_title(DATASET[ds], loc="left", fontsize=fs.FONT_PT, color=fs.INK)
        # Runs that collapsed to chance are not points on an energy/quality
        # trade-off: they spent the full budget and learned nothing. Mark them
        # rather than let them drag the axis down to zero.
        chance = {"fashionmnist": 10.0, "cifar100": 1.0, "tinyimagenet": 0.5}[ds]
        floor_pts = g[g.final_test_acc_pct <= chance * 1.5]
        if len(floor_pts):
            ax.axhspan(0, chance * 1.5, color=fs.GRID, zorder=0)
            ax.annotate(f"collapsed to chance ({len(floor_pts)} runs)",
                        xy=(0.5, 0.035), xycoords="axes fraction",
                        ha="center", fontsize=fs.SMALL_PT, color=fs.INK_2)
    axes[0].set_ylabel("final test accuracy (%)")
    axes[1].set_xlabel("training energy per run (kJ, log scale)")
    arch = [Line2D([], [], ls="none", marker="o", ms=4.2, color=fs.INK_2,
                   label="ResNet-18"),
            Line2D([], [], ls="none", marker="o", ms=4.2, markerfacecolor="none",
                   markeredgecolor=fs.INK_2, markeredgewidth=0.8,
                   label="VGG-16"),
            Line2D([], [], color=fs.INK_2, lw=0.8, ls=(0, (3, 2)),
                   label="frontier")]
    stack_legend(fig, extra=arch, loc="outside lower center", ncol=10,
                 handletextpad=0.15, columnspacing=0.75)
    save(fig, "fig_energy_accuracy")


# --------------------------------------------------------------------------
def fig_instrument(epochs: pd.DataFrame) -> None:
    """Where the two instruments agree, and where they cannot."""
    fig = fs.figure(fs.TEXT_WIDTH_IN, 2.3)
    ax, ax2 = fig.subplots(1, 2, width_ratios=[1.45, 1])

    d = epochs.sort_values("duration_hw_s")
    # Neither series is a stack, so neither wears a stack colour.
    ax.scatter(d.duration_hw_s, d.ratio_total, s=2.5, alpha=0.35,
               color=fs.ACCENT, linewidths=0, zorder=2)
    ax.scatter(d.duration_hw_s, d.ratio_meas, s=2.5, alpha=0.35,
               color=fs.INK_2, linewidths=0, zorder=3)
    ax.axhline(1.0, color=fs.INK, lw=0.6, zorder=1)
    # Every block is on the axis: the old upper limit of 1.35 cut 20 of them.
    top = max(d.ratio_total.max(), d.ratio_meas.max())
    bottom = min(d.ratio_total.min(), d.ratio_meas.min())
    ax.set(xscale="log", xlabel="phase duration (s)",
           ylabel="CodeCarbon / hardware counters",
           ylim=(min(0.9, bottom - 0.02), top + 0.03))
    fs.log_ticks(ax.xaxis, [0.3, 1, 3, 10, 30, 100])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.grid(True, axis="both")
    handles = [Line2D([], [], ls="none", marker="o", ms=4, color=fs.ACCENT,
                      label="total, incl. modelled RAM"),
               Line2D([], [], ls="none", marker="o", ms=4, color=fs.INK_2,
                      label="GPU + CPU, both measured")]
    ax.legend(handles=handles, loc="upper right")
    fs.panel_label(ax, "a")

    share = epochs.groupby(epochs.ecosystem.map(SHORT)).ram_share_pct.mean()
    pos = stack_rows(ax2)
    for name, v in share.items():
        ax2.barh(pos[name], v, height=0.55, color=COLOR[name])
        ax2.annotate(f"{v:.1f}", xy=(v, pos[name]), xytext=(2, 0),
                     textcoords="offset points", va="center", ha="left",
                     fontsize=fs.SMALL_PT, color=fs.INK)
    ax2.set_xlim(0, share.max() * 1.18)
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    ax2.grid(True, axis="x")
    ax2.set_xlabel("modelled RAM share (%)")
    fs.panel_label(ax2, "b")
    save(fig, "fig_instrument")


# --------------------------------------------------------------------------
def fig_repeatability(stats: pd.DataFrame) -> None:
    """Between-run CV: the dispersion the submitted design could not see.

    A dot per stack and phase at the median over the six cells, and a line over
    their range, so the worst case the text quotes is on the figure too.
    Single-column width: the Elsevier build sets this figure in one column.
    """
    s = stats.assign(name=stats.ecosystem.map(SHORT))
    fig = fs.figure(fs.COLUMN_WIDTH_IN, 2.2)
    ax = fig.subplots()
    pos = stack_rows(ax)
    off = {"Training": -0.17, "Inference": 0.17}
    for name in STACKS:
        for phase in ("Training", "Inference"):
            v = s[(s.name == name) & (s.phase == phase)].cv_pct
            if v.empty:
                continue
            y = pos[name] + off[phase]
            ax.plot([v.min(), v.max()], [y, y], color=COLOR[name], lw=0.9,
                    solid_capstyle="round", zorder=2)
            filled = phase == "Training"
            ax.plot(v.median(), y, ls="none", marker=MARKER[name],
                    ms=fs.ms(name, 4.2),
                    markerfacecolor=COLOR[name] if filled else fs.SURFACE,
                    markeredgecolor=COLOR[name], markeredgewidth=0.8, zorder=3)
    ax.set_xlim(0, np.ceil(s.cv_pct.max() * 2) / 2 + 0.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, axis="x")
    ax.set_xlabel("between-run CV of energy (%)")
    handles = [Line2D([], [], ls="none", marker="o", ms=4.2, color=fs.INK_2,
                      label="training"),
               Line2D([], [], ls="none", marker="o", ms=4.2,
                      markerfacecolor=fs.SURFACE, markeredgecolor=fs.INK_2,
                      markeredgewidth=0.8, label="inference"),
               Line2D([], [], color=fs.INK_2, lw=0.9,
                      label="range over six cells")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               columnspacing=1.0)
    save(fig, "fig_repeatability")


# --------------------------------------------------------------------------
def fig_convergence(conditional: pd.DataFrame, by_eco: pd.DataFrame,
                    stem: str = "fig_convergence") -> None:
    """What the collapsed runs were hiding.

    Comparing cross-ecosystem accuracy before and after excluding the runs that
    never left chance shows that almost the entire apparent disagreement between
    stacks was a handful of VGG-16 runs that failed to train.

    The two panel titles used to be derived from the data ("4 of 7
    ecosystems") because literals had gone stale twice. The figure carries no
    titles now; the caption states what each panel shows, and the counts are
    the marks.
    """
    fig = fs.figure(fs.TEXT_WIDTH_IN, 2.25)
    ax, ax2 = fig.subplots(1, 2, width_ratios=[1.2, 1])

    # -- (a) a dumbbell per cell: all runs (hollow) to runs that trained (filled)
    c = conditional.copy()
    c["k"] = c.model.map(MODELS.index) * 10 + c.dataset.map(DATASETS.index)
    c = c.sort_values("k").reset_index(drop=True)
    labels = (c.model.map(MODEL) + " · " + c.dataset.map(DATASET)).tolist()
    for i, r in c.iterrows():
        ax.plot([r.raw_spread_pp, r.converged_spread_pp], [i, i],
                color=fs.LIGHT, lw=1.6, solid_capstyle="butt", zorder=1)
        ax.plot(r.raw_spread_pp, i, ls="none", marker="o", ms=4.6,
                markerfacecolor=fs.SURFACE, markeredgecolor=fs.INK_2,
                markeredgewidth=0.9, zorder=2)
        ax.plot(r.converged_spread_pp, i, ls="none", marker="o", ms=4.0,
                color=fs.INK_2, zorder=3)
    ax.set_yticks(range(len(c)), labels)
    ax.set_ylim(len(c) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, np.ceil(c.raw_spread_pp.max() / 5) * 5 + 1)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, axis="x")
    ax.set_xlabel("cross-ecosystem accuracy spread (pp)")
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=4.6, markerfacecolor=fs.SURFACE,
               markeredgecolor=fs.INK_2, markeredgewidth=0.9, label="all runs"),
        Line2D([], [], ls="none", marker="o", ms=4.0, color=fs.INK_2,
               label="runs that trained")], loc="center right")
    fs.panel_label(ax, "a")

    # -- (b) collapsed VGG-16 runs per stack, CIFAR-100 filled, Tiny hollow
    b = by_eco[by_eco.dataset.isin(("cifar100", "tinyimagenet"))]
    piv = b.pivot_table(index="ecosystem", columns="dataset",
                        values="n_collapsed", aggfunc="sum").fillna(0)
    piv.index = [SHORT[e] for e in piv.index]
    n_runs = int(b.n_runs.max()) if "n_runs" in b else 5
    pos = stack_rows(ax2)
    off = {"cifar100": -0.17, "tinyimagenet": 0.17}
    for name in STACKS:
        if name not in piv.index:
            continue
        for ds in ("cifar100", "tinyimagenet"):
            if ds not in piv.columns:
                continue
            v = float(piv.loc[name, ds])
            y = pos[name] + off[ds]
            ax2.plot([0, v], [y, y], color=COLOR[name], lw=0.9, zorder=2)
            filled = ds == "cifar100"
            ax2.plot(v, y, ls="none", marker=MARKER[name], ms=fs.ms(name, 4.2),
                     markerfacecolor=COLOR[name] if filled else fs.SURFACE,
                     markeredgecolor=COLOR[name], markeredgewidth=0.8, zorder=3)
    ax2.set_xlim(-0.25, n_runs + 0.25)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, axis="x")
    ax2.set_xlabel(f"VGG-16 runs collapsed (of {n_runs})")
    ax2.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=4.2, color=fs.INK_2,
               label=DATASET["cifar100"]),
        Line2D([], [], ls="none", marker="o", ms=4.2, markerfacecolor=fs.SURFACE,
               markeredgecolor=fs.INK_2, markeredgewidth=0.8,
               label=DATASET["tinyimagenet"])], loc="upper right")
    fs.panel_label(ax2, "b")
    save(fig, stem)


# --------------------------------------------------------------------------
def fig_saturation(rel: pd.DataFrame) -> None:
    """The saturation cell's central result: one spread narrows, one widens.

    Per architecture, each stack's per-epoch training energy as a multiple of
    the cheapest of the seven in the same regime: at 32x32 the mean over the
    three datasets (hollow, with their range), at 224x224 the cell (filled).
    On a log axis the distance between two marks is the spread between the
    two stacks, so the LibTorch lineage's spread is R's distance from C++ and
    the seven-stack spread is the rightmost mark's distance from 1x.

    Data: sat_relative_energy.csv, written by 20_saturation.relative_energy on
    sat_spread's definition; the spreads annotated on each panel are computed
    here from that table and agree with sat_spread and tab_saturation.
    """
    t = rel[rel.phase == "Training"].assign(name=rel.ecosystem.map(SHORT))
    d32 = [c for c in t.columns if c.startswith("rel_32_")
           and c.split("rel_32_")[1] in DATASETS]
    lineage = fs.LIBTORCH_LINEAGE

    def spread(frame, cols, names=None):
        f = frame if names is None else frame[frame.name.isin(names)]
        vals = [f[c].max() / f[c].min() for c in cols]
        return min(vals), max(vals)

    def fmt(lo_hi):
        lo, hi = lo_hi
        return (f"{lo:.1f}×" if f"{lo:.1f}" == f"{hi:.1f}"
                else f"{lo:.1f}–{hi:.1f}×")

    hi = max(t.rel_224.max(), t.rel_32_max.max()) * 1.4
    fig = fs.figure(fs.TEXT_WIDTH_IN, 2.45)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for k, (ax, model) in enumerate(zip(axes, MODELS)):
        g = t[t.model == model].set_index("name")
        pos = stack_rows(ax, labels=(k == 0))
        for name in lineage:
            ax.axhspan(pos[name] - 0.5, pos[name] + 0.5, color=fs.BAND,
                       lw=0, zorder=0)
        for name in STACKS:
            if name not in g.index:
                continue
            r, y = g.loc[name], pos[name]
            ax.plot([r.rel_32_mean, r.rel_224], [y, y], color=fs.LIGHT,
                    lw=1.4, solid_capstyle="butt", zorder=1)
            ax.plot([r.rel_32_min, r.rel_32_max], [y, y], color=COLOR[name],
                    lw=0.9, zorder=2)
            ax.plot(r.rel_32_mean, y, ls="none", marker=MARKER[name],
                    ms=fs.ms(name, 4.6), markerfacecolor=fs.SURFACE,
                    markeredgecolor=COLOR[name], markeredgewidth=0.9, zorder=3)
            ax.plot(r.rel_224, y, ls="none", marker=MARKER[name],
                    ms=fs.ms(name, 5.2), color=COLOR[name],
                    markeredgewidth=0, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.85, hi)
        fs.log_ticks(ax.xaxis, nice_log_ticks(1, hi), fs.times)
        ax.grid(True, axis="x")
        ax.set_title(MODEL[model], loc="left", fontsize=fs.FONT_PT, color=fs.INK)
        note = ("spread, 32×32 → 224×224\n"
                f"LibTorch lineage: {fmt(spread(g.reset_index(), d32, lineage))}"
                f" → {fmt(spread(g.reset_index(), ['rel_224'], lineage))}\n"
                f"all seven: {fmt(spread(g.reset_index(), d32))}"
                f" → {fmt(spread(g.reset_index(), ['rel_224']))}")
        ax.annotate(note, xy=(0.98, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=fs.SMALL_PT, color=fs.INK,
                    linespacing=1.25)
    fig.supxlabel("training energy per epoch, relative to the cheapest stack "
                  "(log scale)", fontsize=fs.FONT_PT)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=4.6, markerfacecolor=fs.SURFACE,
               markeredgecolor=fs.INK_2, markeredgewidth=0.9,
               label="32×32, mean of three datasets (line: range)"),
        Line2D([], [], ls="none", marker="o", ms=5.2, color=fs.INK_2,
               label="224×224 cell"),
        Patch(facecolor=fs.BAND, edgecolor="none", label="LibTorch lineage")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    save(fig, "fig_saturation")


def draw(stem: str, fn, **inputs) -> None:
    """Draw one figure, or refuse it when a table it reads has no rows."""
    empty = [name for name, frame in inputs.items() if frame.empty]
    if empty:
        return refuse(stem, f"{' and '.join(empty)} has no rows")
    fn()


def first_campaign_figure() -> None:
    """The collapse figure for the campaign that has the collapses.

    The second campaign has none, so panel (b) is seven bars at zero and panel
    (a) compares a set of runs with itself. The finding is real and belongs to
    the first campaign; the manuscript now says so, and this draws it from that
    campaign's tables under its own name. Never the v2 name: a figure carries no
    campaign on its face, which is the whole reason 23 of the revision log
    exists.
    """
    conditional = TABLES / "v1_convergence_conditional.csv"
    by_eco = TABLES / "v1_convergence_by_ecosystem.csv"
    if not (conditional.exists() and by_eco.exists()):
        return refuse("fig_convergence_first_campaign",
                      "v1_convergence_* not built; run "
                      "15_convergence.py --campaign v1")
    draw("fig_convergence_first_campaign",
         lambda: fig_convergence(pd.read_csv(conditional), pd.read_csv(by_eco),
                                 stem="fig_convergence_first_campaign"),
         v1_convergence_conditional=pd.read_csv(conditional),
         v1_convergence_by_ecosystem=pd.read_csv(by_eco))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--campaign", choices=("v1", "v2"), default="v2",
                    help="v2 (default) draws every figure the paper uses; v1 "
                         "draws only the collapse figure, from the superseded "
                         "campaign's tables, as fig_convergence_first_campaign")
    if ap.parse_args(argv).campaign == "v1":
        first_campaign_figure()
        return

    epochs = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    stats = pd.read_csv(TABLES / "v2_between_run_statistics.csv")
    quality = pd.read_csv(TABLES / "v2_quality_normalised.csv")
    conditional = pd.read_csv(TABLES / "v2_convergence_conditional.csv")
    by_eco = pd.read_csv(TABLES / "v2_convergence_by_ecosystem.csv")

    # The emptiness check is here rather than inside each figure because no
    # figure in this file raises on a zero-row frame -- an empty groupby body
    # never executes, an empty step plot is legal -- so an empty campaign would
    # otherwise produce a complete set of blank figures under the right
    # filenames. One guard, over every figure, naming the table that is empty.
    draw("fig_window_floor", lambda: fig_window_floor(epochs),
         v2_instrument_epochs=epochs)
    draw("fig_energy_ci", lambda: fig_energy_ci(stats),
         v2_between_run_statistics=stats)
    draw("fig_energy_accuracy", lambda: fig_energy_accuracy(quality),
         v2_quality_normalised=quality)
    draw("fig_instrument", lambda: fig_instrument(epochs),
         v2_instrument_epochs=epochs)
    draw("fig_repeatability", lambda: fig_repeatability(stats),
         v2_between_run_statistics=stats)
    # The saturation cell is a campaign of its own, analysed by 20_saturation;
    # without its table there is no figure, and the build says so.
    rel_path = SAT_TABLES / "sat_relative_energy.csv"
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        draw("fig_saturation", lambda: fig_saturation(rel),
             sat_relative_energy=rel)
    else:
        refuse("fig_saturation", f"{rel_path.relative_to(REPO_ROOT)} not "
               "built; run 20_saturation.py")
    # With no collapse anywhere in the campaign this figure has nothing to
    # draw: panel (a) plots one set of runs against itself and panel (b) is an
    # empty axis with a legend. The manuscript states the zero in text and
    # carries the first campaign's figure instead. The code path stays, because
    # a campaign that does collapse needs the figure -- it is the branch that is
    # skipped, not the plot.
    collapsed = int(by_eco[by_eco.dataset.isin(("cifar100", "tinyimagenet"))]
                    .n_collapsed.sum()) if not by_eco.empty else -1
    # -1 is the empty-table case, which draw() refuses with its own reason:
    # "no collapses" and "no data" must not print the same message.
    if collapsed != 0:
        draw("fig_convergence", lambda: fig_convergence(conditional, by_eco),
             v2_convergence_conditional=conditional,
             v2_convergence_by_ecosystem=by_eco)
    else:
        refuse("fig_convergence",
               "no run in this campaign collapsed, so panel (a) would compare "
               "one set of runs with itself and panel (b) would be empty; the "
               "manuscript reports the zero in text and shows "
               "fig_convergence_first_campaign")


if __name__ == "__main__":
    main()
