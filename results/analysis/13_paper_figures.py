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
                        duration is max(phase, ~4 s), which is why the
                        submitted energy-versus-time analysis cannot stand
  fig_energy_ci         energy per ecosystem with genuine between-run
                        intervals, per block, on a common boundary
  fig_energy_accuracy   what the fixed-budget design hides: energy spent
                        against accuracy reached
  fig_instrument        the two instruments agree on energy and disagree on
                        the window, as a function of phase length
  fig_repeatability     between-run coefficient of variation, the quantity
                        the submitted design could not estimate at all
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT  # noqa: E402

TABLES = REPO_ROOT / "results" / "revision" / "tables"
FIGDIR = REPO_ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

SHORT = {
    "Python/PyTorch": "PyTorch", "Python/TensorFlow": "TensorFlow",
    "Python/JAX": "JAX", "Cpp/LibTorch": "C++", "C++/LibTorch": "C++",
    "Java/DL4J": "Java", "R/torch": "R", "Rust/tch": "Rust",
}
# One colour per ecosystem, held constant across every figure. Chosen so that
# no two are confusable in greyscale or with the common forms of colour
# blindness: the first draft put JAX and PyTorch in two greens and R and
# TensorFlow in two purples, which made the scatter plots unreadable.
COLOR = {
    "JAX": "#1b7837",         # green
    "PyTorch": "#f28e2b",     # orange
    "C++": "#2166ac",         # blue
    "Rust": "#8c510a",        # brown
    "TensorFlow": "#7570b3",  # violet
    "Java": "#d62728",        # red
    "R": "#17becf",           # cyan
}
MARKER = {"JAX": "o", "PyTorch": "s", "C++": "^", "Rust": "D",
          "TensorFlow": "v", "Java": "P", "R": "X"}
DATASET = {"fashionmnist": "Fashion-MNIST", "cifar100": "CIFAR-100",
           "tinyimagenet": "Tiny ImageNet"}
MODEL = {"resnet18": "ResNet-18", "vgg16": "VGG-16"}


def tidy_log_x(ax) -> None:
    """Decade labels only.

    Matplotlib labels log minor ticks when a scatter spans less than a decade,
    and the labels then overprint each other into an unreadable smear.
    """
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, stem: str) -> None:
    path = FIGDIR / f"{stem}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote paper/figures/{stem}.png")


# --------------------------------------------------------------------------
def fig_window_floor(epochs: pd.DataFrame) -> None:
    """CodeCarbon's reported duration against the counter-bracketed phase."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0),
                                  gridspec_kw={"width_ratios": [1.15, 1],
                                               "wspace": 0.26})

    for eco, g in epochs.groupby("ecosystem"):
        name = SHORT[eco]
        ax.scatter(g.duration_hw_s, g.duration_cc_s, s=7, alpha=0.45,
                   color=COLOR[name], label=name, marker=MARKER[name],
                   linewidths=0)

    lim = (0.05, max(epochs.duration_hw_s.max(), epochs.duration_cc_s.max()) * 1.6)
    x = np.geomspace(*lim, 400)

    # The excess is bimodal -- a constant below a threshold, essentially nothing
    # above it -- so the model is two regimes, not a floor. Fitting a floor to
    # this gives a high R^2 carried entirely by the ends of the range; see
    # 16_coverage_sensitivity.py.
    d = epochs.assign(excess=epochs.duration_cc_s - epochs.duration_hw_s)
    const = float(d[d.duration_hw_s < 8].excess.median())
    by_len = (d.assign(b=pd.cut(d.duration_hw_s, np.arange(0, 30, 1)))
              .groupby("b", observed=True).excess.median())
    collapsed = by_len[by_len < 1.0]
    thr = float(collapsed.index[0].left) if len(collapsed) else 11.0

    ax.plot(x, x, color="0.35", lw=1.1, ls="--", zorder=5)
    ax.plot(x, np.where(x < thr, x + const, x), color="black", lw=1.7, zorder=6)
    ax.axvline(thr, color="black", lw=0.6, ls=":", zorder=4)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="phase duration, counter-bracketed (s)",
           ylabel="duration reported by CodeCarbon (s)")
    ax.annotate(f"reported $=$ phase $+\\ {const:.2f}\\,$s",
                xy=(0.55, 0.55 + const), xytext=(0.075, lim[1] * 0.34),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.annotate(f"{thr:.0f} s", xy=(thr * 1.12, 0.13), fontsize=8, color="0.35")
    ax.annotate("above: reported $=$ phase", xy=(thr * 3.0, thr * 3.0),
                xytext=(thr * 0.30, thr * 12), fontsize=8.5, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8))
    ax.legend(markerscale=2.6, fontsize=7, loc="lower right", ncol=2,
              handletextpad=0.3, columnspacing=0.9, borderpad=0.3)
    ax.set_title("(a) below a threshold, the reported window is the phase plus a constant",
                 fontsize=8.5, loc="left")

    # -- what that does to power
    d = epochs.copy()
    d["p_hw"] = d.hw_meas_j / d.duration_hw_s
    d["p_cc"] = d.cc_total_j / d.duration_cc_s
    d["bin"] = pd.cut(d.duration_hw_s, [0, 1, 3, 10, 30, np.inf],
                      labels=["<1", "1-3", "3-10", "10-30", ">30"])
    t = d.groupby("bin", observed=True)[["p_hw", "p_cc"]].mean()
    idx = np.arange(len(t))
    ax2.bar(idx - 0.19, t.p_hw, 0.38, color="#2166ac", label="NVML + RAPL counters")
    ax2.bar(idx + 0.19, t.p_cc, 0.38, color="#d6a000", label="CodeCarbon energy / duration")
    for i, (a, b) in enumerate(zip(t.p_hw, t.p_cc)):
        ax2.annotate(f"{a / b:.1f}$\\times$", xy=(i, max(a, b) + 9),
                     ha="center", fontsize=8)
    ax2.set(xticks=idx, xticklabels=t.index, xlabel="phase duration (s)",
            ylabel="mean power (W)", ylim=(0, max(t.p_hw.max(), t.p_cc.max()) * 1.22))
    ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.005, 0.87))
    ax2.set_title("(b) so power is understated for short phases",
                  fontsize=8.5, loc="left")
    save(fig, "fig_window_floor")


# --------------------------------------------------------------------------
def fig_energy_ci(stats: pd.DataFrame) -> None:
    """Training energy per epoch with between-run 95% intervals."""
    train = stats[stats.phase == "Training"].copy()
    train["name"] = train.ecosystem.map(SHORT)
    blocks = [(m, d) for m in ("resnet18", "vgg16")
              for d in ("fashionmnist", "cifar100", "tinyimagenet")]

    # One ecosystem order for all six panels, cheapest overall first. Sorting
    # each panel independently while sharing the y axis silently mislabels every
    # panel but the first -- the bars move and the tick labels do not.
    order = (train.groupby("name").mean_energy_J.median()
             .sort_values().index.tolist())
    pos = {name: i for i, name in enumerate(order)}

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), sharey=True)
    for ax, (model, dataset) in zip(axes.ravel(), blocks):
        g = train[(train.model == model) & (train.dataset == dataset)]
        y = [pos[n] for n in g.name]
        err = np.vstack([g.mean_energy_J - g.ci95_lo_J, g.ci95_hi_J - g.mean_energy_J])
        ax.barh(y, g.mean_energy_J, xerr=err, height=0.7,
                color=[COLOR[n] for n in g.name],
                error_kw={"ecolor": "0.2", "lw": 1.0, "capsize": 2.5})
        ax.set(yticks=range(len(order)), yticklabels=order, xscale="log",
               ylim=(len(order) - 0.5, -0.5))
        ax.set_title(f"{MODEL[model]} / {DATASET[dataset]}", fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        missing = [n for n in order if n not in set(g.name)]
        for n in missing:
            ax.annotate("not measured", xy=(0.02, pos[n]),
                        xycoords=("axes fraction", "data"),
                        va="center", fontsize=7, color="0.5", style="italic")
        if len(g):
            ax.set_xlim(g.mean_energy_J.min() * 0.5, g.mean_energy_J.max() * 2.6)
        tidy_log_x(ax)
    for ax in axes[1]:
        ax.set_xlabel("training energy per epoch (J, log scale)")
    fig.suptitle("GPU + CPU package, one boundary for every ecosystem; "
                 "bars are 95% between-run intervals over five independent runs",
                 fontsize=9, y=1.005)
    save(fig, "fig_energy_ci")


# --------------------------------------------------------------------------
def fig_energy_accuracy(quality: pd.DataFrame) -> None:
    """Energy spent against accuracy reached -- what a fixed budget hides."""
    q = quality.copy()
    q["name"] = q.ecosystem.map(SHORT)
    datasets = ["fashionmnist", "cifar100", "tinyimagenet"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, ds in zip(axes, datasets):
        g = q[q.dataset == ds]
        # Colour and marker carry the ecosystem; fill carries the architecture,
        # so a reader can see at a glance that every collapsed run is a VGG-16.
        for name, gg in g.groupby("name"):
            for model, face, edge in (("resnet18", COLOR[name], "white"),
                                      ("vgg16", "none", COLOR[name])):
                sub = gg[gg.model == model]
                if not len(sub):
                    continue
                ax.scatter(sub.train_energy_total_J / 1000.0, sub.final_test_acc_pct,
                           s=34, facecolors=face, edgecolors=edge, linewidths=0.9,
                           marker=MARKER[name], alpha=0.9,
                           label=name if model == "resnet18" else None)
        # the frontier: nothing to the left of it reaches the same accuracy
        pts = g[g.final_test_acc_pct > {"fashionmnist": 15.0, "cifar100": 1.5,
                                        "tinyimagenet": 0.75}[ds]]
        pts = pts[["train_energy_total_J", "final_test_acc_pct"]].dropna()
        pts = pts.sort_values("train_energy_total_J")
        best, fx, fy = -np.inf, [], []
        for e, a in pts.itertuples(index=False):
            if a > best:
                best = a
                fx.append(e / 1000.0)
                fy.append(a)
        ax.step(fx, fy, where="post", color="0.35", lw=1.0, ls="--", zorder=0)
        ax.set(xscale="log", xlabel="training energy (kJ, log scale)",
               title=DATASET[ds])
        ax.title.set_fontsize(9.5)
        tidy_log_x(ax)
        # Runs that collapsed to chance are not points on an energy/quality
        # trade-off: they spent the full budget and learned nothing. Mark them
        # rather than let them drag the axis down to zero.
        chance = {"fashionmnist": 10.0, "cifar100": 1.0, "tinyimagenet": 0.5}[ds]
        floor_pts = g[g.final_test_acc_pct <= chance * 1.5]
        if len(floor_pts):
            ax.axhspan(0, chance * 1.5, color="0.85", zorder=-2)
            ax.annotate(f"collapsed to chance ({len(floor_pts)} runs)",
                        xy=(0.5, 0.035), xycoords="axes fraction",
                        ha="center", fontsize=7.5, color="0.35")
    axes[0].set_ylabel("final test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    order = np.argsort(labels)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], marker="o", color="0.3", ls="none", ms=6,
                       label="ResNet-18"),
                Line2D([], [], marker="o", markerfacecolor="none",
                       markeredgecolor="0.3", color="0.3", ls="none", ms=6,
                       label="VGG-16")]
    labels += ["filled: ResNet-18", "hollow: VGG-16"]
    fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.11))
    save(fig, "fig_energy_accuracy")


# --------------------------------------------------------------------------
def fig_instrument(epochs: pd.DataFrame) -> None:
    """Where the two instruments agree, and where they cannot."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))

    d = epochs.sort_values("duration_hw_s")
    ax.scatter(d.duration_hw_s, d.ratio_meas, s=4, alpha=0.3,
               color="#2166ac", linewidths=0, label="GPU + CPU, both measured")
    ax.scatter(d.duration_hw_s, d.ratio_total, s=4, alpha=0.3,
               color="#d6604d", linewidths=0, label="CodeCarbon total, incl. modelled RAM")
    ax.axhline(1.0, color="black", lw=0.9)
    ax.set(xscale="log", xlabel="phase duration (s)",
           ylabel="CodeCarbon / hardware counters",
           ylim=(0.9, 1.35))
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    tidy_log_x(ax)
    ax.set_title("(a) energy: agreement is not the problem", fontsize=9.5, loc="left")

    share = epochs.groupby(epochs.ecosystem.map(SHORT)).ram_share_pct.mean().sort_values()
    ax2.barh(np.arange(len(share)), share, height=0.62,
             color=[COLOR[n] for n in share.index])
    ax2.set(yticks=np.arange(len(share)), yticklabels=share.index,
            xlabel="share of the CodeCarbon total that is modelled RAM (%)")
    ax2.set_title("(b) the part no counter can confirm", fontsize=9.5, loc="left")
    save(fig, "fig_instrument")


# --------------------------------------------------------------------------
def fig_repeatability(stats: pd.DataFrame) -> None:
    """Between-run CV: the dispersion the submitted design could not see."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    order = (stats[stats.phase == "Training"]
             .groupby("ecosystem").cv_pct.median().sort_values().index)
    names = [SHORT[e] for e in order]
    width = 0.38
    for i, (phase, off, hatch) in enumerate(
            [("Training", -width / 2, None), ("Inference", width / 2, "///")]):
        vals = [stats[(stats.ecosystem == e) & (stats.phase == phase)].cv_pct.median()
                for e in order]
        ax.bar(np.arange(len(order)) + off, vals, width, label=phase,
               color=["#2166ac" if phase == "Training" else "#92c5de"] * len(order),
               hatch=hatch, edgecolor="white")
    ax.set(xticks=np.arange(len(order)), xticklabels=names,
           ylabel="between-run CV of energy (%)")
    ax.legend(fontsize=8.5)
    ax.set_title("median over the six blocks; five independent runs each",
                 fontsize=9, loc="left")
    save(fig, "fig_repeatability")


def fig_convergence(conditional: pd.DataFrame, by_eco: pd.DataFrame) -> None:
    """What the collapsed runs were hiding.

    Comparing cross-ecosystem accuracy before and after excluding the runs that
    never left chance shows that almost the entire apparent disagreement between
    stacks was a handful of VGG-16 runs that failed to train.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                                  gridspec_kw={"width_ratios": [1.1, 1]})

    c = conditional.copy()
    c["block"] = c.model.map(MODEL) + "\n" + c.dataset.map(DATASET)
    c = c.sort_values("raw_spread_pp")
    idx = np.arange(len(c))
    ax.barh(idx - 0.19, c.raw_spread_pp, 0.38, color="#bdbdbd",
            label="all runs")
    ax.barh(idx + 0.19, c.converged_spread_pp, 0.38, color="#2166ac",
            label="runs that trained")
    ax.set(yticks=idx, yticklabels=c.block,
           xlabel="cross-ecosystem accuracy spread (percentage points)")
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(a) the stacks agree, once the collapses are removed",
                 fontsize=9.5, loc="left")

    by_eco = by_eco[by_eco.dataset.isin(("cifar100", "tinyimagenet"))]
    piv = by_eco.pivot_table(index="ecosystem", columns="dataset",
                             values="n_collapsed", aggfunc="sum").fillna(0)
    piv.index = [SHORT[e] for e in piv.index]
    piv = piv.reindex(sorted(piv.index))
    cols = [c for c in ("cifar100", "tinyimagenet") if c in piv.columns]
    width = 0.8 / max(len(cols), 1)
    for k, col in enumerate(cols):
        ax2.bar(np.arange(len(piv)) + (k - (len(cols) - 1) / 2) * width,
                piv[col], width, label=DATASET[col],
                color=["#4393c3", "#b2182b"][k])
    ax2.set(xticks=np.arange(len(piv)), xticklabels=piv.index,
            ylabel="VGG-16 runs collapsed (of 5)", ylim=(0, 5.4))
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8)
    ax2.set_title("(b) every ecosystem is susceptible; VGG-16 only",
                  fontsize=9.5, loc="left")
    save(fig, "fig_convergence")


def main() -> None:
    epochs = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    stats = pd.read_csv(TABLES / "v2_between_run_statistics.csv")
    quality = pd.read_csv(TABLES / "v2_quality_normalised.csv")

    fig_window_floor(epochs)
    fig_energy_ci(stats)
    fig_energy_accuracy(quality)
    fig_instrument(epochs)
    fig_repeatability(stats)

    conditional = pd.read_csv(TABLES / "v2_convergence_conditional.csv")
    by_eco = pd.read_csv(TABLES / "v2_convergence_by_ecosystem.csv")
    fig_convergence(conditional, by_eco)


if __name__ == "__main__":
    main()
