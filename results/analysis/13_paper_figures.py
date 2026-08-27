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
# One colour per ecosystem, held constant across every figure.
COLOR = {
    "JAX": "#1b7837", "PyTorch": "#4d9221", "C++": "#2166ac",
    "Rust": "#d6604d", "TensorFlow": "#8c6bb1", "Java": "#b2182b",
    "R": "#762a83",
}
DATASET = {"fashionmnist": "Fashion-MNIST", "cifar100": "CIFAR-100",
           "tinyimagenet": "Tiny ImageNet"}
MODEL = {"resnet18": "ResNet-18", "vgg16": "VGG-16"}


def save(fig, stem: str) -> None:
    path = FIGDIR / f"{stem}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote paper/figures/{stem}.png")


# --------------------------------------------------------------------------
def fig_window_floor(epochs: pd.DataFrame) -> None:
    """CodeCarbon's reported duration against the counter-bracketed phase."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    for eco, g in epochs.groupby("ecosystem"):
        name = SHORT[eco]
        ax.scatter(g.duration_hw_s, g.duration_cc_s, s=4, alpha=0.35,
                   color=COLOR[name], label=name, linewidths=0)

    lim = (0.05, max(epochs.duration_hw_s.max(), epochs.duration_cc_s.max()) * 1.6)
    x = np.geomspace(*lim, 200)
    floor = epochs[epochs.duration_hw_s < 1.0].duration_cc_s.median()
    ax.plot(x, x, color="0.35", lw=1.1, ls="--", zorder=5)
    ax.plot(x, np.maximum(x, floor), color="black", lw=1.6, zorder=6)
    ax.axhline(floor, color="black", lw=0.6, ls=":", zorder=4)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="phase duration, counter-bracketed (s)",
           ylabel="duration reported by CodeCarbon (s)")
    ax.annotate(f"reported $=\\max(\\mathrm{{phase}},\\ {floor:.1f}\\,$s$)$",
                xy=(0.11, floor * 1.25), fontsize=8.5)
    ax.annotate("identity", xy=(lim[1] * 0.28, lim[1] * 0.42), fontsize=8.5,
                color="0.35", rotation=34)
    ax.legend(markerscale=3, fontsize=7.5, loc="lower right", ncol=2)
    ax.set_title("(a) the reported window has a floor", fontsize=9.5, loc="left")

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
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("(b) and so power is understated for short phases",
                  fontsize=9.5, loc="left")
    save(fig, "fig_window_floor")


# --------------------------------------------------------------------------
def fig_energy_ci(stats: pd.DataFrame) -> None:
    """Training energy per epoch with between-run 95% intervals."""
    train = stats[stats.phase == "Training"].copy()
    train["name"] = train.ecosystem.map(SHORT)
    blocks = [(m, d) for m in ("resnet18", "vgg16")
              for d in ("fashionmnist", "cifar100", "tinyimagenet")]

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.6), sharey="row")
    for ax, (model, dataset) in zip(axes.ravel(), blocks):
        g = train[(train.model == model) & (train.dataset == dataset)]
        g = g.sort_values("mean_energy_J")
        y = np.arange(len(g))
        err = np.vstack([g.mean_energy_J - g.ci95_lo_J, g.ci95_hi_J - g.mean_energy_J])
        ax.barh(y, g.mean_energy_J, xerr=err, height=0.66,
                color=[COLOR[n] for n in g.name],
                error_kw={"ecolor": "0.25", "lw": 1.0, "capsize": 2.5})
        ax.set(yticks=y, yticklabels=g.name, xscale="log")
        ax.invert_yaxis()
        ax.set_title(f"{MODEL[model]} / {DATASET[dataset]}", fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        if len(g):
            ax.set_xlim(g.mean_energy_J.min() * 0.55, g.mean_energy_J.max() * 2.4)
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
        for name, gg in g.groupby("name"):
            ax.scatter(gg.train_energy_total_J / 1000.0, gg.final_test_acc_pct,
                       s=26, color=COLOR[name], label=name, alpha=0.85,
                       edgecolors="white", linewidths=0.5)
        # the frontier: nothing to the left of it reaches the same accuracy
        pts = g[["train_energy_total_J", "final_test_acc_pct"]].dropna()
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
    axes[0].set_ylabel("final test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    order = np.argsort(labels)
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=7, fontsize=8.5, bbox_to_anchor=(0.5, -0.09))
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


def main() -> None:
    epochs = pd.read_csv(TABLES / "v2_instrument_epochs.csv")
    stats = pd.read_csv(TABLES / "v2_between_run_statistics.csv")
    quality = pd.read_csv(TABLES / "v2_quality_normalised.csv")

    fig_window_floor(epochs)
    fig_energy_ci(stats)
    fig_energy_accuracy(quality)
    fig_instrument(epochs)
    fig_repeatability(stats)


if __name__ == "__main__":
    main()
