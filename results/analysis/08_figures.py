#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected figures.

Addresses:
  R1 c4  / R3 M1 -- one unit (Joules) everywhere, no kWh values under a J label
  R3 minor       -- per model/dataset panels instead of one aggregated bar
  R3 M5          -- absolute scale for the dataset comparison
  R3 M4 / R1 c10 -- Pareto frontier instead of qualitative quadrants
  R1 c15         -- explicit GPU/host energy split
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG_DIR, GPU_TDP_W, ensure_dirs, load, order_ecosystems

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
    }
)

PHASE_COLOR = {"Training": "#2f6f4e", "Inference": "#8a5a2b"}


def _order(df: pd.DataFrame) -> list[str]:
    return order_ecosystems(df["ecosystem"].unique())


def fig_component_stack(df: pd.DataFrame) -> None:
    """GPU vs host energy per ecosystem -- the core of R1 comment 15."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, phase in zip(axes, ["Training", "Inference"]):
        sub = df[df["phase"] == phase]
        g = sub.groupby("ecosystem", observed=True)
        n = g.size()
        comp = pd.DataFrame(
            {
                "GPU": g["gpu_energy_j"].sum() / n,
                "CPU": g["cpu_energy_j"].sum() / n,
                "RAM (modelled)": g["ram_energy_j"].sum() / n,
            }
        )
        comp = comp.loc[comp.sum(axis=1).sort_values().index]
        bottom = np.zeros(len(comp))
        for col, c in zip(comp.columns, ["#2b6cb0", "#c05621", "#a0aec0"]):
            ax.bar(comp.index, comp[col], bottom=bottom, label=col, color=c, edgecolor="white", linewidth=0.5)
            bottom += comp[col].to_numpy()
        ax.set_title(f"{phase}: energy per epoch by component")
        ax.set_ylabel("Energy [J]")
        ax.tick_params(axis="x", rotation=40)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.legend(fontsize=8)
    fig.suptitle("Measured energy is dominated by host-side terms, not by the GPU", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_component_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def fig_panels(df: pd.DataFrame) -> None:
    """Per model x dataset panels (R3 minor: Figures 1 and 2 aggregate too much)."""
    for phase in ["Training", "Inference"]:
        sub = df[df["phase"] == phase]
        models = sorted(sub["model_arch"].unique())
        datasets = ["FashionMNIST", "CIFAR100", "TinyImageNet"]
        fig, axes = plt.subplots(len(models), len(datasets), figsize=(12, 6), sharey="row")
        order = _order(sub)
        for i, m in enumerate(models):
            for j, d in enumerate(datasets):
                ax = axes[i, j]
                cell = sub[(sub["model_arch"] == m) & (sub["dataset"] == d)]
                data = [cell.loc[cell["ecosystem"] == e, "energy_harm_j"].to_numpy() for e in order]
                bp = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True, widths=0.6)
                for patch in bp["boxes"]:
                    patch.set_facecolor(PHASE_COLOR[phase])
                    patch.set_alpha(0.45)
                ax.set_title(f"{m} / {d}", fontsize=9)
                ax.tick_params(axis="x", rotation=80, labelsize=6)
                if j == 0:
                    ax.set_ylabel("Energy [J]")
        fig.suptitle(
            f"{phase}: harmonised energy per epoch, one panel per model x dataset\n"
            "(boxes show within-run epoch dispersion of a single run, not between-run variability)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_panels_{phase.lower()}.png", bbox_inches="tight")
        plt.close(fig)


def fig_pareto(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, phase in zip(axes, ["Training", "Inference"]):
        sub = df[df["phase"] == phase]
        g = sub.groupby("ecosystem", observed=True)
        pts = pd.DataFrame({"t": g["duration_s"].mean(), "e": g["energy_harm_j"].mean()})
        ax.scatter(pts["t"], pts["e"], s=60, color=PHASE_COLOR[phase], edgecolor="black", zorder=3)
        for name, r in pts.iterrows():
            ax.annotate(name, (r["t"], r["e"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        front = pts.sort_values("t")
        keep, best = [], np.inf
        for name, r in front.iterrows():
            if r["e"] < best:
                keep.append(name)
                best = r["e"]
        ax.plot(front.loc[keep, "t"], front.loc[keep, "e"], color="black", lw=1.4, zorder=2, label="Pareto frontier")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean duration per epoch [s]")
        ax.set_ylabel("Mean harmonised energy per epoch [J]")
        rho = pts["t"].rank().corr(pts["e"].rank())
        ax.set_title(f"{phase}  (Spearman rho = {rho:.2f})")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Under a common measurement boundary the energy and time orderings almost coincide",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_pareto.png", bbox_inches="tight")
    plt.close(fig)


def fig_dataset_absolute(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    datasets = ["FashionMNIST", "CIFAR100", "TinyImageNet"]
    for ax, phase in zip(axes, ["Training", "Inference"]):
        sub = df[df["phase"] == phase]
        piv = (
            sub.groupby(["ecosystem", "dataset"], observed=True)["energy_harm_j"]
            .mean()
            .unstack()[datasets]
        )
        piv = piv.loc[piv.mean(axis=1).sort_values().index]
        x = np.arange(len(piv))
        w = 0.26
        for k, d in enumerate(datasets):
            ax.bar(x + (k - 1) * w, piv[d], w, label=d)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=40, ha="right")
        ax.set_ylabel("Energy [J]")
        ax.set_title(f"{phase}: absolute energy per epoch")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Dataset effect on an absolute scale (the manuscript's heatmap is normalised per column, "
        "which cannot show growth across datasets)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dataset_absolute.png", bbox_inches="tight")
    plt.close(fig)


def fig_gpu_load(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = _order(df)
    width = 0.38
    x = np.arange(len(order))
    for k, phase in enumerate(["Training", "Inference"]):
        sub = df[df["phase"] == phase]
        vals = [sub.loc[sub["ecosystem"] == e, "gpu_power_derived_w"].mean() for e in order]
        ax.bar(x + (k - 0.5) * width, vals, width, label=phase, color=PHASE_COLOR[phase])
    ax.axhline(GPU_TDP_W, color="crimson", ls="--", lw=1.2, label=f"L40S board limit ({GPU_TDP_W:.0f} W)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_ylabel("Mean GPU power [W]")
    ax.set_ylim(0, GPU_TDP_W * 1.1)
    ax.set_title("The workload never approaches the GPU power limit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gpu_load.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = load()
    fig_component_stack(df)
    fig_panels(df)
    fig_pareto(df)
    fig_dataset_absolute(df)
    fig_gpu_load(df)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  wrote {p.relative_to(FIG_DIR.parents[2])}")


if __name__ == "__main__":
    main()
