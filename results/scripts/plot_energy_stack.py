#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= CONFIG =========
CSV_PATH = "/home/pampaj/DeepGreen/results/data/combined_data.csv"
OUTPUT_PNG = "../plots/energy"
TITLE = "Energy Consumption by Language"
FIGSIZE = (11, 5.5)
BAR_EDGEWIDTH = 0.6
BAR_LINECOLOR = "white"
ROTATION = 35

FORCE_SCI_FACTOR = None
FORCE_SCI_LABEL  = "×1e−7 J"
# ==========================

def smart_unit_scale(values):
    if FORCE_SCI_FACTOR is not None:
        return 1.0 / FORCE_SCI_FACTOR, FORCE_SCI_LABEL
    maxv = float(np.nanmax(values)) if len(values) else 0.0
    if maxv >= 1:
        return 1.0, "J"
    elif maxv >= 1e-3:
        return 1e3, "mJ"
    elif maxv >= 1e-6:
        return 1e6, "µJ"
    else:
        return 1e9, "nJ"

def plot_stacked(df, group_col, title, filename, palette):
    pivot = (
        df.groupby(["language", group_col], as_index=False)["energy_consumed"]
          .sum()
          .pivot(index="language", columns=group_col, values="energy_consumed")
          .fillna(0.0)
    )

    # 🔽 ordina sempre in ordine crescente
    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values().index]
    totals = totals.loc[pivot.index]

    stack_totals = pivot.sum(axis=0).sort_values(ascending=False)
    pivot = pivot[stack_totals.index]

    scale, unit = smart_unit_scale(pivot.values.flatten())
    pivot_s = pivot * scale
    totals_s = totals * scale

    plt.figure(figsize=FIGSIZE)
    bottoms = np.zeros(len(pivot_s), dtype=float)
    x = np.arange(len(pivot_s.index))

    keys = list(pivot_s.columns)
    colors = [palette[k] for k in keys]

    for k, c in zip(keys, colors):
        vals = pivot_s[k].values
        plt.bar(x, vals, bottom=bottoms, label=k,
                edgecolor=BAR_LINECOLOR, linewidth=BAR_EDGEWIDTH, color=c)
        bottoms += vals

    for xi, tv in zip(x, totals_s.values):
        if tv > 0:
            plt.text(xi, tv * 1.01, f"{tv:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, pivot_s.index, rotation=ROTATION, ha="right")
    plt.ylabel(f"Energy Consumed [{unit}]")
    plt.xlabel("Language")
    plt.title(title)
    plt.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    print(f"✅ Salvato {filename}")

def plot_grouped(df, filename, datasets, palette_resnet, palette_vgg):
    # 🔽 ordina sempre per totale crescente
    totals = df.groupby("language")["energy_consumed"].sum()
    languages = list(totals.sort_values().index)
    models = ["ResNet18", "VGG16"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(languages))
    width = 0.35
    scale, unit = smart_unit_scale(df["energy_consumed"].values)

    for i, model in enumerate(models):
        offset = (-width/2 if model == "ResNet18" else width/2)
        bottoms = np.zeros(len(languages))
        for j, dataset in enumerate(datasets):
            subset = (
                df[(df["modello"] == model) & (df["dataset"] == dataset)]
                  .groupby("language")["energy_consumed"].sum()
                  .reindex(languages, fill_value=0) * scale
            )
            color = palette_resnet[j] if model == "ResNet18" else palette_vgg[j]
            ax.bar(x + offset, subset.values, width, bottom=bottoms,
                   label=f"{model} · {dataset}",
                   edgecolor=BAR_LINECOLOR, linewidth=BAR_EDGEWIDTH, color=color)
            bottoms += subset.values

    ax.set_xticks(x, languages, rotation=ROTATION, ha="right")
    ax.set_ylabel(f"Energy Consumed [{unit}]")
    ax.set_xlabel("Language")
    ax.set_title(f"{TITLE} – grouped by Model and Dataset")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    print(f"✅ Salvato {filename}")

def main():
    df = pd.read_csv(CSV_PATH)
    if "fase" not in df.columns:
        raise ValueError("Manca la colonna 'fase' nel CSV")

    for phase_value in ["train", "eval"]:
        df_phase = df[df["fase"] == phase_value]
        if df_phase.empty:
            print(f"⚠️ Nessun dato per fase={phase_value}, salto…")
            continue

        datasets = sorted(df_phase["dataset"].unique())

        # palette
        palette_model = {
            "ResNet18": plt.get_cmap("Blues")(0.6),
            "VGG16": plt.get_cmap("Oranges")(0.6),
        }
        palette_dataset = {
            d: plt.get_cmap("tab20")(i) for i, d in enumerate(datasets)
        }
        palette_model_dataset = {
            f"{m} · {d}": (plt.get_cmap("Blues")(0.3 + 0.1*i) if m=="ResNet18"
                           else plt.get_cmap("Oranges")(0.3 + 0.1*i))
            for m in ["ResNet18","VGG16"] for i,d in enumerate(datasets)
        }

        out1 = Path(f"{OUTPUT_PNG}__model__{phase_value}.png")
        out2 = Path(f"{OUTPUT_PNG}__dataset__{phase_value}.png")
        out3 = Path(f"{OUTPUT_PNG}__grouped__{phase_value}.png")

        plot_stacked(df_phase.copy(), "modello",
                     f"{TITLE} – by Model (phase: {phase_value})", out1, palette_model)
        plot_stacked(df_phase.copy(), "dataset",
                     f"{TITLE} – by Dataset (phase: {phase_value})", out2, palette_dataset)
        plot_grouped(df_phase.copy(), out3, datasets,
                     [plt.get_cmap("Blues")(0.3+0.2*i) for i in range(len(datasets))],
                     [plt.get_cmap("Oranges")(0.3+0.2*i) for i in range(len(datasets))])

if __name__ == "__main__":
    main()
