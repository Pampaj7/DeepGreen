#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scatter plot doppio (train vs eval) per visualizzare il tradeoff tra energia consumata e tempo di esecuzione.
Mostra i quadranti per entrambe le fasi.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======== CONFIG ========
CSV_PATH = "/home/pampaj/DeepGreen/results/data/combined_data.csv"
OUTPUT_DIR = Path("../plots/plots_tradeoff")
LANG_COL = "language"
TIME_COL = "duration"            # in secondi
ENERGY_COL = "energy_consumed"   # in kWh
FASE_COL = "fase"

# ======== SETUP ========
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
df = pd.read_csv(CSV_PATH)
df.rename(columns={TIME_COL: "time_s"}, inplace=True)

# Funzione per fare lo scatter di una fase
def scatter_phase(ax, data, phase_name):
    med_energy = data[ENERGY_COL].median()
    med_time = data["time_s"].median()

    # Quadranti colorati
    ax.axhspan(0, med_energy, 0, med_time/max(data["time_s"]), facecolor="lightgreen", alpha=0.2)
    ax.axhspan(med_energy, data[ENERGY_COL].max(), 0, med_time/max(data["time_s"]), facecolor="orange", alpha=0.15)
    ax.axhspan(0, med_energy, med_time/max(data["time_s"]), 1, facecolor="yellow", alpha=0.15)
    ax.axhspan(med_energy, data[ENERGY_COL].max(), med_time/max(data["time_s"]), 1, facecolor="red", alpha=0.1)

    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for (lang, row), m in zip(data.groupby(LANG_COL), markers):
        ax.scatter(row["time_s"], row[ENERGY_COL], label=lang,
                   s=80, marker=m, alpha=0.8, edgecolors="black", linewidth=0.6)

    ax.axhline(med_energy, color="gray", linestyle="--", linewidth=1)
    ax.axvline(med_time, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Execution Time (s)", fontsize=12)
    ax.set_ylabel("Energy Consumed (kWh)", fontsize=12)
    ax.set_title(f"{phase_name.capitalize()} Phase", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

# ======== FIGURE TRAIN vs EVAL ========
# FIGURE TRAIN vs EVAL con assi indipendenti
fig, axes = plt.subplots(1, 2, figsize=(15, 7))  # tolto sharey

# Train
df_train = df[df[FASE_COL] == "train"]
scatter_phase(axes[0], df_train, "train")

# Eval
df_eval = df[df[FASE_COL] == "eval"]
scatter_phase(axes[1], df_eval, "eval")

# Legenda unica
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Language", fontsize=11, title_fontsize=12,
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.05))

fig.suptitle("⚡ Low Energy ≠ Fast Execution (Train vs Eval)", fontsize=16, fontweight="bold")

# Nota di avviso
fig.text(0.5, 0.06, "⚠ Different scales used for Train and Eval", ha="center", fontsize=11, color="darkred")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUTPUT_DIR / "scatter_train_eval.png", dpi=300, bbox_inches="tight")
plt.close()
