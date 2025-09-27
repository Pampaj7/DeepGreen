#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ========= CONFIG =========
CSV_PATH = "/home/pampaj/DeepGreen/results/data/combined_data.csv"
OUTPUT_PNG = "../plots/radar_languages_phases.png"
METRICS = ["duration", "cpu_power", "gpu_power", "energy_consumed"]
# ==========================

def make_radar(ax, values, metrics, title):
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values = list(values)  # ✅ fix qui
    values += values[:1]   # chiudi il poligono
    angles += angles[:1]

    ax.plot(angles, values, color="b", linewidth=1.5)
    ax.fill(angles, values, color="b", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, size=11, y=1.1)

def main():
    df = pd.read_csv(CSV_PATH)
    if not {"fase", "language"}.issubset(df.columns):
        raise ValueError("Mancano colonne 'fase' o 'language' nel CSV")

    # normalizza metriche su scala 0-1 globale (train+eval)
    df_norm = df.copy()
    for m in METRICS:
        df_norm[m] = (df_norm[m] - df_norm[m].min()) / (df_norm[m].max() - df_norm[m].min() + 1e-9)

    languages = sorted(df_norm["language"].unique())
    phases = ["train", "eval"]

    nrows = len(languages)
    ncols = len(phases)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             subplot_kw=dict(polar=True),
                             figsize=(ncols*4, nrows*3))

    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, lang in enumerate(languages):
        for j, phase in enumerate(phases):
            ax = axes[i, j]
            subset = df_norm[(df_norm["language"] == lang) & (df_norm["fase"] == phase)]
            if subset.empty:
                ax.set_axis_off()
                continue

            values = [subset[m].mean() for m in METRICS]
            make_radar(ax, values, METRICS, f"{lang} – {phase}")

    plt.suptitle("Radar per Linguaggio e Fase (normalizzato)", size=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Radar multipli salvati in {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
