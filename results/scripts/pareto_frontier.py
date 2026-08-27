#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# DEPRECATED -- superseded by results/analysis/ (JSS revision).
#
# This script is kept for provenance only. Do not use it to regenerate results:
#   * it reads CodeCarbon's ``energy_consumed`` column, which is in kWh, and
#     labels it Joules (an offset of 3.6e6). See results/analysis/01_data_audit.py.
#   * it labels the x axis "Language" while plotting language-framework
#     ecosystems.
#   * it hard-codes an absolute path to the author's home directory.
# The replacement pipeline is results/analysis/run_all.sh.
# =============================================================================


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ========= CONFIG =========
CSV_PATH = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "combined_data.csv")
OUTPUT_PNG = "../plots/pareto_frontier_loglog.png"
FIGSIZE = (8, 6)
# ==========================

def pareto_frontier(df, x_col="duration", y_col="energy_consumed"):
    # ordina per durata crescente
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[True, True])
    pareto_points = []
    min_energy = np.inf
    for _, row in sorted_df.iterrows():
        if row[y_col] < min_energy:
            pareto_points.append(row)
            min_energy = row[y_col]
    return pd.DataFrame(pareto_points)

def main():
    df = pd.read_csv(CSV_PATH)

    if not {"duration", "energy_consumed", "language", "fase"}.issubset(df.columns):
        raise ValueError("CSV mancante di colonne richieste: duration, energy_consumed, language, fase")

    plt.figure(figsize=FIGSIZE)
    languages = df["language"].unique()
    markers = {"train": "o", "eval": "s"}  # cerchio train, quadrato eval

    # scatter
    for lang in languages:
        sub = df[df["language"] == lang]
        for phase, mark in markers.items():
            sub_phase = sub[sub["fase"] == phase]
            if not sub_phase.empty:
                plt.scatter(
                    sub_phase["duration"],
                    sub_phase["energy_consumed"],
                    label=f"{lang} ({phase})",
                    alpha=0.7,
                    s=50,
                    marker=mark
                )

    # calcolo frontiera di Pareto
    pareto_df = pareto_frontier(df)
    plt.plot(
        pareto_df["duration"],
        pareto_df["energy_consumed"],
        color="black",
        linewidth=2,
        label="Pareto frontier"
    )
    plt.scatter(
        pareto_df["duration"],
        pareto_df["energy_consumed"],
        color="red",
        edgecolor="black",
        s=80,
        marker="*",
        zorder=5,
        label="Pareto-optimal"
    )

    # log scale
    plt.xscale("log")
    plt.yscale("log")

    # etichette e stile
    plt.xlabel("Duration [s] (log scale)")
    plt.ylabel("Energy Consumed [J] (log scale)")
    plt.title("Pareto Frontier: Energy vs Duration (log-log, all languages)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220)
    plt.close()
    print(f"✅ Salvato {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
