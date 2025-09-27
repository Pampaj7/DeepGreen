#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ========= CONFIG =========
CSV_PATH = "/home/pampaj/DeepGreen/results/data/combined_data.csv"
OUTPUT_PNG = "../plots/scatter_regression.png"
FIGSIZE = (8, 6)
# ==========================

def fit_loglog(x, y):
    """
    Fit log-log: log(y) = a*log(x) + b
    Restituisce a, b per la retta in log-log.
    """
    mask = (x > 0) & (y > 0)
    x, y = np.log(x[mask]), np.log(y[mask])
    if len(x) < 2:
        return None, None
    a, b = np.polyfit(x, y, 1)
    return a, b

def main():
    df = pd.read_csv(CSV_PATH)

    if not {"duration", "energy_consumed", "language"}.issubset(df.columns):
        raise ValueError("CSV deve contenere: duration, energy_consumed, language")

    plt.figure(figsize=FIGSIZE)
    languages = df["language"].unique()
    cmap = plt.get_cmap("tab10")

    for i, lang in enumerate(languages):
        sub = df[df["language"] == lang]
        plt.scatter(
            sub["duration"],
            sub["energy_consumed"],
            label=lang,
            alpha=0.6,
            s=40,
            color=cmap(i)
        )

        # Fit log-log
        a, b = fit_loglog(sub["duration"].values, sub["energy_consumed"].values)
        if a is not None:
            x_fit = np.linspace(sub["duration"].min(), sub["duration"].max(), 100)
            y_fit = np.exp(b) * x_fit**a
            plt.plot(x_fit, y_fit, color=cmap(i), linewidth=2)

    # log scale
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Duration [s] (log scale)")
    plt.ylabel("Energy Consumed [J] (log scale)")
    plt.title("Energy vs Duration with Language-wise Regression (log-log)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220)
    plt.close()
    print(f"✅ Salvato {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
