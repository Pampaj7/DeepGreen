#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# ========= CONFIG =========
CSV_PATH = "/home/pampaj/DeepGreen/results/data/combined_data.csv"
OUTPUT_DIR = "../tables"
# ==========================

def save_table(df, name):
    """Salva tabella in CSV e Markdown"""
    out_csv = Path(OUTPUT_DIR) / f"{name}.csv"
    out_md  = Path(OUTPUT_DIR) / f"{name}.md"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_markdown(out_md, index=False)
    print(f"✅ Salvata tabella: {out_csv}, {out_md}")

def main():
    df = pd.read_csv(CSV_PATH)

    required = {"language", "fase", "dataset", "energy_consumed", "duration"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV deve contenere le colonne: {required}")

    # === 1. Statistiche di base per linguaggio ===
    stats = df.groupby("language").agg(
        mean_energy=("energy_consumed", "mean"),
        std_energy=("energy_consumed", "std"),
        min_energy=("energy_consumed", "min"),
        max_energy=("energy_consumed", "max"),
        mean_duration=("duration", "mean"),
        std_duration=("duration", "std"),
        min_duration=("duration", "min"),
        max_duration=("duration", "max"),
    ).reset_index()
    save_table(stats, "stats_per_language")

    # === 2. Efficienza (duration/energy) ===
    df["efficiency"] = df["duration"] / df["energy_consumed"]
    eff = df.groupby("language")["efficiency"].mean().reset_index().sort_values("efficiency", ascending=False)
    save_table(eff, "efficiency_ranking")

    # === 3. Train vs Eval per linguaggio ===
    train_eval = df.groupby(["language", "fase"]).agg(
        mean_energy=("energy_consumed", "mean"),
        mean_duration=("duration", "mean"),
    ).reset_index()
    save_table(train_eval, "train_vs_eval")

    # === 4. Best per dataset (min energy e min duration) ===
    best_energy = df.loc[df.groupby("dataset")["energy_consumed"].idxmin()][["dataset", "language", "energy_consumed"]]
    best_energy = best_energy.rename(columns={"language": "best_energy_language", "energy_consumed": "min_energy"})
    best_duration = df.loc[df.groupby("dataset")["duration"].idxmin()][["dataset", "language", "duration"]]
    best_duration = best_duration.rename(columns={"language": "best_duration_language", "duration": "min_duration"})
    best = pd.merge(best_energy, best_duration, on="dataset")
    save_table(best, "best_per_dataset")

    print("🎯 Tutte le analisi completate!")

if __name__ == "__main__":
    main()
