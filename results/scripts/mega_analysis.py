#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========= CONFIG =========
CSV_PATH = "/home/pampaj/DeepGreen/results/data/combined_data.csv"
OUTPUT_DIR = "../plots/full_report"
BASELINE_LANG = "PyTorch"   # per Relative efficiency
FIGSIZE = (10, 6)
# ==========================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["fase"] = df["fase"].str.lower().replace("test", "eval")
    return df

def filter_phase(df, phase):
    if phase == "all":
        return df.copy()
    return df[df["fase"] == phase].copy()

def leaderboard(df, phase):
    agg = df.groupby("language")["energy_consumed"].sum().sort_values()
    agg.plot(kind="bar", figsize=FIGSIZE,
             title=f"Energy Leaderboard by Language (phase: {phase})")
    plt.ylabel("Energy [J]")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/leaderboard_{phase}.png", dpi=220)
    plt.close()

def model_comparison(df, phase):
    agg = df.groupby(["language","modello"])["energy_consumed"].sum().unstack(fill_value=0)
    totals = agg.sum(axis=1).sort_values()
    agg = agg.loc[totals.index]
    agg.plot(kind="bar", figsize=FIGSIZE,
             title=f"Model Comparison (phase: {phase})")
    plt.ylabel("Energy [J]")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_{phase}.png", dpi=220)
    plt.close()

def dataset_effect(df, phase):
    agg = df.groupby(["language","dataset"])["energy_consumed"].sum().unstack(fill_value=0)
    totals = agg.sum(axis=1).sort_values()
    agg = agg.loc[totals.index]
    agg.plot(kind="bar", figsize=FIGSIZE,
             title=f"Dataset Effect (phase: {phase})")
    plt.ylabel("Energy [J]")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dataset_effect_{phase}.png", dpi=220)
    plt.close()

def phase_gap(df):
    train = df[df["fase"]=="train"].groupby("language")["energy_consumed"].sum()
    eval_ = df[df["fase"]=="eval"].groupby("language")["energy_consumed"].sum()
    ratio = (train / eval_).fillna(0).sort_values()
    ratio.plot(kind="bar", figsize=FIGSIZE,
               title="Train/Eval Energy Ratio (all data)")
    plt.ylabel("Ratio (train/eval)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase_gap.png", dpi=220)
    plt.close()

def relative_efficiency(df, phase):
    totals = df.groupby("language")["energy_consumed"].sum().sort_values()
    baseline = totals.get(BASELINE_LANG, np.nan)
    rel = (totals / baseline - 1) * 100
    rel.plot(kind="bar", figsize=FIGSIZE,
             title=f"Relative Efficiency vs {BASELINE_LANG} (phase: {phase})")
    plt.ylabel("% difference in energy")
    plt.axhline(0, color="black", lw=1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/relative_efficiency_{phase}.png", dpi=220)
    plt.close()

def energy_breakdown(df, phase):
    agg = df.groupby("language")[["cpu_energy","gpu_energy","ram_energy"]].sum()
    totals = agg.sum(axis=1).sort_values()
    agg = agg.loc[totals.index]
    agg.plot(kind="bar", stacked=True, figsize=FIGSIZE,
             title=f"Energy Breakdown by Component (phase: {phase})")
    plt.ylabel("Energy [J]")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/energy_breakdown_{phase}.png", dpi=220)
    plt.close()

def heatmap(df, phase):
    pivot = df.groupby(["language","dataset"])["energy_consumed"].sum().unstack(fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title(f"Energy Heatmap (Language × Dataset, phase: {phase})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_{phase}.png", dpi=220)
    plt.close()

def radar_plot(df, phase):
    agg = df.groupby("language")[["energy_consumed","cpu_power","gpu_power","duration"]].mean()
    if agg.empty:
        return
    agg = agg.loc[agg["energy_consumed"].sort_values().index]  # ordina anche il radar
    norm = (agg - agg.min()) / (agg.max() - agg.min())
    labels = norm.columns
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7,7), dpi=220)
    ax = plt.subplot(111, polar=True)

    for lang, row in norm.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=lang)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(f"Radar Plot of Languages (phase: {phase})")
    plt.legend(bbox_to_anchor=(1.1,1.05))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/radar_{phase}.png", dpi=220)
    plt.close()

def main():
    df = load_data()
    for phase in ["train", "eval", "all"]:
        df_phase = filter_phase(df, phase)
        if df_phase.empty:
            print(f"⚠️ Nessun dato per fase={phase}, salto…")
            continue

        leaderboard(df_phase, phase)
        model_comparison(df_phase, phase)
        dataset_effect(df_phase, phase)
        relative_efficiency(df_phase, phase)
        energy_breakdown(df_phase, phase)
        heatmap(df_phase, phase)
        radar_plot(df_phase, phase)

    # questo solo una volta su tutto il dataset
    phase_gap(df)
    print(f"✅ Grafici generati in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
