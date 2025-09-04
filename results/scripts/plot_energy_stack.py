#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os

# ========= CONFIG =========
CSV_PATH = "/home/pampaj/DeepGreen/results/data/all_combined_data.csv"
OUTPUT_PNG = "../plots/energy_stacked"     # base path (senza estensione ok)
STACK_BY = "model"                         # "model" | "dataset" | "model_dataset"

FILTERS = {
    "fase": "eval",        # es: "train" | "inference" | None
    # "dataset": "CIFAR100",
    # "model": "ResNet18",   # oppure "modello" se nel CSV in italiano
}

LANG_ORDER = None  # es: ["C++", "Java", "MATLAB", "Python", "R"] oppure None

TITLE = "Energy Consumption by Language (stacked)"
FIGSIZE = (11, 5.5)
BAR_EDGEWIDTH = 0.6
BAR_LINECOLOR = "white"
ROTATION = 35

# Se vuoi forzare una scala scientifica fissa, metti un numero (es. 1e-7)
# Altrimenti lascia None per autoscale J/mJ/µJ/nJ
FORCE_SCI_FACTOR = None   # es: 1e-7
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


def pick_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Nel CSV mancano le colonne alternative: {candidates}")


# ------- NORMALIZZAZIONI -------
def normalize_model_name(s: str, short=False) -> str:
    """Unisce vgg/vgg16 e resnet/resnet18."""
    if not isinstance(s, str):
        return "NA"
    t = s.strip().lower()

    if re.search(r"\bvgg([ -_]?16)?\b", t):
        return "VGG" if short else "VGG16"

    if re.search(r"\bresnet([ -_]?18)?\b", t):
        return "ResNet" if short else "ResNet18"

    # capitalizzazione gentile
    return s.strip()


def make_stack_key(df, how, col_model, col_dataset):
    if how == "model":
        return df[col_model].fillna("NA").astype(str)
    elif how == "dataset":
        return df[col_dataset].fillna("NA").astype(str)
    elif how == "model_dataset":
        m = df[col_model].fillna("NA").astype(str)
        d = df[col_dataset].fillna("NA").astype(str)
        return m + " · " + d
    else:
        raise ValueError("STACK_BY deve essere 'model', 'dataset' o 'model_dataset'")


def color_palette(n):
    base = plt.get_cmap("tab20").colors
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]


def ensure_png_path(base_path: str, stack_by: str, phase_value) -> Path:
    base = Path(base_path)
    suffix = base.suffix.lower()
    stem = base.stem if suffix else base.name
    parent = base.parent if str(base.parent) not in ("", ".") else Path(".")
    phase_tag = "all" if (phase_value is None or str(phase_value).strip() == "") else str(phase_value)
    filename = f"{stem}__{stack_by}__{phase_tag}.png"
    return parent / filename


def main():
    df = pd.read_csv(CSV_PATH)

    # Mappatura nomi colonne (accetta IT/EN)
    col_lang    = pick_col(df, "language", "linguaggio")
    col_energy  = pick_col(df, "energy_consumed", "energia_consumata")
    col_model   = pick_col(df, "model", "modello")
    col_dataset = pick_col(df, "dataset", "data_set")
    col_phase   = "fase" if "fase" in df.columns else ("phase" if "phase" in df.columns else None)

    # Normalizza i nomi modello (accorpa vgg/vgg16 e resnet/resnet18)
    # Metti short=True se vuoi etichette "VGG"/"ResNet" in legenda
    df[col_model] = df[col_model].apply(lambda s: normalize_model_name(s, short=False))

    # Applica filtri robusti
    for k, v in FILTERS.items():
        if v is None:
            continue
        # se la chiave esiste pari pari
        if k in df.columns:
            df = df[df[k] == v]
            continue
        # mappa chiavi comuni
        if k in ("model", "modello"):
            df = df[df[col_model] == v]
        elif k == "dataset":
            df = df[df[col_dataset] == v]
        elif k in ("fase", "phase") and col_phase:
            df = df[df[col_phase] == v]

    if df.empty:
        raise ValueError("Dopo i filtri, il dataframe è vuoto. Controlla i valori in FILTERS.")

    # Chiave di stacking
    df["stack_key"] = make_stack_key(df, STACK_BY, col_model, col_dataset)

    # Aggregazione energia
    pivot = (
        df.groupby([col_lang, "stack_key"], as_index=False)[col_energy]
          .sum()
          .pivot(index=col_lang, columns="stack_key", values=col_energy)
          .fillna(0.0)
    )

    if pivot.empty:
        raise ValueError("Nessun dato dopo il pivot: verifica colonne e filtri.")

    # Ordine lingue
    totals = pivot.sum(axis=1)
    if LANG_ORDER is None:
        pivot = pivot.loc[totals.sort_values().index]
        totals = totals.loc[pivot.index]
    else:
        order = [l for l in LANG_ORDER if l in pivot.index] + [l for l in pivot.index if l not in LANG_ORDER]
        pivot = pivot.loc[order]
        totals = totals.loc[pivot.index]

    # Ordina le serie della legenda per contributo totale (decrescente)
    stack_totals = pivot.sum(axis=0).sort_values(ascending=False)
    pivot = pivot[stack_totals.index]

    # Scala unità
    scale, unit = smart_unit_scale(pivot.values.flatten())
    pivot_s = pivot * scale
    totals_s = totals * scale

    # Colori
    keys = list(pivot_s.columns)
    colors = color_palette(len(keys))

    # Output path finale
    phase_value = FILTERS.get("fase") if "fase" in FILTERS else (FILTERS.get("phase") if "phase" in FILTERS else None)
    out_png = ensure_png_path(OUTPUT_PNG, STACK_BY, phase_value)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=FIGSIZE)
    bottoms = np.zeros(len(pivot_s), dtype=float)
    x = np.arange(len(pivot_s.index))

    for k, c in zip(keys, colors):
        vals = pivot_s[k].values
        plt.bar(x, vals, bottom=bottoms, label=k,
                edgecolor=BAR_LINECOLOR, linewidth=BAR_EDGEWIDTH, color=c)
        bottoms += vals

    # Etichette totali sopra le colonne
    for xi, tv in zip(x, totals_s.values):
        if tv > 0:
            plt.text(xi, tv * 1.01, f"{tv:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, pivot_s.index, rotation=ROTATION, ha="right")
    plt.ylabel(f"Energy Consumed [{unit}]")
    plt.xlabel("Language")

    subtitle = ""
    if col_phase:
        phase_val = FILTERS.get("fase") if FILTERS.get("fase") is not None else None
        if phase_val is not None:
            subtitle = f" – phase: {phase_val}"
    plt.title(TITLE + subtitle)

    legend_title = {"model": "Model", "dataset": "Dataset", "model_dataset": "Model · Dataset"}[STACK_BY]
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    print(f"✅ Plot salvato in: {out_png}")
    print(f"Stack by: {STACK_BY} | Unit: {unit} | Filtri: {FILTERS}")


if __name__ == "__main__":
    main()
