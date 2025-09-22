#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ========= CONFIG =========
INPUT_DIR = "./"
OUTPUT_FILE = "combined_data.csv"
# ==========================

def normalize_model(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower()
    if "vgg" in t:
        return "VGG16"
    if "resnet" in t:
        return "ResNet18"
    return s.strip()

def normalize_dataset(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower()
    if "cifar" in t:
        return "CIFAR100"
    if "fashion" in t:
        return "FashionMNIST"
    if "tiny" in t:
        return "TinyImageNet"
    return s.strip()

def normalize_phase(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower()
    return "eval" if t == "test" else t

def normalize_language(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower()
    if t in ("c++", "cpp", "c plus plus"):
        return "C++"
    if t in ("pytorch", "torch"):
        return "PyTorch"
    if t in ("tensorflow", "tf"):
        return "TensorFlow"
    if t == "jax":
        return "JAX"
    if t == "rust":
        return "Rust"
    if t == "r":
        return "R"
    return s.strip()

def main():
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv") and f != OUTPUT_FILE]
    if not csv_files:
        raise FileNotFoundError("Nessun CSV trovato nella cartella corrente.")

    all_dfs = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(INPUT_DIR, file))

        # normalizza colonne se presenti
        if "modello" in df.columns:
            df["modello"] = df["modello"].apply(normalize_model)
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].apply(normalize_dataset)
        if "fase" in df.columns:
            df["fase"] = df["fase"].apply(normalize_phase)
        if "language" in df.columns:
            df["language"] = df["language"].apply(normalize_language)

        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ File combinato salvato come {OUTPUT_FILE} con {len(final_df)} righe")

if __name__ == "__main__":
    main()
