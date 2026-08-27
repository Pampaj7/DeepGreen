#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# ========= CONFIG =========
CSV_PATH = Path(__file__).resolve().parent / "cpp_combined_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "cpp_combined_data.csv"
DROP_COLS = ["model"]   # colonne da rimuovere
# ==========================

def main():
    df = pd.read_csv(CSV_PATH)

    # 🔴 Rimuove la colonna se esiste
    for col in DROP_COLS:
        if col in df.columns:
            print(f"Rimuovo colonna: {col}")
            df = df.drop(columns=[col])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ File salvato in: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
