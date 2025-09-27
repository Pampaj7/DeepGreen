#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys

# Uso: python3 fix_matlab_csv.py nomefile.csv
if len(sys.argv) < 2:
    print("❌ Devi passare il nome del file CSV da modificare.")
    sys.exit(1)

FILE = sys.argv[1]

# Carica il CSV
df = pd.read_csv(FILE)

# 🔴 Rimuovi la colonna 'model' se esiste
if "model" in df.columns:
    df = df.drop(columns=["model"])
    print("✅ Colonna 'model' rimossa.")

# 🔵 Aggiungi la colonna 'language' con valore fisso 'MATLAB'
df["language"] = "MATLAB"
print("✅ Colonna 'language' aggiunta con valore 'MATLAB'.")

# 🟢 Modifica colonna 'fase' se esiste
if "fase" in df.columns:
    df["fase"] = df["fase"].replace("test", "eval")
    print("✅ Colonna 'fase' aggiornata: test → eval.")
else:
    print("⚠️ Nessuna colonna 'fase' trovata.")

# Salva lo stesso file (sovrascrive)
df.to_csv(FILE, index=False)
print(f"💾 File salvato: {FILE}")
