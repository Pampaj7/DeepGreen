#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys

# Uso: python3 rename_fase.py nomefile.csv
if len(sys.argv) < 2:
    print("❌ Devi passare il nome del file CSV da modificare.")
    sys.exit(1)

FILE = sys.argv[1]

# Carica il CSV
df = pd.read_csv(FILE)

# Modifica colonna fase se esiste
if "fase" in df.columns:
    df["fase"] = df["fase"].replace("test", "eval")
    print(f"✅ Colonna 'fase' aggiornata: train → eval in {FILE}")
else:
    print("⚠️ Nessuna colonna 'fase' trovata.")

# Sovrascrivi lo stesso file
df.to_csv(FILE, index=False)
print(f"💾 File salvato: {FILE}")
