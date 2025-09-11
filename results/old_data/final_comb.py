import pandas as pd
import os

# Directory contenente i CSV (sostituisci con il percorso reale)
input_dir = "./"
output_file = "all_combined_data.csv"

# Lista per memorizzare i DataFrame
dataframes = []

# Scorri tutti i file nella directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        # Leggi il CSV
        df = pd.read_csv(os.path.join(input_dir, filename))
        # Aggiungi il DataFrame alla lista
        dataframes.append(df)

# Unisci tutti i DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Salva il risultato in un nuovo CSV
combined_df.to_csv(output_file, index=False)
print(f"Tutti i dati uniti salvati in {output_file}")