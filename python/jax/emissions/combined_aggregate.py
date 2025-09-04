import pandas as pd
import os

# Directory contenente i CSV (sostituisci con il percorso reale)
input_dir = "./"
output_file = "jax_combined_data.csv"

# Lista per memorizzare i DataFrame
dataframes = []

# Scorri tutti i file nella directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and filename.startswith("aggregate_"):
        # Estrai model e dataset dal nome del file
        # Rimuovi "aggregate_" per ottenere "vgg16_tiny"
        base_name = filename.replace("aggregate_", "").replace(".csv", "")
        parts = base_name.split("_")
        model = parts[0]  # Prima parte (es. vgg16)
        dataset = parts[1]  # Seconda parte (es. tiny)

        # Leggi il CSV
        df = pd.read_csv(os.path.join(input_dir, filename))

        # Aggiungi le nuove colonne
        df['model'] = model
        df['dataset'] = dataset
        df['language'] = "Jax"

        # Aggiungi il DataFrame alla lista
        dataframes.append(df)

# Unisci tutti i DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Salva il risultato in un nuovo CSV
combined_df.to_csv(output_file, index=False)
print(f"Dati uniti salvati in {output_file}")