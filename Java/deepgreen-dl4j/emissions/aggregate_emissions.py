import os
import pandas as pd
import re
from collections import defaultdict

# Directory contenente i CSV
EMISSIONS_DIR = os.path.dirname(__file__)

# Pattern: vgg16_cifar_train_epoch1.csv, resnet18_fashion_eval_epoch2.csv, ecc.
FILENAME_PATTERN = re.compile(
    r"(?P<model>\w+)[_-](?P<dataset>[\w\d]+(?:\.csv)?)[_-](?P<phase>train|eval)[_-]epoch(?P<epoch>\d+)\.csv"
)

# Raggruppa i file per (modello, dataset)
groups = defaultdict(list)
files_to_delete = []

print("File trovati nella cartella emissions:")
for fname in os.listdir(EMISSIONS_DIR):
    print(" -", fname)
    if fname.endswith('.csv') and ("train" in fname or "eval" in fname) and fname != 'aggregate_emissions.csv':
        match = FILENAME_PATTERN.match(fname)
        if not match:
            print(f"  [IGNORATO] Nome file non corrisponde al pattern: {fname}")
            continue
        model = match.group('model')
        dataset = match.group('dataset')
        phase = match.group('phase')
        epoch = int(match.group('epoch'))
        fpath = os.path.join(EMISSIONS_DIR, fname)
        groups[(model, dataset)].append((fpath, phase, epoch))
        files_to_delete.append(fpath)

if not groups:
    print("Nessun file CSV trovato o nessun file corrispondente al pattern.")
else:
    for (model, dataset), file_phase_list in groups.items():
        print(f"Raggruppo {len(file_phase_list)} file per modello={model}, dataset={dataset}")
        records = []
        for fpath, phase, epoch in file_phase_list:
            print(f"  - Aggiungo {fpath} (fase: {phase}, epoca: {epoch})")
            df = pd.read_csv(fpath)
            df['modello'] = model
            df['dataset'] = dataset
            df['fase'] = phase
            df['epoca'] = epoch
            records.append(df)
        if records:
            df_all = pd.concat(records, ignore_index=True)
            cols = ['modello', 'dataset', 'fase', 'epoca'] + [c for c in df_all.columns if c not in ['modello', 'dataset', 'fase', 'epoca']]
            df_all = df_all[cols]
            out_path = os.path.join(EMISSIONS_DIR, f'aggregate_{model}_{dataset}.csv')
            df_all.to_csv(out_path, index=False)
            print(f"Tabella aggregata salvata in {out_path}")

    # Elimina i file originali
    for f in files_to_delete:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Errore nell'eliminazione di {f}: {e}")
    print(f"Eliminati {len(files_to_delete)} file CSV originali.") 
    