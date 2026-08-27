#!/usr/bin/env bash
# Regenerate every table and figure in results/revision/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for s in 01_data_audit.py 02_main_tables.py 03_statistics.py 04_energy_vs_time.py \
         05_efficiency_index.py 06_dataset_scaling.py 07_carbon_and_scale.py \
         08_figures.py 10_implementation_audit.py 09_campaign_v2.py; do
  echo "### $s"
  "$PY" "$s"
  echo
done
echo "Done. Output in results/revision/."
