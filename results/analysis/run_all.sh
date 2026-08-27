#!/usr/bin/env bash
# Regenerate every table, figure and quoted number, in dependency order.
#
# The pipeline has two halves. 01-08 and 10 audit the earlier campaign, whose
# raw logs are in results/data/; 09 and 11-15 analyse the replicated campaign in
# results/campaign_v2/ and produce everything the manuscript quotes.
#
# Order matters in the second half: 11 writes the consolidated per-block table
# that 14 and 15 read, 09 writes the quality table that 15 reads, and 12 and 13
# read the output of all of them. Running them out of order silently uses stale
# input -- which is how twenty crashed runs once got averaged in as if they were
# measurements.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

run() { echo "### $1"; "$PY" "$1"; echo; }

echo "=== audit of the earlier campaign (results/data/) ==="
for s in 01_data_audit.py 02_main_tables.py 03_statistics.py 04_energy_vs_time.py \
         05_efficiency_index.py 06_dataset_scaling.py 07_carbon_and_scale.py \
         08_figures.py 10_implementation_audit.py; do
  run "$s"
done

echo "=== replicated campaign (results/campaign_v2/) ==="
run 11_instrument_comparison.py   # consolidates per-block records; both instruments
run 09_campaign_v2.py             # between-run statistics, quality normalisation
run 14_v2_statistics.py           # run-level tests, effect sizes, energy vs time
run 15_convergence.py             # training collapses and conditional accuracy

echo "=== manuscript artefacts (paper/) ==="
run 12_paper_numbers.py           # -> paper/generated/{numbers,tab_*}.tex
run 13_paper_figures.py           # -> paper/figures/*.png

echo "Done. Tables in results/revision/, manuscript inputs in paper/."
