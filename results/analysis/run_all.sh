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
# The campaign interpreter, not whatever `python3` resolves to. The system
# interpreter has no pandas, and under it check_consistency.py reports a clean
# run on a repository it has not finished checking.
PY="${PYTHON:-$(cd ../.. && pwd)/.venv-deepgreen/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY; set PYTHON=..." >&2; exit 2; }

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
run 16_coverage_sensitivity.py    # whose energy is the time between phases?
# The one script that reads the superseded campaign, and only to contrast it
# with this one: PyTorch changed precision policy between them and six stacks
# did not, so the pair is a TF32 ablation with a control group.
run 18_precision_ablation.py      # TF32, from the two campaigns
run 19_gpu_utilisation.py         # was the accelerator busy? over the runs the record covers
# The collapse finding belongs to the superseded campaign -- this one has none
# -- so its tables are derived under their own v1_ names and never overwrite
# the v2 ones. 12 reads both and reports the contrast.
echo "### 15_convergence.py --campaign v1"
"$PY" 15_convergence.py --campaign v1
echo

# These measure the machine rather than re-derive the campaign, so they are
# host- and network-dependent and are listed separately. They were in no
# pipeline at all, which meant the macros they feed -- \vCalib*, \vIdle*,
# \vMech* -- came from committed CSVs that a clean checkout would not have, and
# 12_paper_numbers.py reads them behind `if path.exists()`, so the manuscript
# would have compiled with them silently absent.
#
# The two that re-measure the host are behind a flag, because this pipeline is
# itself load. measure_idle.py ran here unconditionally and a full run measured
# the idle host while being the only thing on it -- 3.6 W of CPU package above
# a reading taken an hour earlier on a quiet machine, written straight over a
# manuscript input. Set DEEPGREEN_MEASURE_HOST=1 on a quiet host to re-measure;
# otherwise the committed baseline stands and this says so.
echo "=== measurements of this host (re-measure, do not re-derive) ==="
run 17_window_calibration.py      # between-window drift, if results/calibration/ exists
if [ "${DEEPGREEN_MEASURE_HOST:-0}" = "1" ]; then
  "$PY" ../../scripts/measure_idle.py       || echo "  (skipped: needs an idle machine)"
  "$PY" ../../scripts/probe_reported_window.py || echo "  (skipped: needs network timing)"
else
  for f in v2_idle_baseline.json v2_window_mechanism.csv; do
    p="../revision/tables/$f"
    if [ -f "$p" ]; then
      echo "  reusing $f, measured $(date -r "$p" -u +%Y-%m-%dT%H:%MZ)"
    else
      echo "  !! $f absent and not measured; the macros it feeds will be undefined"
    fi
  done
  echo "  (set DEEPGREEN_MEASURE_HOST=1 on a quiet host to re-measure)"
fi

echo "=== parity of the comparison itself ==="
"$PY" ../../scripts/verify_architecture_parity.py || echo "  ARCHITECTURES DIFFER"
"$PY" ../../scripts/verify_data_parity.py         || echo "  DATA DIFFERS"

echo "=== manuscript artefacts (paper/) ==="
run 12_paper_numbers.py           # -> paper/generated/{numbers,tab_*}.tex
run 13_paper_figures.py           # -> paper/figures/*.png
echo "### 13_paper_figures.py --campaign v1"
"$PY" 13_paper_figures.py --campaign v1   # -> fig_convergence_first_campaign.png
echo

# The replication package is an OUTPUT of this pipeline and belongs at the end
# of it, not before it. Nothing re-ran it for a fortnight, so it held the first
# campaign while parts of the analysis read it as an input -- which is how the
# manuscript came to quote a conformance failure from a campaign it does not
# describe. Nothing reads it now; it is built here so it can never again be
# older than the tree it claims to flatten. It refuses a partial campaign
# itself (consolidate_raw.refuse_if_partial), and a refusal is not a pipeline
# failure: monitoring runs are meant to reach this line and stop.
echo "=== replication package (results/replication/) ==="
"$PY" ../../scripts/consolidate_raw.py || echo "  (not consolidated; see above)"

echo "Done. Tables in results/revision/, manuscript inputs in paper/."
