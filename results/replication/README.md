# Raw measurement records

Every number in the manuscript comes from these tables. The first five are the
campaign's raw output, flattened: nothing here is aggregated, filtered, rounded
or corrected. The last three freeze the inputs of the utilisation analyses that
a clone cannot otherwise recover (see below).

Regenerate them from the campaign tree with

```bash
python3 scripts/consolidate_raw.py           # rewrite
python3 scripts/consolidate_raw.py --check   # verify against results/campaign_v2/
```

and rebuild the run tree from them, or check that they rebuild it exactly, with

```bash
python3 scripts/restore_from_replication.py --all              # into results/campaign_v2/
python3 scripts/restore_from_replication.py --check-roundtrip  # 13,440 of 13,440 files identical
```

| File | Rows | One row is |
|---|---|---|
| `codecarbon.csv.gz` | 12,600 | one measured block, as the software estimator reported it |
| `counters.csv.gz` | 12,600 | the same block, as NVML and the RAPL package counters reported it |
| `metrics.csv.gz` | 6,300 | one epoch of one run: losses and test accuracy |
| `manifests.csv.gz` | 210 | one run: seed, versions, environment, as recorded at launch |
| `data_fingerprints.csv.gz` | 210 | one run's `data_fingerprint.csv`: summary statistics of the test split it read, for the data-parity check |
| `run_windows.csv.gz` | 210 | one run's whole-run window, [manifest `machine_state.utc`, `counters.csv` mtime], in unix seconds |
| `training_windows.csv.gz` | 6,300 | one training block's counter-bracketed interval, in unix seconds |
| `gpu_utilisation_excerpt.csv.gz` | 133,584 | one 1 Hz `nvidia-smi` sample, from `results/gpu_utilisation.csv`, columns unchanged |

210 runs = 7 ecosystems × 2 architectures × 3 datasets × 5 repetitions.
12,600 blocks = 210 runs × 30 epochs × 2 phases (train, eval).

`SHA256SUMS` covers every file. They are written with a zeroed gzip
timestamp, so the same records produce the same bytes on any machine and the
checksums are meaningful. `data_fingerprints.csv.gz` and the three utilisation
files were added after the first four, which did not change.

## Utilisation inputs: windows and the sampler excerpt

`results/analysis/19_gpu_utilisation.py` (whole-run utilisation) and
`20_saturation.py` (utilisation over training blocks) join the 1 Hz
`nvidia-smi` record to time windows derived from file **mtimes**: a run's
window ends at `counters.csv`'s mtime, and a training block's interval is its
`emissions_train_epoch<N>.csv` mtime minus CodeCarbon's `duration`, lasting the
counters' `duration_s`. Git does not keep mtimes and neither does the restore
script, and the full record (about 100 MB) is not distributed. So:

* `run_windows.csv.gz` and `training_windows.csv.gz` hold those windows for
  every complete run, exactly as `results/analysis/common.py`
  (`derive_windows`) computes them on the measurement host, where each run's
  mtimes are first checked against CodeCarbon's own `timestamp` column.
* `gpu_utilisation_excerpt.csv.gz` holds the record's rows inside any of those
  windows (±2 s), plus the record's first sample and the first sample after
  the last window closes, so the coverage rule ("a run counts only if its
  window lies inside the record") decides every run as it does on the full
  record. 157 of the 210 runs are covered; the record began after the others
  ran.

The analysis scripts use the raw tree and the full record when they are
present and trustworthy, and refuse if either disagrees with these files; with
the tree absent or restored (mtimes lost), or the record absent, they read
these files instead and produce identical tables and macros.
`consolidate_raw.py --check` compares all three, text-exact, against the raw
tree and the record.

## The two instruments

Each block appears once in `codecarbon.csv.gz` and once in `counters.csv.gz`,
joinable on `(ecosystem, model, dataset, repetition, phase, epoch)`. The two ran over the same
phase boundaries: the shared bridge starts CodeCarbon and reads the counters
inside one synchronous `START`, and stops and reads inside one synchronous
`STOP`. The counter window is therefore *nested* immediately inside
CodeCarbon's, not identical to it, which costs a near-constant ~0.5 J offset in
the CPU term.

Be aware of what the comparison establishes. Where both counters are exposed —
as they are on this machine — CodeCarbon reads the same NVML register and the
same RAPL files that `counters.csv.gz` holds. The GPU terms agree to about a
millijoule because they are the same integer register read twice. So agreement
here certifies the window and the unit conversion, not accuracy against ground
truth; the interesting differences are in the terms CodeCarbon adds (`ram_energy`)
and in the field it reports beside them (`duration`).

The columns to compare are `energy_consumed` (kWh, CodeCarbon, whole machine
including a RAM model) against `hw_total_j` (J, counters, accelerator plus CPU
package). `cpu_energy` and `gpu_energy` in the CodeCarbon table are the
comparable subset; `ram_energy` has no counter equivalent because there is no
RAM energy counter to read.

## Caveats worth knowing before you use these

* **`duration` in the CodeCarbon table is not the phase.** Two thirds of blocks
  carry seconds of tracker lifetime in which no energy was drawn, in three
  discrete modes (0.01 s, 3.27 s, 4.56 s) that block length predicts but does
  not determine. Power derived from that field is understated by up to 11.2× on
  blocks under half a second. `duration_s` in the counters table is the phase.
* **Twelve of the 105 VGG-16 runs never left chance accuracy.** They are here
  because they happened. The manuscript reports accuracy both with and without
  them rather than excluding them, and `results/analysis/15_convergence.py`
  flags them. Filter on `test_acc`, not on the ecosystem.
* **`longitude`/`latitude`/`country_name`** are CodeCarbon's IP geolocation of
  the measuring machine, resolved to a region rather than a place. They set the
  grid carbon intensity used for the CO₂e figures — and, less obviously, they
  are why `duration` is wrong: fetching them is a blocking network call inside
  `stop()`. See `scripts/probe_reported_window.py`.

* **`Java/DL4J` has no `test_loss`.** Our Java harness records test accuracy per
  epoch and not test loss, so that column is empty for all 900 of its rows. The
  conformance checker reports this as a failing check rather than tolerating
  it.
* **The `Rust/tch` runs on five of the six blocks were re-executed later**, in a
  window five days after the rest of the campaign, on the same machine under the
  same idle conditions. On one of those blocks (VGG-16 / Fashion-MNIST) the
  originals trained on all-zero images through a defect of ours and reached
  chance accuracy; the evidence is kept at
  `results/revision/record/vgg_fashion_pipeline_defect.csv`. On the others the
  same loader defect degraded quality without collapsing it. The between-window
  drift is measured rather than assumed —
  `results/analysis/17_window_calibration.py`, and
  `results/revision/tables/v2_window_calibration.*` — at 0.2 % on training
  energy and 10.6 % on inference.
