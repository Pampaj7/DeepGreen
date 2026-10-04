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
CodeCarbon's, not identical to it, which costs a near-constant offset in the
CPU term: a median of 0.48 J per block <!-- \vOffsetCpuMedianJ --> (0.49 J
<!-- \vOffsetMedianJ --> on the GPU-plus-CPU total, the two GPU terms differing
by a median of 1.2 mJ <!-- \vOffsetGpuMedianMJ -->).

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

Figures below are the manuscript's, from `paper/generated/numbers.tex`; the
comment beside each names the macro, so a regenerated value can be checked
against this text.

* **`duration` in the CodeCarbon table is not the phase.** 75 % of blocks
  <!-- \vWindowPaddedPct --> are filed with a window seconds longer than the
  interval their energy was accumulated over. The excess falls in three modes:
  0.01 s <!-- \vWindowModeOneS --> (25 % of blocks, i.e. none
  <!-- \vWindowModeOnePct -->), 4.57 s <!-- \vWindowModeTwoS --> (75 %
  <!-- \vWindowModeTwoPct -->; itself two populations a second or so apart),
  and a single block at 13.38 s <!-- \vWindowModeThreeS -->. The cause is the
  tracker's blocking network look-ups inside `stop()`, made after the final
  energy reading (`scripts/probe_reported_window.py`). Power derived from that
  field is understated by up to 13.0× <!-- \vPowerUnderstatedWorst --> on
  blocks under half a second <!-- \vPowerUnderstatedWorstBin -->.
  `duration_s` in the counters table is the phase.
* **No run collapsed.** None of the 210 runs stayed at chance accuracy
  (0 collapsed runs <!-- \vCollapseRuns -->; a run counts as collapsed if it
  never exceeded 1.5× chance <!-- \vCollapseFactor -->, as
  `results/analysis/15_convergence.py` defines it). The collapses discussed in
  the manuscript belong to the superseded first campaign, which is not in this
  package.
* **`longitude`/`latitude`/`country_name`** are CodeCarbon's IP geolocation of
  the measuring machine, resolved to a region rather than a place. They set the
  grid carbon intensity used for the CO₂e figures — and, less obviously, they
  are why `duration` is wrong: fetching them is a blocking network call inside
  `stop()`. See `scripts/probe_reported_window.py`.
* **`train_acc` is not one quantity across the stacks.** Only Python/JAX and
  Python/TensorFlow write it. `test_acc` (a percentage) and `test_loss` are
  present for every stack and every epoch, and the conformance checker passes
  110 of 110 checks <!-- \vConformancePassing, \vConformanceChecks --> with
  0 failing <!-- \vConformanceFailing -->.
* **4 runs were replayed 2 days later** <!-- \vLateRuns, \vLateGapDays -->,
  all JAX <!-- \vLateEcosystems --> on VGG-16, in 3 configurations
  <!-- \vLateConfigurations -->. The originals failed on their first training
  batch: the shared classifier head carries a dropout layer, and the JAX
  training step never passed it a random stream. The defect was fixed before
  the remaining JAX VGG-16 runs executed, so every JAX VGG-16 run here comes
  from one source revision; the failed runs were deleted and replayed rather
  than repaired. The remaining 206 runs <!-- \vInterleavedRuns --> ran
  interleaved. The replays are not interleaved with the other ecosystems, so
  between-window drift is measured rather than assumed
  (`results/analysis/17_window_calibration.py`,
  `results/revision/tables/v2_window_calibration.*`): re-executing one
  configuration (PyTorch, ResNet-18 on Fashion-MNIST <!-- \vCalibConfig -->)
  in a third window moved training energy by 1.2 %
  <!-- \vCalibTrainDiffPct --> and inference energy by 3.2 %
  <!-- \vCalibInferDiffPct -->.
