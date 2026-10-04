# Raw measurement records: the accelerator-saturation cell

The contrast cell of spec S7: the same seven ecosystems, two architectures,
harness and instruments as the 210-run campaign in `results/replication/`, on
Imagenette (ten ImageNet classes) at 224×224 with batch 32 and 30 epochs, where
the accelerator rather than the host is the bottleneck. Every saturation number
in the manuscript (`paper/generated/numbers_saturation.tex`,
`tab_saturation.tex`) comes from these tables, through
`results/analysis/20_saturation.py`. The first five are the cell's raw output,
flattened: nothing here is aggregated, filtered, rounded or corrected. The last
three freeze the inputs of the utilisation analysis that a clone cannot
otherwise recover (see below).

Regenerate them from the campaign tree with

```bash
python3 scripts/consolidate_raw.py --campaign saturation           # rewrite
python3 scripts/consolidate_raw.py --campaign saturation --check   # verify against results/campaign_saturation/
```

and rebuild the run tree from them, or check that they rebuild it exactly, with

```bash
python3 scripts/restore_from_replication.py --campaign saturation --all              # into results/campaign_saturation/
python3 scripts/restore_from_replication.py --campaign saturation --check-roundtrip  # 4,480 of 4,480 files identical
```

`20_saturation.py` does not need the run tree: when
`results/campaign_saturation/` is absent it restores this package into a
temporary directory itself and analyses that, with identical results.

| File | Rows | One row is |
|---|---|---|
| `codecarbon.csv.gz` | 4,200 | one measured block, as the software estimator reported it |
| `counters.csv.gz` | 4,200 | the same block, as NVML and the RAPL package counters reported it |
| `metrics.csv.gz` | 2,100 | one epoch of one run: losses and test accuracy |
| `manifests.csv.gz` | 70 | one run: seed, versions, environment, as recorded at launch |
| `data_fingerprints.csv.gz` | 70 | one run's `data_fingerprint.csv`: summary statistics of the test split it read, for the data-parity check |
| `run_windows.csv.gz` | 70 | one run's whole-run window, [manifest `machine_state.utc`, `counters.csv` mtime], in unix seconds |
| `training_windows.csv.gz` | 2,100 | one training block's counter-bracketed interval, in unix seconds |
| `gpu_utilisation_excerpt.csv.gz` | 202,941 | one 1 Hz `nvidia-smi` sample, from `results/gpu_utilisation.csv`, columns unchanged |

70 runs = 7 ecosystems × 2 architectures × 1 dataset × 5 repetitions.
4,200 blocks = 70 runs × 30 epochs × 2 phases (train, eval).

`SHA256SUMS` covers every file. They are written with a zeroed gzip timestamp,
so the same records produce the same bytes on any machine and the checksums
are meaningful.

## Utilisation inputs: windows and the sampler excerpt

`20_saturation.py` reports accelerator utilisation and sampled board power over
each run's **training blocks**, on both sides of the contrast. A block's
interval is its `emissions_train_epoch<N>.csv` mtime minus CodeCarbon's
`duration` (the tracker's start, immediately before the counters are read),
lasting the counters' `duration_s`; a run's whole window ends at
`counters.csv`'s mtime. Git does not keep mtimes and neither does the restore
script, and the full 1 Hz record (about 100 MB) is not distributed. So:

* `run_windows.csv.gz` and `training_windows.csv.gz` hold those windows for all
  70 runs, exactly as `results/analysis/common.py` (`derive_windows`) computes
  them on the measurement host, where each run's mtimes are first checked
  against CodeCarbon's own `timestamp` column.
* `gpu_utilisation_excerpt.csv.gz` holds the record's rows inside any of those
  windows (±2 s), plus the record's first sample and the first sample after the
  last window closes, so the coverage rule ("a run counts only if its windows
  lie inside the record") decides every run as it does on the full record. All
  70 runs are covered.

The 32×32 side of the contrast reads the same three files from
`results/replication/`. The script uses the raw tree and the full record when
they are present and trustworthy, and refuses if either disagrees with these
files; with the tree absent or restored (mtimes lost), or the record absent, it
reads these files instead and produces identical tables and macros.
`consolidate_raw.py --campaign saturation --check` compares all three,
text-exact, against the raw tree and the record.

## Caveats worth knowing before you use these

* **`duration` in the CodeCarbon table is not the phase**, for the reason given
  in `results/replication/README.md`: it includes the tracker's blocking
  network look-ups at `stop()`. `duration_s` in the counters table is the phase.
* **`train_acc` is not one quantity across the stacks.** C++, Java, R, Rust and
  PyTorch do not write it, and among those that do, TensorFlow's VGG-16 reports
  a fraction where the others report a percentage. Nothing in the analysis uses
  it; `test_acc` is a percentage in every stack.
* **No run collapsed.** The lowest final test accuracy of the 70 runs is
  64.3 % (Java/DL4J, ResNet-18), far above the 10 % of chance on ten classes.
* **Every run used two loader threads** (`DEEPGREEN_LOADER_THREADS`, recorded
  in each manifest's `env`).
