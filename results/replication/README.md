# Raw measurement records

Every number in the manuscript comes from these four tables. They are the
campaign's raw output, flattened: nothing here is aggregated, filtered, rounded
or corrected.

Regenerate them from the campaign tree with

```bash
python3 scripts/consolidate_raw.py           # rewrite
python3 scripts/consolidate_raw.py --check   # verify against results/campaign_v2/
```

| File | Rows | One row is |
|---|---|---|
| `codecarbon.csv.gz` | 12,600 | one measured block, as the software estimator reported it |
| `counters.csv.gz` | 12,600 | the same block, as NVML and the RAPL package counters reported it |
| `metrics.csv.gz` | 6,300 | one epoch of one run: losses and test accuracy |
| `manifests.csv.gz` | 210 | one run: seed, versions, environment, as recorded at launch |

210 runs = 7 ecosystems × 2 architectures × 3 datasets × 5 repetitions.
12,600 blocks = 210 runs × 30 epochs × 2 phases (train, eval).

`SHA256SUMS` covers the four files. They are written with a zeroed gzip
timestamp, so the same records produce the same bytes on any machine and the
checksums are meaningful.

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
  because they happened; they are excluded from the accuracy analysis and
  flagged by `results/analysis/15_convergence.py`. Filter on `test_acc`, not on
  the ecosystem.
* **`longitude`/`latitude`/`country_name`** are CodeCarbon's IP geolocation of
  the measuring machine, resolved to a region rather than a place. They set the
  grid carbon intensity used for the CO₂e figures and nothing else.
* **The `Rust/tch` runs on five of the six blocks were re-executed later.**
  The originals trained on all-zero images through a defect of ours and were
  discarded; the evidence is kept at
  `results/revision/record/vgg_fashion_pipeline_defect.csv`. They ran in a
  later time window than the rest of the campaign, on the same machine and the
  same idle conditions. The manuscript states this as a threat.
