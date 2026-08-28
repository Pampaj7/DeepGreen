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
joinable on `(ecosystem, model, dataset, repetition, phase, epoch)`. The two
instruments ran over the *identical* window: the shared bridge starts
CodeCarbon and reads the counters inside one synchronous `START`, and stops and
reads inside one synchronous `STOP`. Any disagreement between the two tables is
therefore a property of the instruments, not of a timing difference — which is
the point of measuring twice.

The columns to compare are `energy_consumed` (kWh, CodeCarbon, whole machine
including a RAM model) against `hw_total_j` (J, counters, accelerator plus CPU
package). `cpu_energy` and `gpu_energy` in the CodeCarbon table are the
comparable subset; `ram_energy` has no counter equivalent because there is no
RAM energy counter to read.

## Caveats worth knowing before you use these

* **`duration` in the CodeCarbon table is not the phase.** Below about 11
  seconds it is the phase plus a constant 3.28 s. Power derived from it is
  understated by up to 6.7×. `duration_s` in the counters table is the phase.
* **Twelve of the 105 VGG-16 runs never left chance accuracy.** They are here
  because they happened; they are excluded from the accuracy analysis and
  flagged by `results/analysis/15_convergence.py`. Filter on `test_acc`, not on
  the ecosystem.
* **`longitude`/`latitude`/`country_name`** are CodeCarbon's IP geolocation of
  the measuring machine, resolved to a region rather than a place. They set the
  grid carbon intensity used for the CO₂e figures and nothing else.
* **The five `Rust/tch` VGG-16 / Fashion-MNIST runs are the re-executed ones.**
  The originals trained on all-zero images through a defect of ours and were
  discarded; the evidence is kept at
  `results/revision/record/vgg_fashion_pipeline_defect.csv`. They ran in a
  later time window than the rest of the campaign, on the same machine and the
  same idle conditions. The manuscript states this as a threat.
