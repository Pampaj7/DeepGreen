# Measurement protocol for the replicated campaign

This protocol replaces the one used for the first campaign. It exists because
three criticisms of the original submission cannot be answered from the existing
logs and require re-execution:

* each configuration was run once, and the 30 epochs of a run were treated as
  repeated measurements (reviewer 1 comment 5, reviewer 3 major comment 2);
* no model quality metric was written to disk, so energy was never normalised by
  the useful work produced (reviewer 1 comment 6, reviewer 3 major comment 3);
* the instrument was configured differently in each ecosystem (reviewer 1
  comments 8 and 9), which `results/analysis/01_data_audit.py` shows accounts for
  a large part of the reported differences.

## 0. Read counters, do not model

Energy comes from `nvmlDeviceGetTotalEnergyConsumption` and Intel RAPL, read
directly (`tools/hardware_counters.py`). CodeCarbon runs over the same window as
a cross-check. Validated on the replication machine: over comparable windows the
two agree to within 1% from 2 s to 40 s (`scripts/validate_instrument.py`).

RAPL needs read permission, once per boot:

    sudo chmod -R a+r /sys/class/powercap/intel-rapl

and it **wraps** — the range is in `max_energy_range_uj`, about 65.5 kJ here,
which at a few tens of watts is under half an hour. Ignoring that silently loses
whole epochs.

Report the boundary explicitly: this is chip-level energy, not wall-plug energy.

## 1. Instrument configuration — identical everywhere

Every ecosystem must use the values in `tools/codecarbon_config.json`:

| setting | value | why |
|---|---|---|
| CodeCarbon version | `>= 3.0` | 2.8.x falls back to a hardcoded constant CPU power (85 W × 0.5 = 42.5 W) when RAPL is unavailable, and models RAM power as a function of *installed* memory. With 503 GB installed this contributes a constant 231 W, making roughly two thirds of the reported energy a deterministic function of wall-clock time. |
| `tracking_mode` | `machine` | The first campaign set `process` for R only -- explicitly, in `R/scripts/tracker_control.py` -- which excluded almost all host energy and made R's totals incomparable. |
| `measure_power_secs` | `1` | The 15 s default exceeds the duration of 69% of the measured blocks. JAX already used 1 s while every other stack used the default. |
| `allow_multiple_runs` | `true` | One tracker per epoch per phase. |

The interpreter matters too: the Rust bridge spawns `python3` from PATH, so the CodeCarbon version measuring that stack depends on the ambient shell. Every bridge must name an explicit interpreter.

Record the resolved configuration with the results. `tools/deepgreen_bench.py`
writes it into `manifest.json` for every run and refuses to start under
CodeCarbon 2.x.

## 2. Toolchain — held constant where it can be

The first campaign spanned CUDA 11.6 (Java) to 12.8 (C++) with different cuDNN
versions, so "C++ is most efficient" was confounded with "C++ has the newest
backend" (reviewer 1 comment 8). For the replicated campaign:

* pin one CUDA and one cuDNN version across every stack that permits it, and
  record the resolved versions per run in `manifest.json`;
* **LibTorch 2.7.0 everywhere**: `torch==2.7.0` for Python, the 2.7.0+cu128
  archive for C++ (already the default), `tch = "0.20"` for Rust, and an R
  `torch` package that bundles 2.7.x. `scripts/check_consistency.py` fails the
  build when these disagree;
* for the four LibTorch stacks (Python/PyTorch, C++/LibTorch, R/torch, Rust/tch)
  pin the **same LibTorch build**. These four share a backend and therefore form
  the study's internal control group: any spread between them is host-side, and
  aligning their backend isolates binding overhead from version differences;
* pin numerical precision explicitly (`precision="fp32"` by default) rather than
  accepting each framework's default. On PyTorch this also means disabling TF32
  matmuls, which are enabled by default on Ada-class hardware and are a
  different precision policy, not a different ecosystem.

Where a version cannot be aligned, report the residual difference in the
limitations rather than absorbing it into the ecosystem effect.

## 3. Replication

* **5 independent repetitions minimum** per (ecosystem, model, dataset); 10 if
  the schedule allows. Each repetition is a fresh process with a distinct seed.
* **Interleave** repetitions. `scripts/run_campaign.py` iterates
  repetition-major and shuffles configuration order inside each repetition with
  a fixed seed, so thermal drift and background load are spread across
  conditions instead of aliasing onto whichever ecosystem ran last.
* **Cool down** between jobs (default 60 s) so runs start from a comparable
  thermal state.
* The **independent run is the unit of analysis**. Aggregate epochs to one value
  per run first, then compute statistics across runs. Never test across epochs.

## 4. Quality metrics — mandatory

Every ecosystem writes `metrics.csv` with one row per epoch containing at least
`train_loss`, `test_loss` and `test_acc`. Most stacks already computed accuracy
and only printed it. Without it, "energy efficiency" cannot be distinguished
from "did less work": a stack that converges to a worse model for less energy is
not more efficient.

Report, per configuration:

* final test accuracy (mean and sd across repetitions);
* accuracy per kilojoule;
* energy to reach a fixed target accuracy, which is the decision-relevant
  quantity for a practitioner and does not depend on the arbitrary 30-epoch
  budget.

`results/analysis/09_campaign_v2.py` computes all three.

## 5. Workload

The audit shows mean GPU power between 27% and 56% of the L40S board limit, and
30–69% of the measured energy being host-side (reviewer 1 comment 15). At that
utilisation the study measures data loading and dispatch overhead more than
deep-learning computation. The replicated campaign should therefore add at
least one configuration that actually loads the accelerator — native-resolution
inputs, a larger batch, and a model whose arithmetic intensity is representative
of current workloads — and report GPU utilisation alongside energy so that
readers can see which regime each result belongs to.

## 6. Data pipeline held constant

`results/analysis/10_implementation_audit.py` shows that the eight ecosystems did
not run the same experiment. The largest single difference was how many CPU
threads decode and feed images:

| ecosystem | first campaign | mean s/epoch |
|---|---|---|
| Rust/tch | rayon over all 96 cores | 8.8 |
| Python/PyTorch, C++/LibTorch | 2 DataLoader workers | 12.6, 12.7 |
| Python/JAX, MATLAB/DLT, Python/TensorFlow | single-threaded generator | 13.7, 35.1, 39.8 |
| Java/DL4J | AsyncDataSetIterator, 2 | 53.6 |
| R/torch | `num_workers = 0` | 102.3 |

Spearman between loader threads and epoch duration is −0.73 (p = 0.04) over
eight points. With the GPU at 24–56% of its power limit and duration explaining
essentially the whole energy spread, the published ranking is largely a ranking
of data-pipeline configurations.

For the replicated campaign:

* **one loader-thread count for every ecosystem** (2 by default), set explicitly
  rather than left to a framework default. `DEEPGREEN_LOADER_THREADS` controls
  the Rust pool; R was moved from 0 to 2; the Keras generators remain the hard
  case and their effective concurrency must be measured, not assumed;
* **one learning rate** (1e-4) — TensorFlow and Rust used 1e-3;
* **one input scaling** (raw [0,1]) — only Rust normalised;
* **one dataset loading strategy** per language, chosen explicitly. In C++ this
  is now `src/dataset/ImageFolder.h`, lazy by default;
* report the loader configuration in the paper as a controlled factor, and
  ideally sweep it (1, 2, 4, 8 workers) for at least one ecosystem, so readers
  can see how much of the effect it accounts for.

## 7. The machine must be idle

`tracking_mode: machine` measures the whole host, not the process. Anything else
running during a tracked block is counted as that block's energy: a package
download, a compile, an editor indexing, a backup. With the GPU at a fraction of
its power limit and host terms making up a large share of the total, unrelated
CPU and disk work is not a rounding error.

This is easy to violate without noticing. During this revision the first attempt
at the replicated campaign overlapped a 2.5 GB LibTorch download and a CUDA
toolkit install; those runs were discarded and the campaign restarted on an idle
machine.

Before starting:

* finish every download, build and environment change first;
* close editors, language servers, sync clients and container daemons that do
  background work;
* check the idle baseline (`nvidia-smi`, a minute of CodeCarbon with no
  workload) and record it alongside the results, so a reader can see what the
  floor was;
* do not run analysis or plotting on the measurement machine while a campaign is
  in flight.

The first campaign gives no way to check this: nothing recorded what else the
host was doing. Record it this time.

## 8. Validation

The custom daemon-based CodeCarbon controller used for the non-Python stacks is
presented as a robustness improvement but was never validated (reviewer 1
comment 13). Before the replicated campaign, run one configuration three ways —
the daemon controller, plain in-process CodeCarbon, and an external wall meter or
`perf`/NVML sampling — and report the agreement. Without that, the controller is
an uncontrolled difference between the Python and non-Python stacks.
