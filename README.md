# DeepGreen AI :seedling:

This repository is the replication package for the paper:

> **Deep Green AI: Energy Efficiency of Deep Learning across Language–Framework Ecosystems**
> Leonardo Pampaloni, Marco Pagliocca, Enrico Vicario, Roberto Verdecchia
> University of Florence, Italy

📄 Manuscript: [`paper/paper.pdf`](./paper/paper.pdf) — source in [`paper/`](./paper)

---

## :pushpin: Overview

DeepGreen AI is an empirical study of how the choice of **language–framework
ecosystem** affects the **energy consumption of deep learning training and
inference** on a GPU. Seven ecosystems train and evaluate two convolutional
architectures on three image-classification datasets, under one written
experiment specification and one measurement contract, and every measurement
block is instrumented twice — once with hardware energy counters and once with
a software estimator reading the same window. The campaign is
**210 runs** <!-- \vRuns --> over **42 configurations** <!-- \vConfigurations -->
(7 ecosystems × 2 architectures × 3 datasets), each executed
**5 times** <!-- \vRepetitions --> as an independent process with a distinct
seed, yielding **12,600 doubly instrumented measurement blocks**
<!-- \vBlocks -->. It was executed on a single dedicated workstation between
30 August and 2 September 2026, with **4 runs** <!-- \vLateRuns --> replayed
**2 days** <!-- \vLateGapDays --> later; **206 of the 210**
<!-- \vInterleavedRuns, \vRuns --> ran interleaved rather than consecutively.
The measured blocks account for **40.4 MJ** <!-- \vCampaignEnergyMJ -->
(**11.2 kWh**) <!-- \vCampaignEnergyKWh --> at the accelerator and CPU package;
an honest campaign-level figure, including development and the time between
blocks, is larger, and the paper says so.

---

## :microscope: Research questions

- **RQ1:** How does language–framework ecosystem choice affect the energy efficiency of DL *training*?
- **RQ2:** How does language–framework ecosystem choice affect the energy efficiency of DL *inference*?
- **RQ3:** How do energy efficiency and execution time relate across ecosystems?
- **RQ4:** To what extent are the answers to RQ1–RQ3 determined by the measurement apparatus rather than by the ecosystems?

---

## :package: The seven ecosystems, and why the ecosystem is the unit

| Ecosystem | Language | Framework | Directory |
|---|---|---|---|
| Python/PyTorch | Python | PyTorch (pinned LibTorch 2.7.0, CUDA 12.8) | [`python/`](./python) |
| Python/TensorFlow | Python | TensorFlow 2.19.1 / `tf_keras` | [`python/`](./python) |
| Python/JAX | Python | JAX + Flax | [`python/`](./python) |
| C++/LibTorch | C++ | LibTorch (same pinned build) | [`cpp/`](./cpp) |
| Java/DL4J | Java | Deeplearning4j 1.0.0-M2.1 | [`Java/deepgreen-dl4j/`](./Java) |
| R/torch | R | `torch` 0.17.0 (bundles its own LibTorch) | [`R/`](./R) |
| Rust/tch | Rust | `tch` (same pinned build) | [`rust/`](./rust) |

The unit of comparison is the **ecosystem**, not the language. For a
practitioner a language is inseparable from the stack it implies — bindings,
framework, runtime, and the CUDA/cuDNN toolchain that stack supports — so each
ecosystem is treated as a single categorical treatment and no attempt is made
to decompose it into separate language, framework, runtime and toolchain
effects. That framing is also what makes the study auditable: four of the seven
stacks are LibTorch bindings, and **3 of them** <!-- \vCtrlStacks --> (Python/PyTorch,
C++, Rust) load the *same* exported TorchScript module on the *same* pinned
backend build. They form an internal control group, so the spread that survives
a common model and a common backend can be reported separately from the spread
that does not. (R/torch is a LibTorch binding too but bundles its own build and
cannot switch a loaded `script_module` between train and eval mode, so it
builds the same architecture itself; the reason is recorded in the
specification and in the paper's threats.)

**MATLAB/DLT is out of scope.** It was part of an earlier campaign, but the
Deep Learning Toolbox does not expose the training loop at the granularity the
specification requires, so its data pipeline, evaluation batching and
measurement boundaries cannot be brought into conformance. Its absence is not a
result about MATLAB. The `matlab/` sources remain for provenance.

---

## :bar_chart: Workload

Two architectures — [**ResNet-18**](https://arxiv.org/abs/1512.03385) and
[**VGG-16**](https://arxiv.org/abs/1409.1556) — on three datasets of increasing
difficulty: [**Fashion-MNIST**](https://github.com/zalandoresearch/fashion-mnist),
[**CIFAR-100**](https://www.cs.toronto.edu/~kriz/cifar.html) and
[**Tiny ImageNet**](https://github.com/rmccorm4/Tiny-Imagenet-200).

All three datasets are written at a common **32×32** three-channel resolution
**once, offline**, by the converters in [`dataloader/`](./dataloader), so that
each stack's own resize is a no-op and no framework's resampling filter can
enter the comparison. (`scripts/normalise_dataset_resolution.py` holds the
measurements that forced this, and rewrote the datasets already converted at
their native size.) Fixing the resolution also keeps the classifier head fixed
across datasets. It is a deliberate trade: at 32×32 the workload does not
saturate the accelerator. One declared contrast cell (spec S7) repeats both
networks on [**Imagenette**](https://github.com/fastai/imagenette) at 224×224,
batch 32, 70 runs in `results/campaign_saturation/`, to measure what that trade
costs (see *Limitations*).

The training recipe is identical everywhere and fixed in the specification:
Adam at `1e-4`, batch size 128, 30 epochs, no schedule, no weight decay,
inputs scaled to `[0, 1]` with no mean/std normalisation, and two data-loader
worker threads in every stack. A fixed-budget comparison of this kind does not
guarantee that every stack converges equally, which is why every stack also
records test accuracy after every epoch.

---

## :straight_ruler: The measurement contract

Energy is read from hardware counters at the boundaries of every measurement
block — NVML's accumulated-energy register for the accelerator and the RAPL
package-energy counters for the CPU, summed at a **board-and-package boundary**,
identical for every ecosystem. `CodeCarbon` runs over the *same* window as a
second, independent reading, in machine tracking mode at a one-second sampling
interval, configured once in `tools/codecarbon_config.json` and applied by all
seven stacks through one shared harness, `tools/deepgreen_bench.py` — the three
Python stacks importing it directly, the four non-Python stacks reaching it over
the line protocol in `tools/deepgreen_tracker.py`, with synchronous
acknowledgement, so a phase boundary and its counter read cannot drift apart.

A **measurement block** is one phase (training or evaluation) of one epoch, and
each block carries two energy readings, a duration, and the test accuracy
reached. CodeCarbon 2.x is refused at startup: it falls back to a constant
modelled CPU power when RAPL is unavailable, which would make much of the
reported energy a deterministic function of wall-clock time.

What "the same experiment" means across stacks is written down in
[`results/analysis/experiment_spec.md`](./results/analysis/experiment_spec.md)
(S1 model, S2 optimisation, S3 data pipeline, S4 backend, S5 measurement,
S6 replication) and enforced against the source, the built binaries and the
campaign's own metric files by **92 automated checks**
<!-- \vConformanceChecks --> in `scripts/check_consistency.py`
(**92 pass, 0 fail**) <!-- \vConformancePassing, \vConformanceFailing -->.

---

## :flashlight: Findings

- **Training.** At a common boundary, training the same model on the same data
  costs between **7.4× and 9.8×** <!-- \vSpreadTrainMin, \vSpreadTrainMax -->
  more energy on one ecosystem than another, depending on the architecture and
  dataset. **C++/LibTorch** <!-- \vSpreadTrainBest --> is cheapest in every
  ResNet-18 block; the cheapest VGG-16 stack changes with the dataset
  <!-- paper/generated/tab_train_energy.tex -->. Between-run variability is
  small — median coefficient of variation **0.49 %**
  <!-- \vCVTrainMedian -->.
- **Inference.** The spread is wider still, **23.1× to 38.7×**
  <!-- \vSpreadInferMin, \vSpreadInferMax -->, on blocks short enough that the
  apparatus is part of what is being measured. The paper names no cheapest
  inference stack, and neither does this README: an earlier execution of this
  same design reported one, and the ranking was an artefact of a measurement
  window that closed over work that had not finished. Reported inference
  rankings should be treated as claims about the apparatus until the window the
  energy was accumulated over is stated.
- **Most of the effect is between stacks that do not share a model and a
  backend.** Among the 3 stacks <!-- \vCtrlStacks --> that load the same
  exported module on the same pinned build — Python/PyTorch, C++ and Rust — the
  spread is **1.1×–1.6×** <!-- \vCtrlSpreadMin, \vCtrlSpreadMax --> against
  **9.8×** <!-- \vSpreadTrainMax --> across all seven.
- **Faster *is* greener here.** On counter-bracketed durations, energy and time
  rank the configurations almost identically: Spearman **ρ = 0.96** for training
  <!-- \vRhoTrain --> and **0.92** for inference <!-- \vRhoInfer -->, pooled over
  all 42 configurations <!-- \vConfigurations -->; within a single
  architecture × dataset × phase cell ρ runs **0.68 to 1.00** over the 12 cells
  <!-- \vCellRhoMin, \vCellRhoMax, \vCellRhoCells -->, so it is not an artefact
  of pooling workloads of different size. The contrary result is reproducible
  here as an instrument artefact: the two instruments
  agree on *energy* to **0.3 %** <!-- \vCCmeasDisagreementPct -->, but the
  duration the estimator reports beside that energy is not the interval the
  energy was accumulated over — **75 % of blocks** <!-- \vWindowPaddedPct -->
  carry seconds of tracker lifetime in which no energy was drawn, so power
  derived by dividing the one field by the other is understated by up to
  **13.0×** <!-- \vPowerUnderstatedWorst --> on the shortest blocks and exact on
  the longest.
- **Accuracy has to be read alongside energy.** On Fashion-MNIST all seven
  stacks converge to within **0.5 percentage points**
  <!-- \vAccBlockSpreadFashionMax -->, so there the difference is energetic
  rather than qualitative. On the harder datasets they do not: up to **6.5
  points** <!-- \vAccBlockSpreadTrainedMax --> on a single (architecture,
  dataset) cell.
- **Zero training collapses.** **0 of 105 VGG-16 runs**
  <!-- \vCollapseRuns, \vCollapsedVggRuns --> converged to chance accuracy in
  this campaign, with an exact one-sided 95 % upper bound of **4.2 %**
  <!-- \vCollapseRateUpperNinetyFive --> on the susceptible cells. In the first
  execution of this design **12 of 105** <!-- \vCollapseFirstVgg,
  \vCollapseFirstVggRuns --> collapsed and burned **10.7 %**
  <!-- \vCollapseFirstWastedPct --> of that campaign's training energy for
  nothing. The cause was not the ecosystem but the **weight initialiser**, which
  frameworks ship differently under one architecture name: in a controlled
  trial, **0 of 6** runs collapsed under He, **2 of 6** under Glorot and **4 of
  6** under Xavier <!-- \vInitCollapseHe, \vInitCollapseGlorot,
  \vInitCollapseXavier, \vInitCollapseTrials -->. Every stack now starts from one
  canonical seeded export with matched initialisers.
- **One precision flag outweighed every binding difference.** This campaign was
  executed twice, and the two executions differ in the precision policy of
  exactly one stack, which makes them an ablation of that policy with a control
  group. Across them, denying TF32 to that stack cost **3.14×**
  <!-- \vPrecisionPytorchRatioMin --> its ResNet-18 training energy on
  CIFAR-100 <!-- \vPrecisionGapDataset -->, while the **3**
  <!-- \vPrecisionControlStacks --> stacks comparable beside it — C++, Rust and
  R, whose policy did not change — moved by at most **1.3 %**
  <!-- \vPrecisionOthersMaxChangePct -->. The Python/PyTorch-versus-C++ ResNet-18
  gap the previous version of this paper leaned on falls from **3.94×** to
  **1.25×** <!-- \vPrecisionGapBefore, \vPrecisionGapAfter --> once the policy is
  common to all seven. A cross-ecosystem energy comparison that does not state
  its precision policy is not comparing ecosystems.

---

## :warning: Limitations the study declares

- **The accelerator is not saturated.** At 32×32 and these dataset sizes the GPU
  runs well below its board power limit for much of each epoch — the
  lowest-utilisation stack sits at **4.5–5.0 %** on ResNet-18
  <!-- \vLowestGpuUtilResnetPct --> and **12.1–13.4 %** on VGG-16
  <!-- \vLowestGpuUtilVggPct -->, over the 157 of 210 runs the 1 Hz sampler
  covers <!-- \vGpuUtilCoveredRuns, \vRuns -->. The campaign therefore compares
  *whole pipelines* — loading, dispatch and kernels together — rather than
  saturated kernels. The 224×224 contrast cell measures what saturation does
  (median training-block utilisation **89.2 %** <!-- \vSatUtilMedian -->, a
  different window from the whole-run figures above): the spread within the
  LibTorch family compresses, to **2.8×** and **1.3×** in training
  <!-- \vSatLibtorchSpreadTrainResnet, \vSatLibtorchSpreadTrainVgg -->, while
  across all seven it widens, to **15.3×** and **19.1×**
  <!-- \vSatSpreadTrainResnet, \vSatSpreadTrainVgg -->, carried by Java/DL4J,
  the one stack without a cuDNN path. It is one point: one dataset, one batch
  size, one card.
- **One host.** All runs are from a single-accelerator desktop (NVIDIA RTX 3090,
  AMD Ryzen 9 5900X, Ubuntu 26.04), which makes the host share of total energy
  smaller than it would be on a multi-socket server. Absolute figures are
  properties of this platform; the design, the specification and the instrument
  characterisation are not.
- **32×32 inputs, two CNNs, three datasets, one contrast cell at 224×224.**
  Results may not generalise to other model families, to transformers, or to
  other resolutions and batch sizes.
- **The design can say that ecosystems differ, not order any particular pair.**
  With 5 runs per group, **0 of 252** <!-- \vPairsSignificant, \vPairsTotal -->
  pairwise comparisons survive Holm correction — and cannot, at any effect size.
  Ordering claims rest on effect sizes and intervals.
- **Host DRAM is in neither counter**, so it is absent from every reported
  figure, and the board-and-package boundary is not whole-system energy.
- **MATLAB is out of scope** (above), and 4 of the 210 runs sit outside the
  interleaved window; a between-window calibration bounds what that is worth.

---

## :gear: Reproducing

### 1. Set up

```bash
git clone https://github.com/Pampaj7/DeepGreen.git
cd DeepGreen
./scripts/setup_environment.sh                 # .venv-deepgreen + instrument + datasets
./scripts/setup_environment.sh --datasets-only # datasets only
```

The script creates **one** environment, `.venv-deepgreen`, installs the pinned
instrument into it and materialises the three datasets as PNG folders under
`$DEEPGREEN_DATA` (default `./data`).

The three Python stacks cannot share an environment: torch's cu128 wheels and
TensorFlow's bundled CUDA libraries resolve to different versions of the same
shared objects, and whichever loses runs on the CPU while reporting a warning.
The campaign driver therefore expects `.venv-deepgreen` (PyTorch),
`.venv-tensorflow` and `.venv-jax`; the setup script does not build the latter
two, so they must be created by hand, from the exact package sets recorded in
[`requirements/tensorflow_env.yml`](./requirements) and
[`requirements/jax_env.yml`](./requirements) (conda exports of the environments
that ran the campaign; `.txt` pip freezes sit beside them). Point the driver at
different interpreters with `DEEPGREEN_PYTHON_TENSORFLOW` /
`DEEPGREEN_PYTHON_JAX` if you build them elsewhere.

Host-specific paths — data, models, CUDA runtime, LibTorch — are set by
`scripts/campaign_env.sh`, and everything else is derived in
[`tools/stack_environments.json`](./tools/stack_environments.json). The
non-Python stacks are built through their own build systems (CMake, Maven,
Cargo, R).

### 2. Run the campaign

```bash
source scripts/campaign_env.sh
python3 scripts/preflight.py --repetitions 5        # every job can actually run
python3 scripts/run_campaign.py --repetitions 5 --print-plan
python3 scripts/run_campaign.py --repetitions 5
```

The driver executes each configuration as independent, interleaved processes
with distinct seeds. Only the Python-hosted stacks can be launched directly from
it; `--print-plan` emits the exact command list, seeds and repetition indices so
the C++, Java, R and Rust stacks can be driven externally on the same schedule.
Runs land in `results/campaign_v2/`.

### 3. Analyse

```bash
./results/analysis/run_all.sh     # -> results/revision/{tables,figures}, paper/generated/
```

Runs the whole pipeline in dependency order and ends by rebuilding the raw
replication tables in `results/replication/`. Set `PYTHON=...` to choose the
interpreter.

The run tree `results/campaign_v2/` is not in version control — what ships is
the flattened package in [`results/replication/`](./results/replication), and
the two are a round trip. The saturation cell ships the same way, in
[`results/replication_saturation/`](./results/replication_saturation)
(`--campaign saturation` on both scripts). To analyse the published campaign without re-running
it, restore the run directories first:

```bash
.venv-deepgreen/bin/python scripts/restore_from_replication.py --all --dry-run
.venv-deepgreen/bin/python scripts/restore_from_replication.py --all
```

### 4. Check conformance

```bash
.venv-deepgreen/bin/python scripts/check_consistency.py           # prints its own summary
.venv-deepgreen/bin/python scripts/check_consistency.py --strict  # exit 1 on any FAIL
.venv-deepgreen/bin/python scripts/consolidate_raw.py --check     # package matches the raw tree
python3 scripts/verify_architecture_parity.py   # same parameter shapes, all 7 stacks
python3 scripts/verify_data_parity.py           # same pixels, all 7 stacks
```

The checker prints its own pass / fail / warn / skip counts; a conforming
repository ends with **0 fail**. Run it under the campaign interpreter, not the
system `python3`: three of its checks read the campaign's records through
pandas, and without pandas it declines to run and would otherwise report a clean
result on a repository it has not finished checking. The same applies to
`consolidate_raw.py` and `restore_from_replication.py`.

### 5. Build the manuscript

```bash
./paper/build.sh              # analysis + numbers + figures + PDF
./paper/build.sh --no-data    # compile only, reusing paper/generated/
```

Every quantity the manuscript quotes is generated by
`results/analysis/12_paper_numbers.py` into `paper/generated/numbers.tex` and
read in at build time — no number in the text is typed by an author. If a value
changes in the data it changes in the manuscript on the next build; if a value
disappears the build fails rather than printing a stale figure.

---

## :open_file_folder: Repository layout

```text
paper/                    # Manuscript (paper.tex, build.sh, generated/ numbers and tables)
REVISION_LOG.md           # What was found and fixed, in order, since the first submission
REVIEWERS_RESPONSE.md     # Point-by-point response to the reviews
SUBMISSION.md             # Submission checklist and what only the authors can do

tools/                    # Shared measurement contract: tracker bridge, hardware counters,
                          #   pinned CodeCarbon config, loader, init and BatchNorm helpers
scripts/                  # Campaign driver, preflight, conformance checker, parity checks,
                          #   TorchScript export, dataset normalisation, probes, setup
models/                   # Exported TorchScript modules + MANIFEST.json (params and hashes)

python/                   # Python stacks (PyTorch, TensorFlow, JAX)
cpp/                      # C++/LibTorch
Java/deepgreen-dl4j/      # Java/Deeplearning4j
R/                        # R/torch
rust/                     # Rust/tch
matlab/                   # MATLAB/DLT — out of scope, kept for provenance
dataloader/               # Dataset download and PNG conversion
data/                     # Materialised datasets (32x32) and their originals

results/replication/      # The campaign flattened into four raw tables + SHA256SUMS
results/replication_saturation/  # The 70-run saturation cell, same tables + data fingerprints and frozen utilisation inputs
results/analysis/         # Analysis pipeline (run_all.sh), experiment_spec.md, repetition_protocol.md
results/revision/         # Generated tables and figures
results/campaign_v2/      # The 210 run directories (counters, emissions, metrics, manifest)
results/calibration/      # Between-window drift calibration runs
requirements/             # Per-stack dependency pins
```

`results/campaign_v2/`, `results/calibration/`, the exported `.pt` modules and
the materialised datasets are too large for version control and are not in the
clone: the datasets come from `scripts/setup_environment.sh`, the modules from
`scripts/export_torchscript_models.py`, and the run directories from
`scripts/restore_from_replication.py`.

Two earlier campaigns are kept as inputs rather than as results.
`results/data/` holds the raw logs of the campaign the first submission
reported, audited by scripts `01`–`08` and `10`;
`results/campaign_v2_first_campaign/` holds the first execution of the present
design, which differs from this one in the precision policy of exactly one
stack and is therefore read by `18_precision_ablation.py` as an ablation, and
nowhere else.

`results/scripts/`, `results/plots/`, `results/tables/` and `results/old_data/`
are the original analysis and plotting code and its output. They are kept for
provenance and are **deprecated**: that code reads CodeCarbon's kWh column and
labels it Joules. Nothing in them should be used.

---

## :arrows_counterclockwise: What changed since the first submission

The first submission described eight ecosystems including MATLAB, a single
software instrument, a different host, and one run per configuration whose 30
epochs were treated as repeated measurements. Auditing that campaign against its
own source found that the stacks had not been running the same experiment
(two learning rates, divergent input shapes and evaluation batching, four
data-loader settings), that the instrument had been configured five different
ways across them, and that energy had been reported in kWh under a Joule label.
Everything in this repository is the replacement: one written specification,
92 executable conformance checks, one shared measurement contract, dual
instrumentation, and all 210 runs re-executed with 5 independent repetitions per
configuration.

- [**REVISION_LOG.md**](./REVISION_LOG.md) — the full record, defect by defect,
  in the order they were found and fixed, ending with what is still open.
- [**REVIEWERS_RESPONSE.md**](./REVIEWERS_RESPONSE.md) — the point-by-point
  response.
- The paper's own *A catalogue of defect classes* section reports the defects as
  classes, including the **12** <!-- \vDefectOurs --> we introduced ourselves
  while building or repairing this study.

The Zenodo record <https://zenodo.org/records/17734884> archives the **earlier**
campaign and is superseded by this one.

---

## :scroll: License

This project is released under the [MIT License](./LICENSE).
