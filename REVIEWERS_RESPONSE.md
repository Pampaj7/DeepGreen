# Response to the JSS reviewers (JSSOFTWARE-D-26-00842)

*Deep Green AI: Energy Efficiency of Deep Learning across Language-Framework Ecosystems*

This document records, comment by comment, what was changed in the replication
package. It distinguishes three outcomes:

* **FIXED** — resolved in this repository; the artefact is named.
* **MANUSCRIPT** — the analysis is done here and the numbers are ready, but the
  change lands in the paper text, which is not in this repository.
* **REQUIRES RE-EXECUTION** — cannot be answered from the existing logs. The
  code and protocol are in place; the measurement server must run them.

Everything below is reproducible with `results/analysis/run_all.sh`.

---

## Summary of what the re-analysis found

Three findings go beyond the reviewers' comments and change how the results
should be read.

**1. The measurement instrument was not the same across ecosystems.** The
campaign used two CodeCarbon major versions, two tracking modes and two sampling
intervals:

| ecosystem | CodeCarbon | tracking mode | CPU power | RAM power | host share of energy |
|---|---|---|---|---|---|
| Python/PyTorch, JAX, TensorFlow, Rust/tch | 2.8.4 | machine | 42.5 W constant (CodeCarbon fallback, 85 W × 0.5) | 188.84 W constant (0.375 W/GB × 503.6 GB installed) | 64–70% |
| C++/LibTorch, Java/DL4J, MATLAB/DLT | 3.0.4 | machine | ~37 W measured (RAPL) | 70 W constant cap | 43–53% |
| R/torch | 2.8.4 | **process** | 42.5 W constant | ~0 W (untracked) | 31% |

For the 2.8.4 stacks, `cpu_energy + ram_energy = 231.3 W × duration` to within
0.5%. Roughly two thirds of their reported energy is therefore a deterministic
linear function of wall-clock time, computed from a constant that depends on how
much RAM is *installed* in the server. Cross-ecosystem comparison of raw
`energy_consumed` compares instrument configurations as much as ecosystems.

The re-analysis reports every result under three boundaries: as measured,
**GPU only** (NVML energy counter, the same instrument everywhere), and
**harmonised** (GPU counter plus one uniform 107 W host model). See
`results/revision/tables/audit_measurement_boundary.md`.

**2. The four LibTorch stacks span 98–100% of the total reported spread.**
Python/PyTorch, C++/LibTorch, R/torch and Rust/tch run the same LibTorch
kernels. Restricting the comparison to those four reproduces essentially the
entire effect (`stats_libtorch_control.md`). Whatever is being measured is
host-side — binding layer, data loading, dispatch, toolchain version — and not a
property of the language or of the deep-learning computation.

**3. The eight ecosystems did not implement the same experiment.** Reading the
source of all eight stacks (`10_implementation_audit.py`) found divergences the
manuscript states do not exist:

| aspect | first campaign |
|---|---|
| learning rate | 1e-4 in six stacks, **1e-3** in Python/TensorFlow and Rust/tch — the paper claims a common learning rate |
| data-loader threads | **0** (R/torch), 1 (JAX, TensorFlow, MATLAB), 2 (PyTorch, C++, Java), **96 = all cores** (Rust, via rayon) |
| input scaling | raw [0,1] in seven stacks, **per-channel mean/std normalisation** in Rust |
| C++ off-the-shelf path | all six `*_imported` CMake targets commented out; `train_model.h` includes `dataset/ImageFolder.h`, which does not exist |
| Rust dataset paths | hard-coded `/home/pampaj/DeepGreen/...` in every binary |

Spearman between loader threads and mean epoch duration is **−0.73 (p = 0.04)**
across the eight ecosystems: the fastest stack decodes on 96 cores, the slowest
on none, and they differ by 11.6×. Together with the GPU-load result, the most
parsimonious reading of the headline ranking is that it ranks **data-pipeline
configurations**, not ecosystems.

All five divergences are now aligned in the source (`impl_alignment_applied.md`).

**4. The best-performing ecosystem was not solving the same task.** Reading the
Rust sources found three defects that all push in the direction of the reported
result (`impl_structural_findings.md`):

* **Fashion-MNIST was trained at 28x28x1.** The Rust loader applied no resize and
  no channel replication, so training ran on the native grayscale image while
  *its own evaluation loop* — and every other ecosystem, in both phases — used
  32x32x3. The conv1 input volume during training was 0.26x everyone else's.
* **`resnet_tiny.rs` trained Tiny ImageNet at its native 64x64** while
  `vgg_tiny.rs` passed `Some(32)` and every other ecosystem used 32x32: four
  times the spatial work for the same nominal configuration, inside one ecosystem.
* **All six Rust binaries evaluated one image at a time** (`iter_batches(1)`)
  while the other seven ecosystems evaluated at batch 128. Batch-1 GPU inference
  is launch-overhead bound.

The published numbers line up with the defects exactly:

| phase | Rust / median of the other stacks |
|---|---|
| training | **0.07x** (ResNet-18 / Fashion-MNIST) … 0.61x |
| inference | 1.51x – 1.95x (CIFAR-100, Fashion-MNIST) |

Rust/ResNet-18/Fashion-MNIST is the cheapest cell in the entire campaign, 14x
below the median, and it is precisely the cell with the smallest input. The
manuscript's **train/inference ranking reversal** — reported as a finding, and
already contested by reviewer 1 comment 14 as not novel — is reproducible for
Rust as a pure artefact of input shape and evaluation batching.

All three are fixed (`rust/src/datasets/fashion.rs`, `rust/src/bin/resnet_tiny.rs`,
all six binaries) and `cargo check --bins` is clean, but the affected numbers
cannot be salvaged: that ecosystem must be re-measured.

**5. Under a common measurement boundary, time predicts energy.** Cross-ecosystem
Spearman correlation between the energy ranking and the time ranking is 1.00 for
inference (zero discordant pairs out of 28) and 0.98 for training (one
discordant pair), against 0.90 and 0.95 as measured. Within an ecosystem the
median R² of energy on duration is 0.98. The manuscript's RQ3 conclusion is not
supported once the instrument is held constant.

---

## Reviewer 1

### Comment 1 — Motivation and grounding; "first large-scale study" vs "case study"

**MANUSCRIPT.** We accept both points. The framing claim is not demonstrated by
this design, and the audit gives a concrete reason: the workload runs the GPU at
27–56% of its board limit, so what is measured is largely host-side overhead
(comment 15). The revised framing is that ecosystem choice matters at the
*binding and runtime* layer, evidenced within a shared backend, and is not
positioned as a first-order lever at the scales where the field's energy problem
lives. "First large-scale empirical study" is withdrawn; the work is described
as a case study throughout, consistent with the abstract.

### Comment 2 — Workload choice

**MANUSCRIPT + REQUIRES RE-EXECUTION.** Accepted as a limitation of the current
data and addressed in the protocol. `results/analysis/repetition_protocol.md`
§5 requires at least one configuration that actually loads the accelerator —
native resolution, larger batch, representative arithmetic intensity — and
requires GPU utilisation to be reported next to energy so that readers can see
which regime a result belongs to. The 32×32 downsampling of Tiny ImageNet, which
removes the property that makes it demanding, is stated explicitly in
`dataset_factors.md`.

### Comment 3 — Ecosystem framing and shared backends

**FIXED, and the control group is now an actual control.** In the submitted
package the four "shared backend" stacks shared neither the model nor the
backend build: Python used torchvision, C++ a hand-written port (its
TorchScript path was commented out and did not compile), Rust another
hand-written port, R torchvision-for-R — over LibTorch 2.6.0, 2.7.0, 2.1.0 and a
bundled build respectively. `scripts/export_torchscript_models.py` now exports
one traced torchvision module that C++, Rust and R all load, and the whole group
is pinned to **LibTorch 2.7.0** (`torch==2.7.0`, the 2.7.0+cu128 archive the C++
build already fetched, `tch = "0.20"`, R `torch` 0.17.0).

Verified, not merely compiled:

```
$ cargo run --example load_shared_module
  resnet18 cifar100    11227812 params  out [2, 100]  ok
  ... all 6 shared modules load, forward and train through the VarStore

$ Rscript R/scripts/load_shared_module_test.r
  ... all 6 shared modules load and forward in R/torch
```

The Rust parameter counts match the Python export exactly, so the graph is
identical rather than equivalent. `scripts/check_consistency.py` fails when the
LibTorch versions disagree — and since a module exported by a newer torch cannot
load into an older LibTorch, a mismatch is a hard failure rather than silent
drift.

**Original response:** `03_statistics.py` adds the shared-backend control the reviewer asks
for. The four LibTorch stacks span 4.4× (as measured) to 9.9× (harmonised) of
energy and 11.6× of time, which is 98–100% of the full eight-stack spread on a
log scale. This is direct evidence that the variation is host-side. The
contribution is reframed accordingly: binding, runtime and toolchain overhead,
measured at ecosystem granularity, not language energy efficiency.
Table: `results/revision/tables/stats_libtorch_control.md`.

### Comment 4 — Energy unit mislabelled (also Reviewer 3, major comment 1)

**FIXED.** The reviewer is correct. `energy_consumed` is kWh. `01_data_audit.py`
demonstrates it three independent ways:

1. `emissions / energy_consumed = 0.3306` — read as kWh this is 331 gCO₂eq/kWh,
   the Italian grid intensity CodeCarbon used; read as J it would imply
   3.3 × 10⁸ gCO₂eq per Joule.
2. Implied mean power read as kWh: 120–511 W, median 328 W, plausible for this
   server. Read as J: 3 × 10⁻⁵ W.
3. `energy_consumed = cpu_energy + gpu_energy + ram_energy` to 1 part in 10¹²,
   so all four columns share the unit.

The conversion now happens in exactly one place (`common.py`, `J_PER_KWH`), and
every derived column carries its unit in its name (`energy_j`, `duration_s`,
`power_w`). All tables and figures are regenerated in Joules.

The per-component CPU/GPU/RAM breakdown the reviewer asks for is in
`table_component_breakdown.md` and `fig_component_breakdown.png`. It is what
exposed finding 1 above.

### Comment 5 — No statistical basis; single run; outliers

**Partly FIXED, partly REQUIRES RE-EXECUTION.**

*What the existing data supports.* `03_statistics.py` reports Kruskal-Wallis with
ε², pairwise Mann-Whitney with Holm correction, and Cliff's delta, and
`02_main_tables.py` reports medians, IQR, CV and bootstrap intervals. Every one
of these is labelled with the caveat that the intervals are over the 30 epochs of
a *single* run: they describe within-run epoch dispersion, not between-run
uncertainty, and the effective number of independent observations per
configuration is 1. The caveat text is in `stats_caveat.md` and is printed by the
script itself so it cannot be silently dropped.

*Outliers.* 158 rows (5.5%) report a sampled GPU power of 0 W and 27 rows (0.9%)
exceed the 350 W board limit, up to 3054 W. These are confined to CodeCarbon's
*sampled* `*_power` columns, which are unreliable because 69% of the measured
blocks are shorter than the 15 s default sampling period. The *energy* columns
come from the NVML counter: GPU power derived from them spans 57–283 W with no
value out of range. All results are therefore computed from the energy columns,
medians are reported alongside means, and the affected rows are flagged rather
than silently averaged in (`audit_outliers.md`).

*What it does not support.* No claim of a statistically significant *ecosystem*
difference can be made from one run per configuration.
`scripts/run_campaign.py` implements 5–10 interleaved independent repetitions
with distinct seeds and cooldowns; `09_campaign_v2.py` analyses them with the
run as the unit. This must be executed on the measurement server.

### Comment 6 — No accuracy reported (also Reviewer 3, major comment 3)

**FIXED in code, REQUIRES RE-EXECUTION for the numbers.** The reviewer is right
that the code computes accuracy — PyTorch, JAX, TensorFlow, MATLAB and Rust all
do — and right that it was never persisted. The Python stacks now write
`metrics.csv` with `train_loss`, `test_loss` and `test_acc` per epoch through
`tools/deepgreen_bench.py`; the same contract is specified for the other stacks
in `repetition_protocol.md` §4. `09_campaign_v2.py` computes final accuracy,
accuracy per kilojoule, and **energy to reach a target accuracy**, which is the
decision-relevant quantity and does not depend on the arbitrary 30-epoch budget.
No accuracy-normalised claim is made until that campaign has run.

### Comment 7 — Not like-for-like at the implementation level

**FIXED in code; MANUSCRIPT for the claim.** The reviewer is right, and reading
all eight ecosystems found the problem reaches further than the paper admits: it
is not only that off-the-shelf implementations differ in layer defaults and data
loading, it is that the *training protocol itself* differs. Learning rate,
input scaling and data-loader thread count are not common across the eight
stacks; see finding 3 above and `impl_protocol_divergence.md`. The C++
off-the-shelf path, which the methodology section relies on, does not build.

All of it is now aligned in the source — one learning rate, one input scaling,
one loader-thread count, one C++ dataset loader — and recorded in
`impl_alignment_applied.md`. The changes to the non-Python stacks are
source-level only: cargo, maven, Rscript and MATLAB are not available in the
revision environment, so they must be built and smoke-tested on the measurement
server.

Accepted without qualification. The study compares
implementations, and part of the measured difference is optimisation maturity
and community investment rather than an intrinsic ecosystem property. This is
now stated as a bound on what the design can establish, not as a threat that has
been mitigated. It is reinforced by the LibTorch control result (comment 3): the
spread survives inside a single backend, which is exactly what "implementation
differences" predicts.

### Comment 8 — CUDA/cuDNN/precision not held constant

**Partly FIXED, partly REQUIRES RE-EXECUTION.** The confound is real, and the
audit adds one the reviewer did not have: the CodeCarbon version itself differs
across stacks and dominates the host-energy term. Actions:

* `repetition_protocol.md` §2 pins one CUDA and one cuDNN version where the stack
  permits, and pins the **same LibTorch build** across the four LibTorch stacks
  so binding overhead is isolated from backend version;
* precision is pinned explicitly (`precision="fp32"`) instead of accepting each
  framework's default, and on PyTorch this includes disabling TF32 matmuls, which
  are on by default on this hardware and are a different precision policy;
* the resolved CUDA, cuDNN, framework and CodeCarbon versions are written into
  `manifest.json` for every run, so the residual differences are reportable
  rather than invisible.

Until re-execution, "C++/LibTorch is most efficient" remains confounded with
"C++/LibTorch has the newest backend and a different CodeCarbon version", and is
reported as such.

### Comment 9 — Sampling interval not stated

**FIXED.** It was not stated because it was not the same. Read from the source of each
bridge: Rust, R and Python/JAX used `measure_power_secs=1`; C++, Java, MATLAB,
Python/PyTorch and Python/TensorFlow used the 15 s default. R additionally set
`tracking_mode="process"`. And the Rust bridge spawns `python3` from PATH, so the
CodeCarbon *version* measuring that stack is whatever the ambient shell resolves
-- which is how one campaign came to mix CodeCarbon 2.8.4 and 3.0.4. 69% of the measured
blocks are shorter than 15 s, and the shortest is 0.51 s. GPU energy comes from
the NVML counter in CodeCarbon 3.x and is robust to this; the sampled power
columns are not, which is where the 0 W and >350 W readings come from.

`tools/codecarbon_config.json` now pins `measure_power_secs = 1`,
`tracking_mode = machine` and CodeCarbon ≥ 3.0 for every ecosystem, and
`deepgreen_bench.py` refuses to start under 2.x. Validation against an external
meter is specified in `repetition_protocol.md` §6 and is still outstanding.

### Comment 10 — Efficiency Index (also Reviewer 3, major comment 4)

**FIXED.** `05_efficiency_index.py` does all three things asked for:

* an α sweep over {0, 0.25, 0.5, 0.75, 1}. The ranking moves by at most one
  position over the whole range in training and not at all in inference — the
  index is insensitive to α, confirming the reviewer's reading;
* a **single** normalisation (divide by the minimum, so 1.0 = best) used in both
  the tables and the heatmap, resolving the inconsistency with Figure 4;
* the Pareto set reported directly, which is what we now recommend using.
  Spearman between normalised energy and normalised time is 0.98 (training) and
  1.00 (inference), so the composite averages two near-collinear quantities.

Recommendation carried into the manuscript: drop the composite, report the
Pareto frontier.

### Comment 11 — Industrial extrapolation (also Reviewer 3, major comment 6)

**FIXED in analysis, MANUSCRIPT for placement.** `07_carbon_and_scale.py` makes
the boundary mismatch explicit: the manuscript multiplies an assumed per-GPU
board power by a ratio derived from CPU+GPU+RAM energy, which are different
boundaries. The ratio that may legitimately be applied to a per-GPU power budget
is the GPU-only one. The three candidate ratios are 7.28× (as measured), 8.83×
(harmonised) and 9.30× (GPU only).

We agree the section should move to an appendix, be restricted to
ResNet-18/VGG-16 at 32×32 on a single L40S, be stated in GPU energy only, and be
given as a range. The monetary figures are withdrawn.

### Comment 12 — Descriptive answers, no mechanism

**Partly FIXED.** `04_energy_vs_time.py` performs the decomposition the reviewer
points at: energy = mean power × duration. 98% (inference) and 107% (training)
of the log spread is attributable to duration, and the mean-power spread is only
1.3–1.6×. Combined with the GPU-load audit, the LibTorch control and the implementation
audit, the mechanism is now concrete rather than hypothesised: the stacks differ
in how long a step takes at low GPU utilisation, and the dominant reason is how
many CPU threads decode and feed images — Spearman −0.73 (p = 0.04) between
loader threads and epoch duration, with Rust on 96 cores at one end and R on
zero workers at the other. This is a data-pipeline story, not a
kernel-efficiency story and not a language story. Layer-level and function-level profiling remains future work and is now
described as necessary rather than optional.

### Comment 13 — Presentation, and the unvalidated daemon controller

**Partly FIXED.** The 7.6× / 7.3× discrepancy is confirmed: the correct value is
7.28×, so 7.3 is right and 7.6 is the error (`table_headline_spreads.md`). The
cover-letter journal name, the 7.6× figure and the "indipendent" typo are
MANUSCRIPT fixes. Validation of the daemon-based CodeCarbon controller against
plain in-process CodeCarbon and an external meter is specified in
`repetition_protocol.md` §6 and is REQUIRED before the controller can be
presented as a robustness improvement.

### Comment 14 — Related work and positioning

**MANUSCRIPT.** Accepted. Alizadeh and Castor, *Green AI: A Preliminary Empirical
Study on Energy Consumption in DL Models Across Different Runtime
Infrastructures*, is directly adjacent and will be cited and positioned against.
We also accept that the training/inference ranking reversal ([26]) and the
energy–time divergence ([18], [28]) are established results, and neither is
claimed as novel. Given finding 3 above, our own phase-reversal and RQ3 results
are additionally weakened once the instrument is held constant, and are reported
as such rather than as confirmations. The remaining contribution is narrower and
is stated narrowly: a controlled comparison of binding and runtime overhead
across eight stacks, four of which share a backend.

### Comment 15 — The workload does not exercise the GPU

**FIXED (analysis), REQUIRES RE-EXECUTION (workload).** Confirmed and quantified.
Mean GPU power derived from the energy counter is 83–197 W against a 350 W board
limit — 24–56% — and never approaches the limit in any configuration. GPU energy
is 30–69% of the measured total; the remainder is host-side.
`fig_gpu_load.png` and `audit_gpu_load.md` report this per ecosystem and phase,
and `fig_component_breakdown.png` shows the split. The protocol requires a
GPU-saturating configuration in the replicated campaign.

---

## Reviewer 3

### Major comment 1 — Energy units

**FIXED.** See Reviewer 1 comment 4. The reviewer's inference is exactly right:
8.79 × 10⁻⁴ read as kWh is 3164 J, which matches the derived mean power. The
full pipeline audit the reviewer asks for — native unit, every conversion, units
stored in the replication data — is `01_data_audit.py` and
`audit_units.md`.

### Major comment 2 — No independent run-level repetitions

**Accepted; REQUIRES RE-EXECUTION.** We agree this is not a future-work point.
Until the replicated campaign runs, every comparative statement in the paper is
demoted to a descriptive observation about one execution campaign, and the
within-run intervals are labelled as such wherever they appear. The protocol,
driver and analysis are in place (`repetition_protocol.md`,
`scripts/run_campaign.py`, `09_campaign_v2.py`).

### Major comment 3 — No accuracy or convergence

**Accepted; FIXED in code, and it immediately caught two defects.** Persisting
accuracy is not only needed to normalise energy by useful work -- it is the only
thing that would have revealed that the TensorFlow stack was evaluating with
`training=True` pinned into its functional graph (21% test accuracy against 83%
on train, 86% once removed), and that the Fashion-MNIST class `T-shirt/top` had
produced a nested directory that different loaders read differently. Neither is
visible in energy or runtime.

With the quality metric in place, the three Python ecosystems now agree to within
one percentage point after one epoch on Fashion-MNIST (PyTorch 87.11%, JAX
86.46%, TensorFlow 86.06%), which is the evidence that they solve the same
problem.

**Original response:** See Reviewer 1 comment 6. We accept the framing:
without a quality measure the differences cannot be called "energy efficiency"
without qualification, and the fixed 30-epoch budget does not establish
equivalent learning outcomes — particularly once precision policies differ. The
harness now persists accuracy and loss per epoch, and the analysis computes
energy-to-target-accuracy in addition to accuracy per kilojoule.

### Major comment 4 — RQ3 stronger than the analysis supports

**FIXED, and the conclusion changes.** `04_energy_vs_time.py` replaces the
qualitative quadrants with: within-ecosystem Pearson/Spearman per phase
(median R² = 0.98); cross-ecosystem Spearman and Kendall between the energy and
time rankings under each energy definition, with the discordant pairs listed by
name; and the power/duration decomposition.

The result does not support the manuscript's claim. Under the harmonised
boundary there are zero discordant pairs in inference and one in training. The
specific examples the manuscript gives — R versus Java, MATLAB versus PyTorch —
are precisely the pairs that disappear once the CodeCarbon host power model is
held constant, i.e. they are instrument artefacts. RQ3 is rewritten: execution
time is a good proxy for energy in this workload, and the apparent exceptions
came from the measurement setup.

Efficiency Index: see Reviewer 1 comment 10.

### Major comment 5 — Figure 4 cannot show growth across datasets

**FIXED.** Correct. `06_dataset_scaling.py` reports absolute energy per epoch and
a single common reference cell instead of per-column normalisation
(`fig_dataset_absolute.png`, `dataset_absolute_energy_*.md`).

The claim also weakens under the correct scale. Training energy grows 1.79×
from Fashion-MNIST to Tiny ImageNet, but per training image it is 114.8, 109.3
and 123.1 mJ respectively — essentially flat. The effect is dataset *size*, not
difficulty. `dataset_factors.md` lists what "complexity" conflates: training-set
size, class count, channel count and native resolution all differ
simultaneously, and Tiny ImageNet is downsampled 64×64 → 32×32. The design
cannot isolate any one of them, and we now say so instead of claiming a scaling
result.

### Major comment 6 — Industrial simulation

**Accepted.** See Reviewer 1 comment 11. The boundary mismatch the reviewer
identifies is confirmed numerically and the monetary claims are withdrawn.

### Major comment 7 — Number of experimental runs

**FIXED.** The reviewer's arithmetic is right and 7,200 is wrong. The campaign
contains **2,880** tracked measurement blocks: 8 ecosystems × 2 models ×
3 datasets × 30 epochs × 2 phases, i.e. 1,440 training and 1,440 inference
blocks, from 48 configurations run **once** each. The explicit table the reviewer
asks for is `audit_design.md`, generated from the data rather than restated.

The carbon figure changes accordingly: the tracked blocks total **4.02 kWh** and
**1.33 kg CO₂eq** at 331 gCO₂eq/kWh, not 3.3 kg. This is a lower bound — it
excludes compilation, dataset preparation, idle time and discarded runs — and is
now stated as such (`campaign_carbon_footprint.md`).

### Major comment 8 — Language versus ecosystem

**FIXED.** One unit throughout: **language-framework ecosystem**, written as
`Python/PyTorch`, `C++/LibTorch`, `Java/DL4J`, `R/torch`, `Rust/tch`,
`MATLAB/DLT`, `Python/TensorFlow`, `Python/JAX`. The mapping is in
`common.py::ECOSYSTEM` and is applied to every table, figure axis and caption; no
figure axis is labelled "Language" any more. Statements are phrased as
"Rust/tch versus Java/DL4J". The count is eight ecosystems over six languages,
stated consistently.

### Minor comments

| comment | outcome |
|---|---|
| 7.6× vs 7.3× | **FIXED.** The correct value is 7.28×. All ratios are regenerated from one script (`table_headline_spreads.md`). |
| Related-work positioning and comparison table | **MANUSCRIPT.** A comparison table over workload, languages, frameworks, hardware, energy measurement and contribution will be added, positioned against [26], [27], [18], [28] and Alizadeh & Castor. |
| CodeCarbon sampling interval and configuration | **FIXED.** Reported in `audit_measurement_boundary.md`, pinned in `tools/codecarbon_config.json`. It was not uniform; see Reviewer 1 comment 9. |
| "energy consumption" vs "sustainability" | **MANUSCRIPT.** Terminology restricted to operational energy; sustainability claims removed. |
| Figures 1 and 2 aggregate too much | **FIXED.** `fig_panels_training.png` and `fig_panels_inference.png` give one panel per model × dataset with the within-run distribution shown. |
| Shorten the Efficiency Index section | **FIXED in analysis.** With the α sweep showing the ranking is insensitive to α, we recommend dropping the composite for the Pareto set. |
| Proofreading, "indipendent", repetition across sections | **MANUSCRIPT.** |

---

## Final state of the seven in-scope ecosystems

All seven now run on the accelerator under one shared measurement contract
(`tools/deepgreen_tracker.py`) and write per-epoch accuracy. Verified end to end
on an RTX 3090:

| ecosystem | GPU | shared contract | shared model | test acc., 1 epoch CIFAR-100 |
|---|---|---|---|---|
| Python/PyTorch | yes | yes | torchvision | 19.84% |
| Python/TensorFlow | yes (2.19.1) | yes | Model Garden | — |
| Python/JAX | yes (0.10.2) | yes | flaxmodels | — |
| C++/LibTorch | yes | yes | **shared TorchScript** | 20.01% |
| Rust/tch | yes | yes | **shared TorchScript** | 20.62% |
| R/torch | yes | yes | own (see below) | 20.79% |
| Java/DL4J | yes (JCublas) | yes | own | 17.14% |

Five independent implementations within 3.6 percentage points, and the four over
LibTorch within one. This is the check reviewer 1 comment 6 and reviewer 3 major
comment 3 asked for, and it is what caught two further defects
(TensorFlow evaluating in training mode; Fashion-MNIST's nested class directory).

**Two residual divergences we could not remove**, both properties of the
ecosystems rather than oversights:

* **Java/DL4J is on CUDA 11.6.** DL4J 1.0.0-M2.1 links against the CUDA 11.6
  runtime and is the last release of that API; there is no CUDA 12 backend. On a
  CUDA 12 host the backend fails with `libcudart.so.11.0: cannot open shared
  object file` and ND4J reports only "no backend on your classpath". The pom now
  pulls `org.bytedeco:cuda-platform-redist` so the stack carries its own runtime
  and is reproducible, but the toolchain difference remains — this is reviewer 1
  comment 8, and it is not removable without changing framework.
* **R/torch cannot use the shared model.** In `torch` 0.17.0 a script module's
  `$train()`/`$eval()` raise `unused argument`, and the handle is not reachable
  through the documented fields, so a loaded module cannot be switched between
  phases. Since the shared modules are exported in training mode, this stack
  would evaluate with batch norm on batch statistics. It builds its own model,
  and architecture parity does not hold for it.

## The revised manuscript

`paper/` contains a rewritten manuscript. It is **not** the submitted study with
corrected numbers: the original conclusions do not survive a common measurement
boundary, so presenting them corrected would misrepresent what the data supports.
It is an instrument and implementation audit — what the campaign actually
measured, a catalogue of fourteen defect classes with measured effect sizes, and
the specification and tooling that make the comparison possible. Every number in
it is traceable to `results/revision/` via `paper/README.md`.

It deliberately does not report a corrected ranking of ecosystems, and says so.

## What is still outstanding

These cannot be closed from this repository:

1. **Run the replicated campaign** — 5–10 independent repetitions per
   configuration, with accuracy logged, under the pinned instrument
   configuration. `scripts/run_campaign.py`, then `09_campaign_v2.py`.
2. **Add a GPU-saturating workload** so the study speaks to the regime it
   motivates (Reviewer 1 comments 2 and 15).
3. **Align CUDA, cuDNN and the LibTorch build** across stacks, at minimum within
   the four-stack LibTorch control group (Reviewer 1 comment 8).
3b. ~~**Build and smoke-test the aligned non-Python stacks.**~~ **Done.**
   Toolchains were provisioned without root and every non-Python stack now
   builds from these sources: C++ 13/13 targets (including the six `*_imported`
   ones that did not exist in the submitted package), Java `mvn compile` under
   JDK 21, Rust `cargo check --bins` clean under tch 0.20, R `parse()` plus a
   runtime `jit_load` test. MATLAB is out of scope (see above).
4. **Validate the daemon-based CodeCarbon controller** against in-process
   CodeCarbon and an external meter (Reviewer 1 comment 13).
5. **Rewrite the manuscript** against the corrected numbers: units, run counts,
   carbon, RQ3, dataset scaling, the extrapolation, related work, terminology.

Given finding 2 — that the whole effect reproduces inside a single shared
backend — the honest reframing is a study of **binding, runtime and toolchain
overhead at low GPU utilisation**, with the LibTorch group as the controlled
core and the other four stacks as context. That is a narrower claim than the
submitted one, and it is one this design can actually support.
