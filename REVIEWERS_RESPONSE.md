# Response to the JSS reviewers (JSSOFTWARE-D-26-00842)

*Deep Green AI: Energy Efficiency of Deep Learning across Language-Framework Ecosystems*

This document records, comment by comment, what was changed. It distinguishes
three outcomes:

* **FIXED** — resolved in this repository; the artefact is named.
* **MANUSCRIPT** — the change lands in the paper text, `paper/paper.tex`.
* **REQUIRES RE-EXECUTION** — needed a new campaign.

**Status: the re-execution is complete — 210 runs of 210 — and every quantity in
this document comes from it.**

An earlier replicated campaign also completed in full, and an earlier draft of
this document reported its numbers and stopped there. Auditing it afterwards
showed that three of the study's headline claims were confounded by the campaign
itself, not by the ecosystems it compares:

* **RQ1's mechanism.** TF32 was on for some stacks and off for others, so part
  of what was reported as an ecosystem effect was a numerical-precision policy.
* **Every VGG-16 comparison.** The seven stacks were training four different
  networks, spanning 9.1× in parameter count, under a specification that claimed
  the counts had been checked. They had not been checked anywhere.
* **The collapse attribution.** The per-ecosystem collapse rates reflect
  differing weight initialisers, not the ecosystems.

Reporting numbers from that campaign would have meant reporting those confounds.
So the stacks were aligned — one exported set of weights, one initialiser, one
precision policy, one data pipeline, all of it enforced by 92 conformance
checks and proved by architecture and data parity fingerprints — and the
campaign was re-executed, from 23:09 on 30 August 2026 to 07:47 on 2 September.
Four JAX/VGG-16 runs that predate a Flax dropout-PRNG fix were replayed on
4 September, so all 210 runs come from one source revision. `REVISION_LOG.md`
records the alignment and the re-execution in full.

The design is unchanged: 7 ecosystems × 2 architectures × 3 datasets × 5
independent interleaved repetitions = 210 runs, each block instrumented twice
(NVML + RAPL hardware counters, and CodeCarbon over the identical window). Raw
records are under `results/campaign_v2/`; aggregates under
`results/revision/tables/`; the consolidated replication package — 12,600
counter rows over 210 runs, from 13,441 files and 54 MB of per-block CSVs — under
`results/replication/`, which `consolidate_raw.py` now refuses to build from a
partial campaign.

**Every quantity in this document is re-derived from that campaign.** It comes
from `paper/generated/numbers.tex` — 327 generated macros — and from the tables
under `results/revision/tables/`, and nothing here is a number an author typed
that the pipeline could have produced. `scripts/check_consistency.py` reports
**92 pass, 0 fail**:

```
$ .venv-deepgreen/bin/python scripts/check_consistency.py | tail -3
  92 pass, 0 fail, 0 warn, 0 skip
```

Where a figure belongs to the superseded campaign, or to the source audit of the
submitted one, it is labelled as such at the point it appears. Every item
previously marked **REQUIRES RE-EXECUTION** is now closed by measurement, with
one exception, which is declared rather than answered: the workload does not
saturate the accelerator (Reviewer 1, comments 2 and 15). Re-execution is not
what would close that one.

Everything below is reproducible with `results/analysis/run_all.sh`.

### A correction we owe the reviewers before anything else

The first version of this response reported energy tables that came from
CodeCarbon's *total* — its modelled RAM term included — under captions saying
hardware counters. That is the defect this work catalogues as "modelled term
inside a measured total", and it is the same class of error as the
kilowatt-hours-labelled-as-Joules that the reviewers caught in the submitted
paper. It is fixed at the source: `09_campaign_v2.py` now reads `counters.csv`
and reports the counter quantity over the counter-bracketed window. Every
number below and in the manuscript comes from that. The spread figures that fix
produced — 19.1× moving to 18.2× — belonged to the superseded campaign and are
withdrawn with it: on the re-executed campaign the training spread is
**7.4×–9.8×** depending on the block (`tab_spread.tex`).

Two further self-corrections are recorded in this document rather than quietly
absorbed: the model of the estimator's reported duration (twice wrong, see
below) and the composition of the shared-backend control group, which contained
a stack sharing neither the pinned build nor the exported module.

**Eight more corrections, to claims this document itself made.** They are
collected here rather than scattered as apologies through the comment replies.
Each reviewer comment below is left exactly as the reviewer wrote it; only our
answers have changed.

1. **"JAX is cheapest at inference" was wrong, and the reason was ours.**
   `block_until_ready` appeared nowhere in the repository, so JAX's evaluation
   returned device arrays and the first force happened outside the tracked
   block: 35.9 % of the work finished after the block closed. Synchronised, JAX
   is not cheapest at inference in any block of the campaign — **C++/LibTorch is
   cheapest in all six** (`v2_instrument_ranking.csv`).
2. **"12 of 105 VGG-16 runs collapse" is a first-campaign fact with a
   first-campaign cause.** It is reported below as one. Under one exported set
   of weights the phenomenon does not recur: **0 of 105** VGG-16 runs, 0 of 210
   overall (`v2_convergence_by_model.csv`).
3. **"One failing check" was a fact about a campaign this document does not
   describe.** `check_consistency.py`'s three metrics checks read
   `results/replication/metrics.csv.gz` — an output of the analysis pipeline
   that nothing rebuilt — so the Java/DL4J `test_loss` failure they disclosed
   came from the first campaign, where that column is NaN in all 900 rows, while
   the campaign reported here has no NaN in it at all. The checks read the
   campaign now, through the same completeness gate the tables are built behind:
   **92 pass, 0 fail**.
4. **The between-window calibration figures are withdrawn and the measurement
   redone.** The calibration on disk had been produced on 29 August by the
   harness that pinned TF32 off for Python/PyTorch, and it was carried forward
   against a campaign in which every run records `DEEPGREEN_TF32=1`. The join
   still matched and the table was still written: training energy **201 % apart
   at 293 standard deviations** of the within-window spread, under a manuscript
   sentence calling the drift "below the noise the design already carries". That
   is a precision policy wearing a between-window-drift label.
   `17_window_calibration.py` refuses the comparison now when the two windows do
   not record the same precision policy, or when the calibration predates the
   campaign; the calibration was re-executed under the
   current harness on 4 September (`results/calibration/`, five runs; the
   superseded one is kept as `results/calibration_first_harness/`); and the drift
   it measures is **−1.25 % at 1.62 σ** on training and −3.23 % at 3.95 σ on
   inference (`v2_window_calibration.csv`).
5. **The 4.81× TF32 figure had no script behind it.** The four-cell table in
   `REVISION_LOG.md` §1 has no CSV, script or commit anywhere in this
   repository's history, and its per-step figures are roughly 50× the campaign's
   own measurement of the same two configurations — almost certainly 50-step
   totals mislabelled as per-step. It is withdrawn. The effect is now measured
   twice and scripted both times: as a campaign-level contrast
   (`18_precision_ablation.py` → `v2_tf32_campaign_contrast`), where the one
   cell that is a clean contrast puts Python/PyTorch at **3.14×** with three
   control stacks moving ≤ 1.3 %; and as a kernel probe (`scripts/probe_tf32.py`
   → `v2_tf32_ablation`), six cells, where denying TF32 to cuDNN costs **3.62×**
   the GPU energy and 3.12× the time, and disabling cuDNN outright costs
   **13.08× the energy and 16.30× the time**. The second figure replaces the
   13.3×/16.7× bound quoted for Deeplearning4j's missing cuDNN path, which had
   the same provenance problem.
6. **"Faster *is* greener here (ρ ≈ 1)" is stronger than the data.** Over the 42
   configurations, energy and time rank at ρ = 0.96 in training and 0.92 in
   inference, with 7.1 % and 11.5 % of pairs discordant. Inside a block, where
   the workload is fixed and only the ecosystem varies, ρ runs from **0.68 to
   1.00** over the twelve cells (`v2_stats_cell_rho.csv`). Time is a good proxy
   for energy in this workload; it is not the same measurement.
7. **"Largely phase-consistent" is withdrawn.** The cheapest stack at training is
   not the cheapest at inference in **3 of 6** blocks, and the two orderings are
   identical in **none** of them; the phase correlation runs 0.54 to 0.93
   (`v2_stats_phase_consistency.csv`). The submitted phase-reversal finding does
   not reproduce as an ecosystem property, but neither is the ranking
   phase-invariant, and this document said it was.
8. **"A saturating workload would compress the spread" is a direction we cannot
   name.** The claim was that because the measured spread is dominated by
   host-side work, saturating the accelerator must shrink it. That does not
   follow: it depends on where the spread comes from, which a black-box
   comparison of this shape cannot establish. The manuscript now states the
   limitation without naming a direction, and so does this document.

### What the re-execution changed in the conclusions

| Submitted claim | After re-execution |
|---|---|
| Training spread 4.6× (Rust best, Java worst) | 7.4×–9.8× depending on the block, at a common board-and-package boundary; C++ cheapest on every ResNet-18 block, R and Java the two most expensive |
| Energy reported in Joules | The submitted figures were kilowatt-hours; a factor of 3.6×10⁶ |
| "Faster is not greener" | Faster is *mostly* greener here: ρ = 0.96 training and 0.92 inference across configurations, 0.68–1.00 within a block. The contrary result reproduces as an artefact of the estimator's reported duration |
| Rankings are phase-dependent | Still phase-dependent, but less so: the cheapest stack changes between phases in 3 of 6 blocks and the orderings are identical in none |
| 30 epochs as repeated measurements | 5 independent runs per configuration; median between-run CV 0.49 % training, 1.17 % inference |
| Accuracy not recorded | Recorded per epoch by every stack; convergence within 0.3 pp on Fashion-MNIST (91.3–91.6 %) |
| CodeCarbon as sole instrument | Dual instrument; over 12,600 blocks the two agree to 0.3 % on the terms both read, and disagree by 8.7 % on the total, which is the modelled RAM term |

### The finding that justifies the reviewers' insistence on repetitions

Reviewer 3's major comment 2 asked for independent run-level repetitions.
Executing them produced a result neither we nor the reviewers anticipated, and
it is the clearest possible vindication of the request.

**VGG-16 did not always train, and the reason was the initialiser.** In the
first replicated campaign, 12 of 105 VGG-16 runs converged to *exactly* chance
accuracy — 1.00 % on CIFAR-100 (100 classes), 0.50 % on Tiny ImageNet (200
classes) — and stayed there for all 30 epochs, having consumed the full energy
budget while learning nothing. ResNet-18 never did this (0 of 105), and neither
did VGG-16 on Fashion-MNIST. VGG-16 as shipped has no batch normalisation, and a
plain 16-layer network under Adam at 1e-4 over many classes can settle into
predicting one class.

**In the re-executed campaign it does not happen at all: 0 of 105 VGG-16 runs,
0 of 210** (`v2_convergence_by_model.csv`). That is the measured consequence of
the alignment, and it is why the finding is reported here as a property of the
recipe *and its initialiser* rather than of the ecosystems.

The first campaign's per-ecosystem spread was Java 5/10, TensorFlow 4/10,
C++ 2/10, PyTorch 1/10, and never in three (JAX 0/10, R 0/10, Rust 0/10). That
table is not homogeneous — the exact Freeman–Halton test gives **p = 0.0040**,
which replaces both the permutation p = 0.0676 we first quoted and the
chi-square p = 0.0065 we contrasted it with, and which needs neither an
approximation (the smallest expected frequency is 1.7) nor a seed. But the
inhomogeneity is not evidence about ecosystems. Holding framework, optimiser,
learning rate and data order fixed and varying only the initialiser gives 0 of 6
collapses under He, 2 of 6 under Glorot and 4 of 6 under Xavier
(`results/revision/record/initialiser_trials.csv`), and Deeplearning4j — the one
stack with a hand-rolled initialiser, whose stem weights are 4.6× wider than
torchvision's — carried 5 of the 12. What the table ranks is the four different
weight distributions the stacks shipped. We make no claim about which stacks are
more robust, and the chi-square test on per-ecosystem rates is dropped.

Two consequences, both first-campaign measurements:

1. **It is the strongest evidence that the specification worked.** On VGG-16 /
   CIFAR-100 the cross-ecosystem accuracy spread was 20.2 percentage points over
   all runs and **1.3 points** over the runs that trained. Almost the entire
   apparent disagreement between stacks was collapsed runs, not different
   computations. With every stack now initialised from the same exported
   weights, the worst raw spread anywhere in the re-executed campaign is
   6.5 points, on ResNet-18 / CIFAR-100, and conditioning on convergence changes
   it by nothing.
2. **It silently breaks fixed-budget energy comparison.** 10.7 % of that
   campaign's training energy — 4.8 MJ of 45.0 MJ — was spent on runs that
   learned nothing. An ecosystem that draws unlucky initialisations looks equally
   expensive and much less accurate. With one run per configuration, the
   published number is whichever outcome that seed produced, and nothing in the
   energy data distinguishes the two cases. In the re-executed campaign the
   figure is 0.0 MJ of 36.0 MJ.

**A failed recipe and a broken pipeline are indistinguishable in an energy
table**, and separating them turned up a defect of our own. Both give chance
accuracy at full energy cost. The per-epoch traces separate them completely: in
a genuine collapse the training loss does not move either — all 12 sit at a loss
of ln(N) to within 0.0007, since a network stuck on one class has nothing left
to fit — whereas a pipeline defect shows training loss falling normally while
test loss *rises*.

Four further runs met the chance-accuracy criterion with training loss falling
by 86–87 % and final accuracy up to 14.7 %. Counted as collapses they would have
read as "VGG-16 sometimes fails to train". They are a defect, and it is ours.
The record is preserved as
`results/revision/record/vgg_fashion_pipeline_defect.csv`, because the runs were
deleted and re-executed and the evidence is no longer in the campaign. It also
records the discriminator's one measured miss: five runs carried the defect and
it flagged four, the fifth having finished above the 1.5× chance threshold.

The mechanism is worth stating precisely, because it is not carelessness. In the
submitted package `vgg_fashion.rs` applied a private copy of the input transform
on **both** the training and the evaluation path, and the loader returned raw
uint8 — resizing uint8 and dividing by 255 is correct, so the binary was
internally consistent even while disagreeing with every other ecosystem. This
revision made the loader produce float [0,1] (specification S3) and separately
switched evaluation to batched inference (also S3). **Each change is correct on
its own.** Together they left the private transform on the training path only,
resizing an already-float tensor with a function that returns uint8: every value
truncated to zero. The network trained on black images and was evaluated on real
ones.

Fixed, the four runs re-executed, and a conformance check now forbids
per-binary transforms outright — with a comment stripper, because the first
version of the check failed on the comment explaining why it exists.

Reproduce with `results/analysis/15_convergence.py`.

### The instrument finding

The campaign produced one result about the measurement tool itself that we
believe is new and that bears on any study using it.

Over all 12,600 blocks, CodeCarbon's energy agrees with hardware counters to
0.3 % on the terms both instruments read — because on this platform it reads the
same two registers we do, `nvmlDeviceGetTotalEnergyConsumption` and the RAPL
`energy_uj` files. The accelerator terms differ by a median 1.2 mJ; the whole
residual is a flat 0.48 J in the CPU term, correlated with block duration at
r = 0.02, from nesting our counter reads inside the tracker's window. So this is
not an accuracy result and we no longer present it as one: where the counters
are exposed, the estimator's energy is not a model at all. Its *total* is a
different quantity, 8.7 % above the counters, and that difference is the
modelled RAM term — 7.6 % of what it reports.

The agreement is a property of long blocks. Under one second it degrades to
0.9 % on average and 38 % in the worst block; above 30 seconds it is 0.01 %.

The `duration` column it writes beside that energy is a different matter. It is
**not the interval the energy was accumulated over**: 75 % of blocks carry
seconds of tracker lifetime in which no energy was drawn. The excess is
trimodal — 0.01 s in 25 % of blocks, 4.57 s in 75 %, and 13.38 s in exactly one.
We can now say what it is. It is the cost of `EmissionsTracker.stop()` with
geolocation lookups still outstanding: measured against block length on this
host, closing a 2–8 s block costs 4.56 s with two lookups outstanding, a 9–11 s
block costs 3.01 s with one, and a block of 12 s or more costs 0.012 s with
none. Block length predicts which mode a block lands in; it does not determine
it. 154 blocks longer than the threshold are padded and 3 shorter than it are
not, R is unpadded in all 1800 of its blocks while having the longest ones in
the campaign, C++ is padded in all 1800 of its own, and the wide mode itself
splits by hour of day — a median 4.58 s between 21:00 and 09:00 against 3.29 s
between 09:00 and 21:00, which is the network the lookup goes over.

Consequently any power or energy-per-second quantity derived by dividing
CodeCarbon's energy by CodeCarbon's duration is understated by up to **13.0×**
on blocks under half a second — 20 W reported against 216 W measured — and is
correct above ten seconds. The bias falls hardest on the fastest ecosystems and
on the inference phase, which is exactly where a cross-ecosystem comparison
lives. This is, as far as we can determine, the mechanism behind the submitted
"faster is not greener" finding.

Reproduce with `results/analysis/11_instrument_comparison.py`.

---

## Summary of what the source audit of the submitted campaign found

Five findings go beyond the reviewers' comments and change how the submitted
results should be read. **Every figure in this section is from the submitted
campaign** — eight ecosystems, 48 configurations, one run each — because that is
what was audited. They are the reason the study was rebuilt, not results of the
rebuilt study.

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
this design, and the audit gives a concrete reason: in the submitted campaign the
workload ran the GPU at 24–56% of its board limit, and the re-executed campaign,
sampled at 1 Hz, puts mean accelerator utilisation between 4.7% (R/torch on
ResNet-18) and 79.9% (Java/DL4J on VGG-16). What is measured is therefore
substantially host-side overhead (comment 15). The revised framing is that
ecosystem choice matters at the
*binding and runtime* layer, evidenced within a shared backend, and is not
positioned as a first-order lever at the scales where the field's energy problem
lives. "First large-scale empirical study" is withdrawn; the work is described
as a case study throughout, consistent with the abstract.

### Comment 2 — Workload choice

**REQUIRES RE-EXECUTION, and it is the one item re-execution did not close.**
Accepted as a limitation and addressed in the protocol.
`results/analysis/repetition_protocol.md` §5 requires at least one configuration
that actually loads the accelerator — native resolution, larger batch,
representative arithmetic intensity — and requires GPU utilisation to be reported
next to energy so that readers can see which regime a result belongs to. The
32×32 downsampling of Tiny ImageNet, which removes the property that makes it
demanding, is stated explicitly in `dataset_factors.md`.

The reporting half is done: `19_gpu_utilisation.py` reads the 1 Hz record and
puts utilisation beside energy for the 157 of 210 runs the record covers, and the
manuscript quotes it. The workload half is not, and cannot be by re-running the
same configurations. See comment 15.

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

**What the control group says now.** With the group defined as the stacks that
share both the exported module and the pinned build — C++/LibTorch,
Python/PyTorch and Rust/tch, three of the four, since R can load neither — the
spread inside it is **1.1×–1.6×** against a full seven-stack spread of 7.4×–9.8×,
which is 3–21% of the log spread: 21% at most on ResNet-18 and 3% at least on
VGG-16 (`v2_stats_libtorch_control.csv`). The wider LibTorch *family*, R
included, still accounts for 100% of the ResNet-18 spread and 62% of the VGG-16
spread, because R is the most expensive stack on ResNet-18. So the reframing
stands, but it is sharper than the submitted evidence supported: once the module
and the build are genuinely held fixed, most of the remaining spread is *not*
inside the shared-backend group, and what carries it is R's data pipeline and
Java's missing cuDNN path rather than binding overhead as such.

**Original response (submitted campaign):** `03_statistics.py` adds the
shared-backend control the reviewer asks for. The four LibTorch stacks span 4.4×
(as measured) to 9.9× (harmonised) of energy and 11.6× of time, which is 98–100%
of the full eight-stack spread on a log scale. This is direct evidence that the
variation is host-side. The contribution is reframed accordingly: binding,
runtime and toolchain overhead, measured at ecosystem granularity, not language
energy efficiency. Table: `results/revision/tables/stats_libtorch_control.md`.

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

**CLOSED.** The replicated campaign makes the run the unit of analysis.
`results/analysis/14_v2_statistics.py` computes Kruskal-Wallis with ε² across
ecosystems within each block, pairwise Mann-Whitney with Holm correction, and
Cliff's delta, all on **run totals** — five independent numbers per
configuration rather than 30 correlated epochs of one. Between-run dispersion is
now measurable rather than assumed: the median coefficient of variation of
training energy is 0.49 % and the worst is 1.21 % (inference 1.17 % and 3.04 %),
so the effects reported are far larger than the noise the design carries.

The tests themselves are reported with their limits rather than their headline.
The omnibus is significant in all 12 blocks with ε² ≥ 0.95 and p ≤ 1.2×10⁻⁵. Not
one of the 252 pairwise comparisons survives Holm correction — the smallest
adjusted p is 0.167, against a smallest raw p of 0.0079 — because 21 pairs per
block at five runs each cannot, whatever the separation. 99 % of those pairs are
nevertheless *large* by Cliff's delta. We report the effect sizes and say that
the pairwise tests are underpowered, rather than reporting the omnibus alone and
letting it stand in for them.

*Outliers.* The submitted analysis had no way to tell an outlier from a
measurement artefact. It does now: every block carries two independent readings,
and a value that only one instrument sees is an instrument artefact. The
sampled-power anomalies below were exactly that.

*What the original data supported.* `03_statistics.py` reports Kruskal-Wallis with
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

**CLOSED.** The reviewer is right
that the code computes accuracy — PyTorch, JAX, TensorFlow, MATLAB and Rust all
do — and right that it was never persisted. The Python stacks now write
`metrics.csv` with `train_loss`, `test_loss` and `test_acc` per epoch through
`tools/deepgreen_bench.py`; the same contract is specified for the other stacks
in `repetition_protocol.md` §4, and all seven stacks now honour it.
`09_campaign_v2.py` computes final accuracy, accuracy per kilojoule, and
**energy to reach a target accuracy**, which is the decision-relevant quantity
and does not depend on the arbitrary 30-epoch budget.

The measured result: under the same epoch budget all seven ecosystems converge
to within **0.3 percentage points** on Fashion-MNIST (91.3–91.6 %), which is the
strongest available evidence that they are now running the same experiment. On
CIFAR-100 they separate by 4.0 points (30.7–34.7 %) and on Tiny ImageNet by
1.7 points (14.2–15.9 %), and there the energy comparison has to be read
alongside the accuracy reached rather than instead of it — a stack that converges
more slowly looks efficient precisely because it accomplished less. R/torch is
the top of both of those ranges and the most expensive stack on ResNet-18, which
is the case in point.

One caveat we owe on the derived quantity: accuracy per kilojoule divides the
final test accuracy by the **training** energy of the run, evaluation excluded
(`09_campaign_v2.py`). That is deliberate — the accuracy is what training bought
— but it is not the run's total energy, and the manuscript now says so where the
number appears.

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

**CLOSED, with two documented residuals.** The confound is real, and the
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

Two divergences survive by construction and are reported in the manuscript
rather than hidden: DL4J 1.0.0-M2.1 is the last release of its API and links
CUDA 11.6, so Java cannot join the CUDA 12 group; and R's `torch` cannot switch a
script module between train and eval mode, so R alone cannot load the shared
TorchScript module. The manuscript reports the four-stack LibTorch control group
separately for exactly this reason, so a reader can see how much spread survives
when the backend is genuinely held fixed.

In the original data, "C++/LibTorch is most efficient" remained confounded with
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

**Done in the manuscript.** The composite index and both of its tables are
gone. In their place the manuscript reports an energy-versus-**accuracy**
frontier (Figure "Energy spent against accuracy reached"), which is the
trade-off a practitioner actually faces and which the submitted design could not
draw, because no ecosystem recorded its accuracy.

The replicated campaign strengthens the reviewer's point rather than softening
it. On counter-bracketed durations, energy and time correlate at ρ = 0.96 in
training and 0.92 in inference over the 42 configurations, so a composite of
normalised energy and normalised time averages two quantities that are very
nearly the same measurement. Whatever such an index ranks, it is not two
dimensions. The one place they come apart is inside a block, where the workload
is fixed and only the ecosystem varies: there ρ runs from 0.68 to 1.00 over the
twelve cells, and the lowest is ResNet-18 / CIFAR-100 training, with 5 discordant
pairs of 21. That residual is a real second dimension and it is small; it does
not rescue a composite index, and we report it rather than round it away.

### Comment 11 — Industrial extrapolation (also Reviewer 3, major comment 6)

**FIXED, by taking the ratio at the same boundary as the thing it multiplies.**
The reviewer is exactly right: the submitted version multiplied an assumed
*per-GPU board power* by a ratio derived from CPU + GPU + RAM energy. Those are
different boundaries, and the product is not a quantity.

Rather than delete the section, we made it dimensionally honest. The scenario
assumes a per-accelerator power budget, so the multiplier is now computed from
**accelerator energy alone** — the NVML counter, CPU package term excluded.
The boundary choice is not cosmetic: on the re-executed campaign the ecosystem
spread is 24.6× at the counter total and 22.2× at the accelerator boundary, and
35.9× if the multiplier is taken on time instead. Those are three different
answers to the same scenario, and which one a reader gets depends entirely on
which quantity the per-accelerator budget is multiplied by.

The section is retained with the arithmetic stated as arithmetic, and with two
caveats that come from our own data rather than from convention: the multipliers
were measured on a single-accelerator desktop at 32×32, a regime where data
movement and dispatch weigh more than they would at production scale; and the
whole calculation is a scaling of *relative* differences, which is only as sound
as the boundary they were measured at — the point the reviewer was making.

### Related: measurement coverage, and a wrong turn we took

Asking what boundary a number belongs to led us to check what fraction of each
run lies inside a measured block at all. It is not uniform: tracked time is 46%
of wall time for C++/LibTorch and 100% for R/torch. More than half of a C++ run
happens outside any measured block.

Our first reading was that the low-coverage stacks do a lot of work between
phases and are flattered by a per-phase comparison, and we built a bounded
sensitivity analysis around charging that time back. **That reading was wrong.**
The gap preceding each block is almost exactly the amount by which CodeCarbon's
reported window exceeds the counter-bracketed phase — the two correlate at
r = 0.98 over 12,390 blocks, their medians are 3.51 s and 3.29 s and differ by
0.26 s, and the window excess accounts for 97% of all untracked time; for three
ecosystems it accounts for slightly more than all of it, the estimator's window
overlapping its own gap. Coverage tracks nothing about the ecosystems except the
median length of their blocks: C++ has 3.1-second blocks and 46% coverage, R has
33.5-second blocks and 100%.

The untracked time is the instrument holding its window open. It would not exist
in an uninstrumented run, and charging it to the ecosystems would charge them for
the cost of being measured. Per-phase energy is the right quantity and the
spreads stand as reported.

What it does bear on is anyone planning a campaign of this shape: under
whole-machine tracking that overhead is real energy drawn from the wall and
attributed to nobody, and its share grows as blocks get shorter — exactly where
the instrument is already least reliable. Across this campaign it is 10.5 hours
and 2.4 MJ, 6% of the total.
`results/analysis/16_coverage_sensitivity.py`.

### A second correction, to our own model of the instrument

We fitted CodeCarbon's reported duration twice and got it wrong twice, in the
same way. First as a floor, `max(phase, 3.99 s)`, R² = 0.982 at a mean absolute
error of 1.79 s. Then, on finding that wrong in the middle of the range, as two
regimes — phase plus 4.58 s below 11 s — R² = 0.998 at 0.53 s, less than a third
of the error, which looked conclusive.

It is not. The excess is trimodal (0.01 s, 4.57 s, 13.38 s) and block length does
not determine which mode a block lands in: 154 blocks longer than 11 s are
padded, 3 shorter than it are not, and one ecosystem is unpadded in every one of
its 1800 blocks while another is padded in every one of its own. Both fits were
curves through a predictor that does not govern the phenomenon, and both earned
their R² against a variable whose range is dominated by phase length itself.

We report both failures because they are the paper's own argument turned on us,
twice: a high R² on a skewed predictor is not evidence of the right functional
form. The third attempt is not a fit at all. `probe_reported_window.py` measures
what `EmissionsTracker.stop()` costs against the length of the block it closes,
on this host: 4.56 s with two geolocation lookups outstanding, 3.01 s with one,
0.012 s with none, the count set by how much of the tracker's 8-second API-call
interval the block has already consumed. That is a mechanism, it predicts the
three modes and the 12-second threshold, and it leaves the 154 padded blocks
above the threshold as what they are — network latency, which the diurnal split
of the wide mode (4.58 s at night against 3.29 s by day) shows independently.
The manuscript states the consequence without a fitted model: divide each
instrument's energy by its own duration, per phase-length bin, which is what it
should have done first.

### Comment 12 — Descriptive answers, no mechanism

**Partly FIXED.** `04_energy_vs_time.py` performs the decomposition the reviewer
points at: energy = mean power × duration. 98% (inference) and 107% (training)
of the log spread is attributable to duration, and the mean-power spread is only
1.3–1.6×. Combined with the GPU-load audit, the LibTorch control and the implementation
audit, the mechanism is now concrete rather than hypothesised: the stacks differ
in how long a step takes at low GPU utilisation, and in the submitted campaign
the dominant reason was how many CPU threads decode and feed images — Spearman
−0.73 (p = 0.041) between loader threads and epoch duration, with Rust on
96 cores at one end and R on zero workers at the other.

That divergence is gone: every stack now runs the two loader workers the
specification mandates, and the spread survives anyway. The 1 Hz record says why
for the extreme case. R/torch holds 4.5–5.0 % accelerator utilisation on
ResNet-18 and 12.1–13.4 % on VGG-16, with 1,271–3,001 MiB of device memory
resident, and its accelerator power stays between 126 W and 141 W whichever
network it is training, where every other stack's moves by 100 W or more with the
network. The card is attached and idle; R's two workers are simply slower per
image. This is still a data-pipeline story, not a kernel-efficiency story and not
a language story, but it is now measured at the device rather than inferred from
a thread count. Layer-level and function-level profiling remains future work and
is now described as necessary rather than optional.

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

**REQUIRES RE-EXECUTION for the workload; FIXED for the measurement of it.**
Confirmed and quantified. In the submitted campaign, mean GPU power derived from
the energy counter was 83–197 W against a 350 W board limit — 24–56% — and never
approached the limit in any configuration; GPU energy was 30–69% of the measured
total. `fig_gpu_load.png` and `audit_gpu_load.md` report that per ecosystem and
phase.

The re-executed campaign measures the same thing directly rather than deriving
it. Sampled at 1 Hz alongside the runs, mean accelerator utilisation spans
**4.7% to 79.9%** across ecosystem and architecture — R/torch on ResNet-18 at the
bottom, Java/DL4J on VGG-16 at the top — and mean accelerator power spans
127–258 W on a 350 W card (`19_gpu_utilisation.py`,
`v2_gpu_utilisation_by_ecosystem.md`). The record began after the campaign did,
so it covers 157 of the 210 runs, and the table says so rather than averaging
over whatever happened to be there.

**Still open, and stated as such in the manuscript.** The re-executed campaign
runs the same 32×32 workload, so it inherits the limitation: it measures whole
pipelines more than saturated kernels. We previously wrote that this bounds the
consequence in a known direction — that a saturating workload would *compress*
the spread, because the spread we measure is dominated by host-side work. That
does not follow, and we withdraw it: which way saturation moves the ranking
depends on where the spread comes from, and a black-box comparison of this shape
cannot establish that. Adding a saturating configuration is the single most
useful follow-up and is listed first in Future Work.

---

## Reviewer 3

### Major comment 1 — Energy units

**FIXED.** See Reviewer 1 comment 4. The reviewer's inference is exactly right:
8.79 × 10⁻⁴ read as kWh is 3164 J, which matches the derived mean power. The
full pipeline audit the reviewer asks for — native unit, every conversion, units
stored in the replication data — is `01_data_audit.py` and
`audit_units.md`.

### Major comment 2 — No independent run-level repetitions

**CLOSED.** We agreed this was not a future-work point, and it is now done.
Each of the 42 configurations was executed five times as an independent process
with a distinct seed, **interleaved** rather than run back to back so that drift
in machine state is spread across conditions instead of aliasing onto whichever
ecosystem ran last: 210 runs in total. Every interval, test and effect size in
the revised manuscript is computed across those runs.

The design also answers a question the submitted one could not ask: how precise
is the apparatus? Median between-run CV is 0.49 % for training and 1.17 % for
inference, worst case 1.21 % and 3.04 %. That is low — and worth stating plainly,
because low run-to-run variance is what made a single-run design look adequate in
the first place. It is a property of the measurement, not evidence that the
measurement is of the right thing.

Four of the 210 runs were not part of the interleaved schedule: the JAX/VGG-16
runs that predate the Flax dropout fix, replayed two days after the campaign
finished. The between-window drift this exposes them to is measured rather than
assumed — one configuration re-executed in a third window under the current
harness gives **−1.25 % at 1.62 standard deviations** of the within-window spread
on training and −3.23 % at 3.95 σ on inference (`v2_window_calibration.csv`).

### Major comment 3 — No accuracy or convergence

**Accepted; FIXED in code, and it immediately caught two defects.** Persisting
accuracy is not only needed to normalise energy by useful work -- it is the only
thing that would have revealed that the TensorFlow stack was evaluating with
`training=True` pinned into its functional graph (21% test accuracy against 83%
on train, 86% once removed), and that the Fashion-MNIST class `T-shirt/top` had
produced a nested directory that different loaders read differently. Neither is
visible in energy or runtime.

With the quality metric in place, the three Python ecosystems agreed to within
one percentage point after one epoch on Fashion-MNIST (PyTorch 87.11%, JAX
86.46%, TensorFlow 86.06%) in the pre-campaign smoke test. The campaign itself
is the stronger statement: over 30 epochs and five repetitions, all seven
ecosystems land within 0.3 percentage points of each other on Fashion-MNIST.

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

**The replicated campaign identifies the mechanism.** The submitted RQ3 used
CodeCarbon's `duration` column as the time axis. That column is not the interval
the energy was accumulated over: 75 % of blocks carry seconds of tracker
lifetime in which no energy was drawn. Every such block — which is every
inference phase in the faster stacks, and many of the training epochs — was
therefore recorded with a *stretched* time and a *correct* energy. That is
precisely the shape of a "fast but energy-hungry" data point, and it is
manufactured entirely by the instrument.

Recomputed on counter-bracketed durations, energy and time rank the
configurations at Spearman ρ = 0.96 in training and 0.92 in inference — not the
ρ ≈ 1 an earlier draft of this document claimed, and not the divergence the
manuscript claimed either. `11_instrument_comparison.py` reproduces the window
characterisation; `14_v2_statistics.py` reproduces the correlations.

Efficiency Index: see Reviewer 1 comment 10.

### Major comment 5 — Figure 4 cannot show growth across datasets

**FIXED.** Correct. `06_dataset_scaling.py` reports absolute energy per epoch and
a single common reference cell instead of per-column normalisation
(`fig_dataset_absolute.png`, `dataset_absolute_energy_*.md`).

The claim also weakens under the correct scale. In the submitted campaign,
training energy grows 1.79× from Fashion-MNIST to Tiny ImageNet, but per training
image it is 114.8, 109.3 and 123.1 mJ respectively — essentially flat. The effect
is dataset *size*, not
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

The carbon figure changes accordingly: the submitted campaign's tracked blocks
total **4.02 kWh** and **1.33 kg CO₂eq** at 331 gCO₂eq/kWh, not 3.3 kg. This is a
lower bound — it excludes compilation, dataset preparation, idle time and
discarded runs — and is now stated as such (`campaign_carbon_footprint.md`). The
campaign this document reports is larger and is stated on the same basis: 210
runs, 12,600 blocks, **40.4 MJ = 11.2 kWh** inside the measured windows, of which
36.0 MJ is training.

### Major comment 8 — Language versus ecosystem

**FIXED.** One unit throughout: **language-framework ecosystem**, written as
`Python/PyTorch`, `C++/LibTorch`, `Java/DL4J`, `R/torch`, `Rust/tch`,
`MATLAB/DLT`, `Python/TensorFlow`, `Python/JAX`. The mapping is in
`common.py::ECOSYSTEM` and is applied to every table, figure axis and caption; no
figure axis is labelled "Language" any more. Statements are phrased as
"Rust/tch versus Java/DL4J". The submitted campaign's count was eight ecosystems
over six languages; the campaign reported here is **seven ecosystems over five
host languages** (Python, C++, Java, R and Rust), MATLAB being out of scope, and
that count is stated consistently and generated from the data.

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
The campaign supersedes it as evidence: over the full 30 epochs and five
repetitions, the seven agree to 0.3 points on Fashion-MNIST and 4.0 on
CIFAR-100.

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

`paper/paper.tex` is the revised manuscript, in the same Elsevier `cas-dc`
format as the submission. It is a substantial rewrite rather than a corrected
version of the original: the measurement apparatus is now part of the
contribution, RQ4 is added, and every result section is rebuilt on the
replicated campaign.

Every quantity the manuscript quotes is **generated**, not transcribed.
`results/analysis/12_paper_numbers.py` writes `paper/generated/numbers.tex` (a
`\newcommand` per value) and the result tables; `paper.tex` reads them at build
time. This is a direct response to how the unit error occurred: a number that no
author types is a number no author can mislabel.

It builds, at 36 pages in the `cas-dc` two-column format.

Figures are regenerated by `results/analysis/13_paper_figures.py`. The four
submitted figures are replaced by seven: the instrument's duration floor and its
effect on power, energy per ecosystem with genuine between-run intervals, energy
against accuracy reached, the two instruments against each other, between-run
repeatability, the convergence traces of this campaign, and — generated by the
same script under `--campaign v1` — the first campaign's VGG-16 collapses against
ResNet-18, which is the only figure in the paper drawn from the superseded
records and is labelled as such.

### The apparatus now polices the manuscript as well as the experiment

Two numbers in the text had drifted from what the code did, in exactly the way
the unit error did. The manuscript quoted 57 conformance checks while the
checker ran 63, and claimed five catalogue entries were the authors' own while
four carried the mark. Both are now counted by
`results/analysis/12_paper_numbers.py` from the checker and from the table
itself, so neither can drift again: the checker runs **92 checks, 92 passing,
0 failing**, and the defect catalogue holds **34 entries, 12 of them ours**. The
highlights uploaded to the submission form are expanded from the manuscript's own
`highlights` environment by `scripts/emit_highlights.py` for the same reason.

Counting them was not enough on its own, and this is the second correction of
this kind we owe. `12_paper_numbers.py` counted a *skipped* check as a passing
one, so a run in which three checks could not read their input would have
published 92 of 92 passing; it now refuses to emit the conformance macros at all
unless every check actually ran. It also refuses to emit any macro whose value
formats as `nan`, after a chi-square that is undefined on a table of zeros
reached the manuscript as "p = nan".

The raw records — 13,441 files and 54 MB of per-block CSVs — are consolidated by
`scripts/consolidate_raw.py` into four gzipped tables under
`results/replication/`, 2.3 MB, carrying the identical records with the run
identity on every row: 12,600 counter rows, 12,600 estimator rows, 6,300 epoch
metric rows and 210 manifests. `--check` verifies the package against the raw
tree, and the script now refuses to build a package at all unless the campaign is
complete and every run directory in it passed the completeness gate. It runs at
the *end* of `run_all.sh`, because it is an output of the pipeline; it used to be
read as an input, which is the defect corrected above.

## What is still outstanding

1. **A GPU-saturating workload.** At 32×32 the accelerator runs between 4.7% and
   79.9% utilisation depending on stack and network, so the campaign measures
   whole pipelines more than saturated kernels (Reviewer 1 comments 2 and 15).
   This is the one item the re-execution did not close, and it is the one item we
   cannot bound in a known direction: whether saturation compresses the ecosystem
   spread depends on where that spread comes from, which this design cannot
   establish. The manuscript states it as a limitation on external validity
   without naming a direction.
2. **A wall-meter reference.** We report a chip-level boundary (NVML + RAPL) and
   decline to extrapolate to whole-system energy. Relating the two needs
   hardware we do not have. The counters are un-baselined — they include the
   machine's static draw within their own boundary — and the idle floor is
   measured rather than assumed: on a quiet host, 60 seconds at a one-minute load
   average of 0.26, the accelerator draws 24.6 W and the CPU package 38.9 W,
   63.5 W together (`v2_idle_baseline.json`, measured 5 September 2026, 10:44
   UTC).
3. **Accelerator utilisation for 53 runs.** The 1 Hz sampler was started after
   the campaign began, so it covers 157 of 210 runs. The tables say which, and no
   utilisation figure is averaged over an unstated subset.
4. **Two residual toolchain divergences**, both properties of the ecosystems
   rather than of the design: DL4J 1.0.0-M2.1 is the last release of its API and
   links CUDA 11.6, so Java cannot join the CUDA 12 group; and R's `torch`
   cannot switch a script module between train and eval mode, so R alone cannot
   load the shared TorchScript module of specification S1.
5. **No harness provenance in the manifests.** All 210 manifests record the
   environment, the CodeCarbon configuration, the precision policy and the
   machine state, but not the revision of the harness that wrote them. Provenance
   for this campaign is established by argument — one source revision, the four
   replayed runs included — where it should be a field in the file.
6. **MATLAB** remains out of scope. It cannot be brought into conformance with
   the specification, and including a stack that provably runs a different
   experiment would reintroduce the confound the design exists to remove.

---

## What the re-execution changed in this document

Listed so that no reader has to work it out from the dates. Each of these was a
quantity or a claim taken from the superseded campaign; each has now been
replaced from the generated macros, and this table records the replacement rather
than announcing one.

| Section | Claim as it stood | What the re-executed campaign gives |
|---|---|---|
| VGG-16 collapse | "12 of 105 VGG-16 runs", the per-ecosystem spread (Java 5/10, TensorFlow 4/10, C++ 2/10, PyTorch 1/10, and never in JAX, R, Rust) | Kept, as a **first-campaign** result with a first-campaign cause: the rates reflect four different initialisers (He 0/6, Glorot 2/6, Xavier 4/6). With one exported set of weights the phenomenon does not recur — **0 of 105** VGG-16 runs, 0 of 210 overall. |
| VGG-16 energy, everywhere it appears | every per-ecosystem VGG-16 figure and ratio | Re-measured on one canonical 15,028,644-parameter network. The VGG-16 spread is 7.4× in every dataset, TensorFlow or JAX cheapest and Java most expensive. |
| RQ1 mechanism | the attribution of the spread to ecosystem behaviour | TF32 is now allowed for all seven, and the effect it was hiding is reported as a declared ablation: 3.14× on the one cleanly comparable cell, three control stacks moving ≤ 1.3 %. |
| JAX inference ranking | JAX as cheapest at inference | Withdrawn. Measured with asynchronous dispatch unsynchronised, 35.9 % of the work finished after the tracker closed. Synchronised, **C++/LibTorch is cheapest at inference in all six blocks** and JAX ranks third to fifth. |
| Loader configuration table | "0 (R/torch), 1 (JAX, TensorFlow, MATLAB), 2 (PyTorch, C++, Java), 96 (Rust)" | Kept as a description of the state that was *found* in the submitted package. All seven stacks now run two loader workers, verified by the checker; MATLAB is out of the study. |
| Homogeneity of collapse | the permutation p-value, p = 0.07, contrasted with chi-square p = 0.0065 | Replaced by the exact Freeman–Halton test: on the first campaign's table **p = 0.0040**, with no approximation and no seed. On this campaign's table there is nothing to test — every cell is zero — and the chi-square macro is not emitted at all rather than typeset as `nan`. |
| Between-window calibration | "training energy 0.21 % apart at 0.61 σ", from a calibration run on 29 August against the first campaign | Withdrawn. Run against the re-executed campaign the same calibration reads 201 % apart at 293 σ, because the two windows differ in precision policy and not only in time. Re-measured under the current harness on 4 September: **−1.25 % at 1.62 σ** on training, −3.23 % at 3.95 σ on inference. |
| Conformance | "one failing check" | The check was reading a superseded replication package. Reading the campaign: **92 pass, 0 fail, 0 skip**. |
| Precision ablation | the 4.81× four-cell table in `REVISION_LOG.md` §1 and the 13.3×/16.7× cuDNN bound in §19 | Both had no script behind them and quoted 50-step totals as per-step figures. Both are re-derived and scripted: campaign contrast **3.14×**, kernel probe **3.62×** for TF32 and **13.08× / 16.30×** for cuDNN disabled. |

The instrument comparison, the padding and window-floor results, the boundary
argument and the source-level defects are properties of the tooling and the code
rather than of the campaign's numbers. Their *magnitudes* have moved with the new
records — the padded share from 67 % to 75 %, the worst power understatement from
11.2× to 13.0×, the two-instrument agreement from 0.5 % to 0.3 % — and the
mechanism behind the window excess is now measured rather than fitted, but none
of the conclusions drawn from them changes.
