# Response to the JSS reviewers (JSSOFTWARE-D-26-00842)

*Deep Green AI: Energy Efficiency of Deep Learning across Language-Framework Ecosystems*

This document records, comment by comment, what was changed. It distinguishes
three outcomes:

* **FIXED** — resolved in this repository; the artefact is named.
* **MANUSCRIPT** — the change lands in the paper text, `paper/paper.tex`.
* **REQUIRES RE-EXECUTION** — needed a new campaign.

**Status: the replicated campaign has been executed.** Every item previously
marked *REQUIRES RE-EXECUTION* is now closed with measured numbers, with one
exception stated below (the workload still does not saturate the accelerator —
Reviewer 1 comments 2 and 15). The design
is 7 ecosystems × 2 architectures × 3 datasets × 5 independent interleaved
repetitions = 210 runs, each block instrumented twice (NVML + RAPL hardware
counters, and CodeCarbon over the identical window). Raw records are under
`results/campaign_v2/`; aggregates under `results/revision/tables/`.

Everything below is reproducible with `results/analysis/run_all.sh`.

### What the re-execution changed in the conclusions

| Submitted claim | After re-execution |
|---|---|
| Training spread 4.6× (Rust best, Java worst) | 7.8×–19.1× depending on the block, at a common chip-level boundary |
| Energy reported in Joules | The submitted figures were kilowatt-hours; a factor of 3.6×10⁶ |
| "Faster is not greener" | Faster *is* greener here (ρ ≈ 1). The contrary result reproduces as an artefact of the estimator's duration floor |
| Rankings are phase-dependent | Largely phase-consistent once inference is measured rather than estimated |
| 30 epochs as repeated measurements | 5 independent runs per configuration; median between-run CV 0.46 % |
| Accuracy not recorded | Recorded per epoch by every stack; convergence within 0.7 pp on Fashion-MNIST |
| CodeCarbon as sole instrument | Dual instrument; the two agree on energy to 0.5 % over 11,423 blocks |

### The finding that justifies the reviewers' insistence on repetitions

Reviewer 3's major comment 2 asked for independent run-level repetitions.
Executing them produced a result neither we nor the reviewers anticipated, and
it is the clearest possible vindication of the request.

**VGG-16 does not always train.** In 12 of 90 VGG-16 runs the network converges
to *exactly* chance accuracy — 1.00 % on CIFAR-100 (100 classes), 0.50 % on
Tiny ImageNet (200 classes) — and stays there for all 30 epochs, having consumed
the full energy budget while learning nothing. ResNet-18 never does this
(0 of 95). Neither does it on Fashion-MNIST. VGG-16 as shipped has no batch
normalisation, and a plain 16-layer network under Adam at 1e-4 over many classes
can settle into predicting one class; the initialisation decides whether it
does.

It appears in four ecosystems independently (Java 5/10, TensorFlow 4/10,
C++ 2/10, PyTorch 1/10) and never in two (JAX 0/10, R 0/10). That spread is
*not* statistically distinguishable from a common rate at this sample size
(permutation test, p = 0.10), so we report susceptibility as a property of the
recipe and explicitly decline to rank ecosystems by robustness — a chi-square
test on the same table gives p = 0.017, which is exactly the kind of
small-count over-confidence the reviewers were right to guard against.

Two consequences:

1. **It is the strongest evidence that the specification worked.** On VGG-16 /
   CIFAR-100 the cross-ecosystem accuracy spread is 20.2 percentage points over
   all runs and **1.3 points** over the runs that trained. Almost the entire
   apparent disagreement between stacks was collapsed runs, not different
   computations.
2. **It silently breaks fixed-budget energy comparison.** 11.7 % of the
   campaign's energy — 5.1 MJ — was spent on runs that learned nothing. An
   ecosystem that draws unlucky initialisations looks equally expensive and much
   less accurate. With one run per configuration, the published number is
   whichever outcome that seed produced, and nothing in the energy data
   distinguishes the two cases.

**A failed recipe and a broken pipeline are indistinguishable in an energy
table**, and separating them turned up a defect of our own. Both give chance
accuracy at full energy cost. The per-epoch traces separate them completely: in
a genuine collapse the training loss does not move either (all 12 change by
≤ 0.0 %, since a network stuck on one class has nothing left to fit), whereas a
pipeline defect shows training loss falling normally while test loss *rises*.

Four further runs met the chance-accuracy criterion with training loss falling
by 86–87 %. Counted as collapses they would have read as "VGG-16 sometimes fails
to train". They are a defect, and it is ours.

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

Over every block, CodeCarbon's energy agrees with hardware counters to 0.5 %.
Accuracy is not the problem. But the `duration` column it writes beside that
energy is **not the interval the energy was accumulated over**: it is
`max(phase_duration, ≈3.99 s)`, a relation that accounts for R² = 0.993 of its
variance. Above about four seconds the two windows coincide to 16 ms; below it,
the reported duration is a floor. The shortest phase in our campaign takes
0.12 s and is filed as 3.38 s.

Consequently any power or energy-per-second quantity derived by dividing
CodeCarbon's energy by CodeCarbon's duration is understated by up to 6.8× for
sub-second phases and is exact above ten seconds. The bias is a monotone
function of phase length, so it falls hardest on the fastest ecosystems and on
the inference phase — which is exactly where a cross-ecosystem comparison lives.
This is, as far as we can determine, the mechanism behind the submitted
"faster is not greener" finding.

Reproduce with `results/analysis/11_instrument_comparison.py`.

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

**MANUSCRIPT; the workload limitation persists and is now quantified.**
Accepted as a limitation and addressed in the protocol. `results/analysis/repetition_protocol.md`
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

**CLOSED.** The replicated campaign makes the run the unit of analysis.
`results/analysis/14_v2_statistics.py` computes Kruskal-Wallis with ε² across
ecosystems within each block, pairwise Mann-Whitney with Holm correction, and
Cliff's delta, all on **run totals** — five independent numbers per
configuration rather than 30 correlated epochs of one. Between-run dispersion is
now measurable rather than assumed: the median coefficient of variation of
training energy is 0.46 % and the worst is 1.29 %, so the effects reported are
comfortably resolved by the design.

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
to within 0.7 percentage points on Fashion-MNIST (91.0–91.6 %), which is the
strongest available evidence that they are now running the same experiment. On
CIFAR-100 they separate by 13.1 points, and there the energy comparison has to
be read alongside the accuracy reached rather than instead of it — a stack that
converges more slowly looks efficient precisely because it accomplished less.

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
it. On counter-bracketed durations, energy and time now correlate at ρ ≈ 0.98,
so a composite of normalised energy and normalised time averages two quantities
that are very nearly the same measurement. Whatever such an index ranks, it is
not two dimensions.

### Comment 11 — Industrial extrapolation (also Reviewer 3, major comment 6)

**FIXED, by taking the ratio at the same boundary as the thing it multiplies.**
The reviewer is exactly right: the submitted version multiplied an assumed
*per-GPU board power* by a ratio derived from CPU + GPU + RAM energy. Those are
different boundaries, and the product is not a quantity.

Rather than delete the section, we made it dimensionally honest. The scenario
assumes a per-accelerator power budget, so the multiplier is now computed from
**accelerator energy alone** — the NVML counter, CPU package term excluded.
The boundary choice is not cosmetic: it moves the ecosystem spread from 20.4× to
18.4×.

The section is retained with the arithmetic stated as arithmetic, and with two
caveats that come from our own data rather than from convention: the multipliers
were measured on a single-accelerator desktop at 32×32, a regime where data
movement and dispatch weigh more than they would at production scale; and the
whole calculation is a scaling of *relative* differences, which is only as sound
as the boundary they were measured at — the point the reviewer was making.

### Related: measurement coverage, and a wrong turn we took

Asking what boundary a number belongs to led us to check what fraction of each
run lies inside a measured block at all. It is not uniform: tracked time is 44%
of wall time for JAX and 99.8% for R. More than half of a JAX run happens
outside any measured block.

Our first reading was that the low-coverage stacks do a lot of work between
phases and are flattered by a per-phase comparison, and we built a bounded
sensitivity analysis around charging that time back. **That reading was wrong.**
The gap preceding each block is almost exactly the amount by which CodeCarbon's
reported window exceeds the counter-bracketed phase — the two correlate at
r = 0.98 over all 11,623 blocks, their medians differ by 0.23 s, and the window
excess accounts for 96% of all untracked time. Coverage tracks nothing about the
ecosystems except the median length of their blocks: JAX has 3.2-second blocks
and 44% coverage, R has 37-second blocks and 99.8%.

The untracked time is the instrument holding its window open. It would not exist
in an uninstrumented run, and charging it to the ecosystems would charge them for
the cost of being measured. Per-phase energy is the right quantity and the
spreads stand as reported.

What it does bear on is anyone planning a campaign of this shape: under
whole-machine tracking that overhead is real energy drawn from the wall and
attributed to nobody, and its share grows as blocks get shorter — exactly where
the instrument is already least reliable.
`results/analysis/16_coverage_sensitivity.py`.

### A second correction, to our own model of the instrument

We first fitted CodeCarbon's reported duration as `max(phase, 3.99 s)` and
reported R² = 0.993. That model is wrong in the middle of the range. The excess
is bimodal — either about 3.28 s or about 13 ms, with nothing in between and the
transition near 11 seconds — so the reported duration is *the phase plus a
constant* below the threshold, not a floor. The two-regime model gives
R² = 0.998 at a mean absolute error of 0.42 s against 1.14 s for the floor.

We report this because the failure mode is the paper's own argument turned on
us: R² = 0.993 on a heavily skewed predictor looked conclusive while being
carried entirely by the blocks where both models agree. The consequence for
power is unchanged.

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

**FIXED (analysis); the workload limitation remains open.** Confirmed and quantified.
Mean GPU power derived from the energy counter is 83–197 W against a 350 W board
limit — 24–56% — and never approaches the limit in any configuration. GPU energy
is 30–69% of the measured total; the remainder is host-side.
`fig_gpu_load.png` and `audit_gpu_load.md` report this per ecosystem and phase.

**Still open, and stated as such in the manuscript.** The replicated campaign
runs the same 32×32 workload, so it inherits the limitation: it measures data
loading and dispatch more than deep learning. We now bound the consequence in a
known direction rather than leaving it implicit — a saturating workload would
*compress* the ecosystem spread, because the spread we measure is dominated by
host-side work. Adding a saturating configuration is the single most useful
follow-up and is listed first in Future Work.

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
is the apparatus? Median between-run CV is 0.46 % for training and 1.10 % for
inference. That is low — and worth stating plainly, because low run-to-run
variance is what made a single-run design look adequate in the first place. It
is a property of the measurement, not evidence that the measurement is of the
right thing.

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

**The replicated campaign identifies the mechanism.** The submitted RQ3 used
CodeCarbon's `duration` column as the time axis. That column is not the interval
the energy was accumulated over: it is `max(phase_duration, ≈3.99 s)`,
R² = 0.993. Every phase shorter than four seconds — which is every inference
phase in the faster stacks, and the training epochs of the fastest — was
therefore recorded with a *stretched* time and a *correct* energy. That is
precisely the shape of a "fast but energy-hungry" data point, and it is
manufactured entirely by the instrument.

Recomputed on counter-bracketed durations, energy and time rank the
configurations at Spearman ρ ≈ 1 in both phases. `11_instrument_comparison.py`
reproduces the floor characterisation; `14_v2_statistics.py` reproduces the
correlations.

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

Figures are regenerated by `results/analysis/13_paper_figures.py`. The four
submitted figures are replaced by five: the instrument's duration floor and its
effect on power, energy per ecosystem with genuine between-run intervals, energy
against accuracy reached, the two instruments against each other, and
between-run repeatability.

## What is still outstanding

1. **A GPU-saturating workload.** At 32×32 the accelerator runs well below its
   board power limit, so the campaign measures data loading and dispatch more
   than deep learning (Reviewer 1 comments 2 and 15). This bounds the absolute
   figures in a known direction — a saturating workload would compress the
   spread — and it is stated as such in the manuscript rather than left implicit.
2. **A wall-meter reference.** We report a chip-level boundary (NVML + RAPL) and
   decline to extrapolate to whole-system energy. Relating the two needs
   hardware we do not have.
3. **Two residual toolchain divergences**, both properties of the ecosystems
   rather than of the design: DL4J 1.0.0-M2.1 is the last release of its API and
   links CUDA 11.6, so Java cannot join the CUDA 12 group; and R's `torch`
   cannot switch a script module between train and eval mode, so R alone cannot
   load the shared TorchScript module of specification S1.
4. **MATLAB** remains out of scope. It cannot be brought into conformance with
   the specification, and including a stack that provably runs a different
   experiment would reintroduce the confound the design exists to remove.
