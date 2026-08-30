# Revision log — third review round

Running record of what changed in the harness and why, kept as the work happens
so the manuscript rewrite is not reconstructed afterwards from commit messages.
Every entry names the defect, the evidence, and what it costs the claims the
submitted paper makes.

Companion documents: `REVIEWERS_RESPONSE.md` answers the JSS reviewers;
`results/analysis/experiment_spec.md` is the specification;
`scripts/check_consistency.py` enforces it.

---

## Where this round came from

Five reviewers with different backgrounds — measurement systems, statistics,
replication and JSS artefact standards, deep learning practice, and
claim-evidence auditing — read the repository and the manuscript. Four
delivered; the fifth stopped on a rate limit and is still owed.

Three of the paper's headline claims turned out to be confounded, and the
decision was taken to re-execute the whole campaign rather than disclose the
confounds and narrow the paper.

---

## 1. Precision: one flag was worth more than every binding in the study

**Defect.** Specification S4 asked for fp32 with TF32 disabled. One stack of
seven obeyed it. The pin lived in `tools/deepgreen_bench.py`, which only the
three Python stacks import, and of those only PyTorch reached cuDNN through it:
TensorFlow was handed a `mixed_precision` policy, which is a different axis, and
JAX was handed nothing. So Python/PyTorch ran true-fp32 convolutions and the
other six took cuDNN's Ampere default.

**Evidence.** RTX 3090, ResNet-18, batch 128, median of three, separating the
two flags that actually differ (the other two the harness set are already every
binding's default):

| `cudnn.allow_tf32` | `deterministic` | wall | GPU J | |
|---|---|---|---|---|
| True | False | 0.352 s | 78.3 | C++ / Rust / R / Java |
| True | True | 0.360 s | 105.4 | 1.35× |
| False | False | 1.088 s | 376.6 | 4.81× |
| False | True | 1.221 s | 407.6 | 5.21× — Python, as pinned |

Against a campaign PyTorch-vs-C++ training gap of 3.24–3.79× on ResNet-18 and
1.07× on VGG-16. The VGG-16 null identifies the mechanism: `matmul.allow_tf32`
is `False` by default in every binding, so VGG's GEMM-dominated head runs fp32
everywhere and the gap disappears exactly where the workload stops being
convolution. No account of binding overhead predicts that.

**What it costs the paper.** RQ1's mechanism — "whatever separates those three
cannot be the kernels, because the kernels are the same code" — is false as
written, and the quantity it supports ("about half the log spread on ResNet-18
is binding overhead") reproduces from one configuration flag.

**Fix.** `DEEPGREEN_TF32`, one variable, applied through the frameworks for the
Python stacks and through `NVIDIA_TF32_OVERRIDE` for the four processes that
never import the harness. Verified equivalent to within 0.5%. The default is
TF32 *allowed* — every framework's own behaviour on Ampere, and 57 h of machine
time against 138 h for the same 210 runs — which amends S4. `cudnn.deterministic`
is pinned nowhere now; it was Python-only, worth a further 1.35×, and not
expressible in DL4J or R.

**Note for the manuscript.** The 4.81× is not lost: it becomes a declared
precision ablation, which is a more useful finding than a hidden confound.

---

## 2. The networks were not the same networks

**Defect.** VGG-16 ran as four different networks spanning 9.1× in parameters:
134,670,244 (torchvision head, in PyTorch, C++ and Rust), 34,006,948
(Flatten + 4096 + 4096, in TensorFlow and Deeplearning4j), 15,028,644 (512 + 512,
in R) and 14,765,988 (a linear probe on global average pooling, in JAX). S1
claimed parameter counts were checked against the exported module. No such check
existed anywhere in the tree.

ResNet-18 was parity for five stacks of seven. TensorFlow's Model Garden adds a
projection shortcut on the first stage where torchvision uses the identity
(+4,224). Deeplearning4j built the graph by hand and diverged on three axes: a
truncated-normal fan_in initialiser, `ConvolutionMode.Truncate` instead of
explicit padding, and a bias on every convolution that torchvision omits ahead
of BatchNorm.

**What it costs the paper.** Half the study's energy comparisons were comparing
models rather than ecosystems.

**Fix.** One canonical definition per architecture in
`scripts/export_torchscript_models.py`, seeded, with `models/MANIFEST.json`
recording parameter counts and a hash of the weights. Every stack asserts its
own count at startup against `DEEPGREEN_EXPECTED_PARAMS`, carried by the driver
so no stack needs a JSON parser. All seven now agree on both architectures.

**Design decision to record.** The VGG-16 head is the CIFAR head, not
torchvision's. At 32×32 the map reaching torchvision's classifier is 1×1×512 and
`AdaptiveAvgPool2d((7,7))` replicates it — verified,
`allclose(pooled, feat.expand(-1,-1,7,7))` is `True` — so `Linear(25088, 4096)`
is fed 49 copies of 512 values and 102.8M of 134.7M parameters have rank at most
512. Measuring that measures Adam state bandwidth, not VGG-16. Per step: 29.67 ms
and 10.11 J against 14.14 ms and 4.22 J. `HEAD = "imagenet"` takes the other
branch.

**Also worth stating.** "Byte-identical module" was always a stronger claim than
the argument needed. Two exports from one seed produce identical weights and
different bytes — TorchScript's archive is not reproducible even with timestamps
fixed — so the manifest carries a hash of the parameters in name order, which is
the property the study actually requires and which anyone can recompute.

---

## 3. The initialiser, not the ecosystem, drives the collapse

**Defect.** Twelve of 105 VGG-16 runs converged to exactly chance and sat at a
loss of ln(N) for all thirty epochs. The manuscript reported the rate as a
property of the ecosystem, and asserted that initialisation was *not* the
deciding factor because three stacks loading a byte-identical module disagreed
about which repetitions collapsed. That premise was false: PyTorch built
torchvision fresh and loaded no module, and C++ and Rust each loaded a module
from a separate unseeded export.

**Evidence.** Holding framework, optimiser, learning rate and data order fixed
and varying only the initialiser: 0 of 6 collapses under He, 2 of 6 under
Glorot, 4 of 6 under Xavier. Deeplearning4j, the one stack with a hand-rolled
initialiser, carried 5 of the 12.

Deeplearning4j's `WeightInit.RELU` is a truncated normal with sqrt(2 / fan_in);
torchvision uses `kaiming_normal_(fan_out, relu)`, untruncated, sqrt(2 / fan_out).
For ResNet-18's 7×7 stem that is 0.116 against 0.025 — **4.6× wider**. The
comment in the file noted the difference and kept the fan_in version.

**Fix.** `TorchWeightInit` in the Java tree, and `torch_init.py` in the
TensorFlow tree. Both compute fan_out from the weight shape rather than trusting
the framework's, because DL4J divides a convolution's fan_out by the stride
product and torch does not: without that, every stride-2 convolution came out
exactly twice as wide while every stride-1 convolution matched.

**What it costs the paper.** Section 5 needs rewriting. The finding is real and
worth keeping, but it is "VGG-16 without batch normalisation under Adam at 1e-4
is initialisation-critical, and frameworks ship different defaults" — a defect
class about framework defaults, not about ecosystems. The χ² test on
per-ecosystem collapse rates is testing the initialiser and should go.

---

## 4. JAX closed its measurement window before its work finished

**Defect.** `block_until_ready` appeared nowhere in the repository. `eval_step`
returns device arrays and the first force is `float(vl)`, outside the tracked
block.

**Evidence.** On the campaign's own eval loop shape: 35.9% of the work finishes
after the block closes, and the energy attributed to the block is 0.70× the true
figure. JAX is the stack the study reports as cheapest at inference.

**Fix.** `jax.block_until_ready` before each context exits, in both model files.

**Incidental.** The same probe re-confirmed the NVML register's 10 Hz update
rate from the other direction: a 44 ms window reported 0.0 J, because it is
shorter than one tick.

---

## 5. Two stacks were scored on a different test set

**Defect.** `deepgreen_loader.py` batched with `drop_remainder=True` and
TensorFlow and JAX computed steps as `samples // batch_size`, so both trained on
fewer gradient steps per epoch and evaluated on 9,984 of 10,000 test images —
always the same 16, the test loader being unshuffled over a sorted file list.
Accuracy is the denominator of this study's energy-per-unit-quality metric.

**Fix.** `drop_remainder=False` and ceiling division. Confirmed by the same
method that detected the defect: `test_acc × N / 100` is an integer only for
N = 10,000 now, in every stack.

---

## 6. Batch normalisation was configured three different ways

**Defect.** Found by the smoke test, not by reading. torch uses momentum 0.1 and
eps 1e-5; Deeplearning4j's builder used `.decay(0.9)` and `.eps(1e-5)`, which
matches; Keras defaults to momentum 0.99 and eps 1e-3, and nothing overrode it.

**Evidence.** One epoch of Fashion-MNIST, ResNet-18: TensorFlow scored 77.4%
against 86–88% for every other stack, with train accuracy identical (83.0%) and
test loss 0.77 against 0.35–0.37. Aligning momentum and epsilon moved it to
86.57% and the test loss to 0.366.

**What it costs the paper.** Nine points of accuracy on one stack, converging
only partly over thirty epochs, in the denominator of the efficiency metric and
in the energy-to-target table.

---

## 7. Seeds that no stack used

**Defect.** The S6 check reported "5 of 5 distinct seeds" having asked the
campaign *planner*, which never sees whether a stack uses the seed it is handed.
Java hardcoded `RNG_SEED = 123` for its data order, so all five repetitions
shuffled identically. JAX never passed the seed to its loader, so `tf.data`
shuffled with `seed=None`. Two of six Rust binaries rebuilt the generator inside
the epoch loop, replaying one permutation thirty times.

**Fix.** All three threaded from the run contract; checks now read the source
rather than the planner.

---

## 8. R trained on [0, 1/255]

**Defect.** `R/models/resnet18.r` divided by 255 an array
`image_folder_dataset`'s loader already returns in [0, 1] — measured:
`base_loader` on a campaign PNG gives range 0 to 1. `R/models/vgg16.r` used
`transform_to_tensor` and was correct. One ecosystem, two input pipelines, under
an S3 that fixes the pipeline for all stacks and a check that covered the input
transform for Rust alone. Largely absorbed by the BatchNorm after conv1, which
is why it never showed in the accuracies.

---

## 9. The instrument's own defects

- **RAPL multi-wrap guard compared microjoules to watts.** `65532610987 / 35.0`
  is 1.87e9 seconds, so the threshold was 59 years rather than 31 minutes and the
  guard was inert for every realisable block — with `J_PER_UJ` defined three
  lines above it and unused. It was also consulted only when the delta came back
  negative, so two wraps landing above the opening reading were never checked.
  No published number moves: the longest block in the campaign is 101 s.
- **Manifests recorded no mutable machine state.** No power limit, clocks,
  persistence mode, ECC, governor, boost, driver version or timestamp, for any
  run. Now recorded for all seven stacks. Persistence mode is **Disabled** on
  this host, which is the most plausible mechanism for the bimodal accelerator
  power state two reviewers found independently.
- **`DEEPGREEN_LIBTORCH` pointed at a scratch directory under `/tmp`.** It has
  since been cleaned up, and the Rust binaries built against it no longer resolve
  `libtorch_cuda.so`. Now defaults to the 2.7.0+cu128 the C++ build fetches into
  the repository.
- **Counter files are opened in append mode and the driver never cleared a run
  directory**, so re-running a job over its own output silently doubled the rows.
  Found by re-running one smoke test and noticing Java had four blocks where
  every other stack had two. The driver refuses now unless `--force`.

---

## 10. The checker could not fail

- **`check_regex` was existential where the specification is universal.** It
  passed when *one* file in a glob matched, so the Java learning-rate check
  passed while four siblings carried 1e-3 and 1e-5, citing as evidence a file in
  `expt/imported/` that the campaign does not execute. A reviewer proved it by
  changing only the executed classes to `lrAdam = 1e-2` and `batchSize = 999`;
  the checker printed `[ok]`. Universal now, and it fails on that sabotage.
- **The `DEEPGREEN_RUN_DIR` guard was a tautology** —
  `hardcoded and "DEEPGREEN_RUN_DIR" not in bench` can only be true when the
  preceding check has already failed. It resolves both output paths against a
  temporary directory now, and injecting the original defect makes it FAIL.
- **`--strict` returned 0 on a repository with a known failing check**, because
  the three data-driven checks are guarded by `import pandas` and the system
  interpreter has none: 73 checks became 71, the FAIL became a SKIP, and SKIP did
  not affect the exit code.
- **Thirteen checks added** for clauses that had none. Three of them failed on
  the comments explaining why they exist — the lesson `strip_comments` was
  written for, applied to C-style comments and not to the languages that use
  `#`. A fourth flagged the corrected code, because the ceiling idiom
  `-(-x.samples // batch_size)` contains the floor form as a substring.

The checker also caught me the next day: a helper module dropped into
`python/tensorflow/models/` made five universal checks fail, because they require
every file in that glob to carry the learning rate, the batch size and the shared
harness, and a helper carries none of them. The fix is not to narrow the glob but
to put shared code where shared code lives -- `tools/torch_init.py`. A check that
fires on a file which should not have been there is a check working.

90 checks now, 89 passing. The one failure is Java's `test_loss` and it stays
until the campaign is re-run: the check reads recorded metrics, not source, and
the source now computes it.

---

## 11. What Java's missing test loss actually cost

Deeplearning4j recorded `Double.NaN` for test loss in all 900 of its epoch rows.
The manuscript disclosed this as a single failing check and described the
collapse/pipeline discriminator as degrading gracefully to its remaining arm.
It does not: the expression ANDs two arms and any comparison with NaN is false,
so the conjunction could never fire and Java's sensitivity to pipeline defects
was zero rather than reduced — for the stack carrying five of the twelve
collapses. Test loss is now computed in the same inference pass as the accuracy,
so Java's eval blocks do not measure twice the work.

---

## Smoke test, before committing 57 hours

One epoch of ResNet-18 on Fashion-MNIST, every stack, into a scratch campaign
directory. All seven complete, all seven assert their parameter count, all seven
record a populated test loss.

| stack | test acc | train J | eval J |
|---|---|---|---|
| C++/LibTorch | 86.22 | 1121 | 52 |
| Rust/tch | 86.62 | 1366 | 91 |
| Python/PyTorch | 87.74 | 1416 | 178 |
| Python/JAX | 86.52 | 2058 | 342 |
| Python/TensorFlow | 86.57 | 2371 | 118 |
| R/torch | 86.92 | 11898 | 2422 |
| Java/DL4J | 86.18 | 18367 | 2213 |

Accuracy spread 1.56 pp after one epoch, against 10.34 pp before the BatchNorm
alignment. Two defects were found by running this that no amount of reading had
found: the BatchNorm divergence, and a global `LD_LIBRARY_PATH` I had just added
that made LibTorch's bundled cuDNN shadow JAX's and killed it with a segfault.

---

## Still open

- Re-run the campaign (~57 h) and re-derive every number.
- Rewrite the sections this invalidates: RQ1's mechanism, every VGG-16
  comparison, the collapse attribution, JAX's inference ranking, the boundary
  naming errors, the omnibus *p* bound that rounds the wrong way, the within-cell
  ρ macros, the ε² saturation, the exact collapse test, the Welch-versus-rank
  discussion, the idle-subtraction sensitivity, the bimodal power state.
- Re-derive every numeric claim in `REVIEWERS_RESPONSE.md` from the generated
  macros; several overclaim against the current tree.
- Make the replication package round-trip (13,037 of ~13,230 files differ on
  restore: float truncation, manifest type coercion, list flattening, row order).
- Re-run the claim-evidence review that stopped on a rate limit.
- Unchanged from before: CRediT roles, competing-interest declaration, Zenodo
  deposit, author photographs.
