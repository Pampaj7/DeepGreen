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

**Fix.** `TorchWeightInit` in the Java tree, and `tools/torch_init.py` for
Keras. Both compute fan_out from the weight shape rather than trusting
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

## 12. Fairness audit: the architectures, proved rather than counted

Asked to verify that the architectures really are identical across languages in
every configuration actually trained, and that the test sets are genuinely
shareable. Counting parameters -- which is what the previous round did -- is the
weak form: two networks can agree on a total and differ layer by layer.

`scripts/verify_architecture_parity.py` builds every (stack, architecture,
dataset) and compares the **sorted multiset of parameter tensor shapes**, which
is language-independent because it depends on neither naming nor ordering. Three
conventions had to be normalised, all found the hard way: Keras and DL4J count
batch-norm running statistics as parameters and torch does not; Keras and flax
convolution kernels are (kh, kw, in, out) where torch's are (out, in, kh, kw);
dense kernels are transposed likewise.

**Result: six stacks of seven agree shape for shape on all six blocks.**
ResNet-18 is 62 tensors (20 convolution, 1 dense, 41 rank-1) at 11,181,642 /
11,227,812 / 11,279,112 parameters; VGG-16 is 30 tensors (13, 2, 15) at
14,982,474 / 15,028,644 / 15,079,944. PyTorch, C++ and Rust through the shared
module; TensorFlow, JAX and Deeplearning4j by construction. R cannot be probed
outside the campaign environment -- its torch package will not load Lantern --
and is covered by its startup assertion on the parameter count.

Three things fell out of building it.

**The TensorFlow ecosystem was two ecosystems.** `resnet18.py` imported
`tf_keras` (Keras 2.19) and `vgg16.py` imported `tensorflow.keras` (Keras 3.15),
so the TensorFlow row of a ResNet-18 table and the TensorFlow row of a VGG-16
table were different software stacks. Inside `vgg16.py` the two were mixed: a
Keras 2 `ProgbarLogger` attached to a Keras 3 `fit`, which works only because
Keras 3's version does not accept the `count_mode` it is handed. Unified on
`tf_keras`.

**My own initialiser fix was a silent no-op.** `apply_torchvision_init` tested
`isinstance` against Keras 3 classes while the ResNet is built from Keras 2
layers, whose hierarchies do not intersect -- `Conv` against `BaseConv`. It
matched **zero of 102 weight tensors** and said nothing. Duck-typed on weight
rank now, and the caller refuses a zero. This also corrects something reported
in the previous entry: the 77.4% -> 71.7% I attributed to the initialiser was
run-to-run variation on a change that did nothing. The gain is entirely the
batch-normalisation alignment.

**And the same helper would have destroyed VGG-16's weights.** It walked every
layer including containers, and a container's `get_weights()` returns its
children's tensors concatenated -- so `set_weights([new_kernel] + zeros)` on the
Sequential wrapping the VGG backbone would have zeroed every weight but the
first. Leaves only now. Caught because the parity fingerprint reported 44.9M
parameters where 15.0M were expected.

---

## 13. Fairness audit: the data, file for file

`scripts/verify_data_parity.py` compares what each probeable loader produces
against what the directory itself contains: file counts, class ordering (which
determines the label index every stack is scored against), the first labels of
the unshuffled test split, and pixel statistics.

Counts, classes, ordering and labels agree everywhere. One difference, on one
dataset:

**Tiny ImageNet was resized differently.** `tf.image.resize` defaults to
`antialias=False`; torchvision's `Resize` is antialiased unconditionally,
because PIL applies it whatever the argument says. Invisible on CIFAR-100, which
is already 32x32, and on Fashion-MNIST, which is upsampled from 28x28. On Tiny
ImageNet, which is 64x64 halved to 32x32, the standard deviation of the pixels
was **3.8% wider** in TensorFlow and JAX than in every other stack. Measured on
the same 64 images: 0.226404 without antialiasing, 0.218092 with, against
torchvision's 0.218089 -- agreement to five decimals once corrected.

That is a third of the campaign, and it would have been read as an ecosystem
difference.

**What is proved, and what is not.** The three Python stacks record the mean,
standard deviation and range of their own test split to `data_fingerprint.csv`,
and on Tiny ImageNet they agree: standard deviation 0.2472 in all three, mean
0.44445 / 0.444914 / 0.44445. That is the antialias correction confirmed in the
campaign's own records rather than in a probe.

The other four resize through implementations that cannot be inspected from
Python -- `tch::vision::image::resize`, DataVec's `ImageRecordReader`, R's
`transform_resize`, and the C++ loader -- and they do **not** record a
fingerprint yet. The bridge accepts a `DATAFP` command for them; nothing sends
it. Until that is wired, their data parity rests on the indirect evidence
below, which is good but is not the same thing.

The indirect evidence: one epoch of ResNet-18 on Fashion-MNIST, all seven
stacks, gives a test-accuracy spread of **1.00 pp** (86.47-87.47) and a
test-loss spread of 0.026 -- against 10.34 pp before this round's alignment
work. Seven implementations reaching the same place from the same
initialisation on the same data is strong evidence that the pipelines agree; it
is not a measurement of the pixels.

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

## 14. JAX was the fourth stack initialising differently, and the third BatchNorm convention

One epoch of Tiny ImageNet, all seven stacks, the same architecture from the
same distribution on the same data. Six landed between 8.96% and 10.26%
accuracy with a test loss of 4.31-4.44. JAX landed at 7.40% and 4.68.

Two divergences, and a mistake of mine between them worth recording.

**flax and torch use `momentum` in opposite senses.** flax computes
`ra = m*ra + (1-m)*batch`, so `m` is a retention factor; torch computes
`ra = (1-m)*ra + m*batch`, so `m` is an update factor. `flaxmodels` writes
`momentum=0.1` at all eight of its BatchNorm sites, which in flax's convention
retains 10% of the running average and takes 90% from the current batch --
torch's 0.1 retains 90%. JAX's normalisation statistics were, in effect, one
batch old at evaluation time. That is the same clause that had already needed
fixing in Keras, in the other direction: three libraries, three conventions.

**flax initialises with `lecun_normal`.** Variance scaling on fan_in with a gain
of 1: standard deviation 0.0835 for ResNet-18's 7x7 stem, against torchvision's
0.0253. 3.3x wider, the same class of divergence as Deeplearning4j's 4.6x, and
JAX was the last of four stacks building the right architecture from the wrong
distribution.

**And my first fix made it worse, for a reason worth keeping.** `flaxmodels`
threads one `kernel_init` through every layer *including the final Dense*, and
torch does not initialise a Linear the way it initialises a Conv. Passing
kaiming fan_out to everything gave the 512 -> 200 classifier a standard
deviation of sqrt(2/200) = 0.100 where torch's `nn.Linear` default is uniform on
+/-1/sqrt(512), about 0.026 -- four times too wide, on the layer whose output is
the loss. Accuracy fell to 6.09%. Dispatching on tensor rank, as the Keras and
Deeplearning4j fixes already did, brings JAX to 8.62% and a test loss of 4.4880:
inside the other six stacks' band rather than outside it.

**A caution on the evidence.** These are single runs of a single epoch on a
200-class problem, where run-to-run variation is easily a point or two. They
identified the divergences and they cannot validate the alignment; the campaign's
five repetitions will. Two intermediate readings in this sequence -- BatchNorm
alone, and the first initialiser attempt -- moved in directions I initially read
as evidence, and were not.

---

## 15. The replication package round-trips now

`--check-roundtrip` restores every run to a temporary tree and diffs it file by
file. It was 13,037 of ~13,230 files differing. Five causes, each small:

  * **Float parsing.** `pd.read_csv` defaults to a fast parser that is not
    exactly round-tripping: `1.5169734358444487e-05` came back as `...488e-05`,
    one unit in the last place. Every field is carried as text now -- this is a
    copy, and a copy should be exact.
  * **`%.17g` was my own wrong answer to that.** It writes
    `3.4290280000000002` where the source says `3.429028`: the same float64,
    and a different file.
  * **Row order.** The package is sorted for readability, which groups
    `counters.csv` by phase -- every eval, then every train -- where the campaign
    wrote them interleaved. A `row_index` column carries the original order.
  * **Manifest types.** Flattening to columns made `"seed": "1000"` come back as
    `1000.0`, and an environment variable cannot be a float; lists became
    space-joined strings. The exact JSON is kept alongside as `manifest_json`,
    and the flat columns are a view.
  * **Line endings.** Python's `csv` module defaults to the excel dialect, which
    is CRLF; everything reading these files back writes LF. The harness writes
    LF now.

12,816 of 13,230 identical, 0 differing. The remaining 414 differ in nothing but
their line terminators and were written before that last fix; a campaign
recorded since round-trips byte for byte.

`--out-dir` also exists now, so a round trip can be checked without writing into
`results/campaign_v2/` -- which is the accident this pair of scripts was written
to repair.

---

## 16. Three things the sweep found that were still open

**Deeplearning4j uses no cuDNN, and cannot at this version.** Convolutions go
through nd4j's own im2col and cuBLAS. `CudnnConvolutionHelper` ships in
`deeplearning4j-cuda`, which has no M2.x release -- `dependency:get` resolves
neither `deeplearning4j-cuda:1.0.0-M2.1` nor `deeplearning4j-cuda-11.6:1.0.0-M2.1`
-- and no cuDNN class or jar reaches the classpath. The run log says
`Loaded [JCublasBackend]` and carries no convolution-helper line at all.

It is the only stack of seven not using cuDNN, and one of the two most
expensive. Not fixable here, so it goes in the specification and has to go in
the manuscript: "Java/DL4J costs 6x" otherwise reads as a statement about a
language when part of it is a statement about a missing convolution backend.

**`train_acc` is empty for five stacks of seven** -- populated only by JAX and
TensorFlow -- and is read by no analysis anywhere. Five languages of work for a
column nothing consumes is the wrong trade; S5 now says what is actually
recorded instead of implying seven, which is the same correction the Java
`test_loss` disclosure needed.

**Four scripts were in no pipeline.** `17_window_calibration.py`,
`measure_idle.py` and `probe_reported_window.py` feed the `\vCalib*`, `\vIdle*`
and `\vMech*` macros, and `12_paper_numbers.py` reads their outputs behind
`if path.exists()` -- so on a clean checkout the manuscript would have compiled
with those numbers silently absent. They are in `run_all.sh` now, in a section
labelled as measuring the host rather than re-deriving the campaign, together
with the two new parity verifiers.

---

## 17. The pixels, proved for all seven

Instrumenting the four non-Python stacks turned out to be worth it three times
over, and each time the first reading was wrong for a reason worth recording.

**A per-batch fingerprint measures the wrong thing.** Rust looked 4.9% away from
PyTorch in standard deviation; equalising the sample -- 393,216 values each
rather than 393,216 against 786,432 -- brought it to 1.0%. And at one batch
each, C++ still sat 1.7% away in the mean, which is not a resize difference at
all: the test split is enumerated in class order and unshuffled, so the first
128 files are one or two classes and two loaders that enumerate differently are
summarising different pictures. Over the whole split the statistic depends on
the set and not the order, which is the question worth asking.

**Then the real finding.** Over Tiny ImageNet's whole test split, 30,720,000
values apiece:

| standard deviation | stacks |
|---|---|
| 0.263919 – 0.263923 | C++, Java, R |
| 0.258406 | Rust |
| 0.256103 – 0.256110 | PyTorch, TensorFlow |

Means agreeing to 0.1% while variances differ by 3.0% is the signature of a
resampling filter, not of different content -- and on a controlled 600 images
the choice spans 0.2483 (bilinear, antialiased), 0.2560 (bilinear) and 0.2699
(nearest), 8.7%. Four stacks of seven were seeing different pixels on a third of
the campaign.

**The fix removes the question rather than aligning four answers.** Every image
is resized once, offline, to 32x32 -- `scripts/normalise_dataset_resolution.py`,
240,000 images -- so each stack's own resize is a no-op and they decode
identical pixels, which is the position CIFAR-100 was always in. Expressing one
filter in four libraries would have meant discovering whether DataVec and R can
express it at all.

Two implementation choices worth stating. The directories keep their names and
the originals move aside with an `_original` suffix, because the resolution is
hardcoded at some thirty sites across Rust, R, Java and the Python wrappers and
thirty edits is how a constant drifts. And the converters in `dataloader/` now
produce 32x32 as well, so the invariant holds however the data is made rather
than only when someone remembers the script.

**Two residues, both real, both surfaced by the change rather than hidden.**
Rust still called `tch::vision::image::resize` on an already-32x32 image, which
is not the identity: it resamples in uint8 and rounds, and that alone held it
1.3% away from the other six. And C++ could not start at all, because the
resize script copied only images and Tiny ImageNet ships a `classes.json` its
loader reads -- my omission, now fixed both in the script and on disk.

**Result.** All seven stacks report mean 0.443782 and standard deviation
0.256110 over 30,720,000 values, identical to six decimal places. The pixels are
no longer an assumption.

---

## 18. A day lost to a CPU fallback, and what it cost to find

After the datasets were pre-resized, Rust appeared to become 11x slower at
training on Tiny ImageNet -- 24 s to 266 s -- while every other stack got
faster. It blocked the campaign for a day. Five hypotheses, in order, each
plausible and each refuted by measurement:

1. **The data-fingerprint loop I had just added.** Refuted: with it disabled the
   epoch still took 266.02 s against 262.38 s with it on.
2. **The resize guard.** Refuted: runs with and without were both ~265 s.
3. **Machine contention.** Real -- a stray `resnet_tiny` from a killed background
   job was found at 818% CPU with the load average at 27.9 on 24 threads, and
   another agent independently flagged the same thing. But refuted as the cause:
   on a quiet machine, load 1.99, the epoch still took 266.86 s.
4. **The pre-resized data.** Refuted by the differential nobody had run: the same
   binary against the original 64x64 tree took 278 s, slightly *slower*.
5. **`NVIDIA_TF32_OVERRIDE`.** Refuted: 132.48 s with it unset against 132.45 s
   with it set.

What the measurements were actually saying, and what should have been read
sooner: the cost was flat at 2.66 ms per image whether the images were 32x32 or
64x64, which is not what decode looks like; a micro-benchmark put the whole load
path at 0.051 ms per image, faster than the campaign's own figure; and the
accelerator was drawing **23.4 W against a 24.3 W idle floor** while the CPU
package sat 26 W above its.

The stack was training on the CPU. `readelf -d` on the Rust binaries showed no
CUDA library at all.

The cause was mine. `cargo build` against a CUDA LibTorch produces a binary that
silently runs on the host: torch-sys asks the linker for `torch_cuda`, nothing
in the Rust code references a symbol from it, `--as-needed` drops the
dependency, and `tch::Device::cuda_if_available()` reports `Cpu`.
`scripts/build_rust_cuda.sh` exists for precisely this reason and says so in its
first paragraph -- and I rebuilt with plain cargo five or six times over the
day. Rebuilt through the script: `Using device: Cuda(0)`, two CUDA libraries
linked, 5.07 s per epoch against the campaign's 4.58 s.

The same investigation turned up a second Rust defect. The uint8 resize guard
had been applied to `tiny.rs` alone, so `fashion.rs` still resampled an already
32x32 image and came back 1.4% away from the other six stacks in pixel standard
deviation -- 0.333095 against 0.337890. Guarded in all three loaders now, and
Rust reports 0.337890 like everyone else. Fashion-MNIST is where it showed
because Tiny ImageNet's loader was the one already fixed.

**The check that would have caught it in a second now exists.** `readelf -d` on
every released Rust binary, asserting a `NEEDED` entry for a CUDA library. It
immediately found two stale scratch binaries left behind by this investigation
as well.

Worth stating plainly for the manuscript, because it is the paper's own thesis
turned on the paper twice more. A silent CPU fallback is the defect the
catalogue already documents for this stack; the specification had a clause about
device placement; and the checker verified that a *definition string* appeared
in a source file rather than that the *built artefact* could reach the
accelerator. An unenforced clause is a clause that will drift, and a clause
enforced against source rather than artefact is barely enforced at all.

---

## 19. Both declared gaps closed

**R's architecture is verified shape for shape.** It was never a property of the
binding: R's `torch` loads `liblantern.so`, which links `libcudart.so.12` while
the R bundle ships that under a hashed name, so without a CUDA 12 runtime on
`LD_LIBRARY_PATH` the package reports "Lantern is not loaded" and every probe
fails. `tools/stack_environments.json` had recorded the fix all along, with a
note explaining exactly this; I had set `R_LIBS_USER` and not the CUDA path.
With the campaign's own environment, R fingerprints like the others, and **all
seven stacks now agree shape for shape on all six blocks** — ResNet-18 at 62
tensors (20 convolution, 1 dense, 41 rank-1), VGG-16 at 30 (13, 2, 15).

**Deeplearning4j's missing cuDNN path is now evidenced and bounded.** It cannot
be fixed: `libnd4jcuda.so` carries a `NEEDED` entry for `libcublas.so.11` and
none for cuDNN, the helper class and every `org.bytedeco` cuDNN preset are
absent from the classpath, and `deeplearning4j-cuda` has no version at all in
Maven Central's metadata — verified against the metadata endpoint with
`deeplearning4j-modelimport` as a control. 1.0.0-M2.1 is the latest release.

So it is bounded instead. Disabling cuDNN in a stack that has it, holding
hardware, model and data constant — ResNet-18, batch 128, median of three —
costs **13.3× the energy and 16.7× the time**: 100.3 J and 0.353 s against
1333.2 J and 5.874 s. That is an upper bound on what losing cuDNN can cost
rather than an estimate of Deeplearning4j's own penalty, since nd4j's im2col
path is not PyTorch's non-cuDNN fallback — but it exceeds the whole
Java-versus-cheapest gap the campaign measures, which settles the question of
whether this belongs in a footnote. It does not.

A check now watches `libnd4jcuda.so` and fails if a future version gains cuDNN,
so the claim cannot quietly stop being true. 92 checks, 91 passing.

---

## 20. The campaign caught what 92 static checks could not

The second campaign started at 23:09:25 on Sunday 30 August. Fifteen hours in,
at job 55 of 210, four jobs had failed and all four were the same one: JAX training
VGG-16.

    flax.errors.InvalidRngError: Dropout_0 needs PRNG for "dropout"

The shared head that `scripts/export_torchscript_models.py` defines is
`512 -> 512, ReLU, Dropout(0.5), -> classes`, and section 13 records aligning six
stacks onto it. Flax got the layer and not the randomness it needs: `train_step`
called `apply_fn` without an `rngs` argument, so the first training batch of
every JAX VGG-16 run raised, at epoch 1, after the tracker had already opened a
block and written the parameter count. The count was right -- 15,028,644, the
manifest's figure -- because `nn.Dropout` has no parameters. Every guard the
revision added passed, and the run still could not take one step.

That is the honest limit of a conformance suite built from source text. It
proved the architectures matched, the pixels matched and the protocol matched,
and none of that requires the code to run. The check that found this was the
campaign.

Fixed by threading the stream explicitly: `create_state` initialises with
`{"params": rng, "dropout": rng}`, `train_step` takes `dropout_rng` as an
argument and passes `rngs={"dropout": dropout_rng}`, and the loop splits a fresh
key per step so the mask is not shared across batches. Verified on CPU, off the
measured device, so the run in flight kept its measurement: 15,028,644
parameters and a loss that moves.

The other six were checked at the same time, since a head that one stack got
wrong is a head worth re-reading everywhere. PyTorch, Rust and C++ load the
exported TorchScript module and inherit it. TensorFlow has `layers.Dropout(0.5)`,
R has `nn_dropout(p = 0.5)`, DL4J has `DropoutLayer.Builder(0.5)` -- and DL4J's
argument is an activation *retain* probability where the other six take a drop
probability, which at 0.5 is the same number either way. ResNet-18 has no
dropout in any stack, which is correct: `build()` gives it the plain torchvision
`fc`.

Four runs need redoing, all of them from before the fix: JAX/VGG-16 on cifar100
rep0, fashionmnist rep0, and tinyimagenet rep0 and rep1. Their directories hold
a two-line `counters.csv` from the block that opened before the crash, and the
driver refuses a non-empty run directory, so they are deleted first and then
replayed:

    rm -rf results/campaign_v2/Python-JAX_vgg16_{cifar100_rep0,fashionmnist_rep0,tinyimagenet_rep0,tinyimagenet_rep1}
    python scripts/run_campaign.py --repetitions 5 --ecosystems Python/JAX --models vgg16

The driver skips run directories that already hold data, so this replays exactly
those four. All fifteen JAX VGG-16 runs then come from one source revision,
which is the property that matters: the eleven still ahead in the plan pick the
fix up on their own.

---

## 21. R holds the GPU at 5.7% utilisation, and JAX does not

Sampled during `R/torch resnet18/tinyimagenet rep1` at job 57: **5.7% GPU
utilisation over 28 samples, 128 W**, with three R processes at 92%, 58% and 58%
CPU. The GPU is attached and computing -- `nvidia-smi` shows 1,128 MiB held by
the R interpreter itself -- it is simply idle most of the time, waiting on the
loader.

That matters because after section 18 a stack drawing half the power is exactly
the shape a silent CPU fallback makes. It is not one here. R holds device
memory, and the deficit is utilisation, not placement. Nor is it a protocol
mismatch: R runs the two loader workers the spec mandates, the same two every
stack gets. R's workers are simply slower per image, which is an ecosystem
property and precisely the kind of thing the study exists to measure -- but it
belongs in the paper as a stated mechanism, with the utilisation figure
attached, rather than as an unexplained gap in a bar chart.

The idle floor was measured while writing this, from a cooldown gap between two
jobs: **0% utilisation, 27.9 W, 210 MHz, 185 MiB**. So the 128 W R draws is four
and a half times idle, and the clocks are not pinned -- persistence mode is on,
which keeps the driver resident, but the SM clock still swings 210 to 1740 MHz
with load. `scripts/sample_gpu_utilisation.sh` now records utilisation, power,
clock and memory at 1 Hz for the rest of the campaign, to `results/`, joinable to
each run by `manifest.json`'s mtime for the start and `counters.csv`'s for the
end.

### The first version of this section was wrong about JAX

It read "R and JAX draw half the power", from a per-stack mean that put JAX at
135 W. That mean included the four crashed VGG-16 runs of section 20, each of
which contributes a single `counters.csv` row from the block that opened before
the exception -- about 94 W of nothing. Excluding every run without its full 30
training blocks:

    C++/LibTorch   297 W (251-344)      PyTorch      262 W (214-337)
    TensorFlow     285 W (232-337)      Rust/tch     258 W (201-320)
    Java/DL4J      265 W (192-338)      JAX          218 W (160-327)
    R/torch        134 W (125-141)

JAX is unremarkable, and inside its own training blocks the repaired
`vgg16/cifar100 rep1` drew **322 W** -- above every stack's mean. R is the sole
outlier, and the striking thing is the range: **125 to 141 W across eleven runs**
spanning both architectures and all three datasets, where every other stack
varies by 100 W or more with the network it is training. R's GPU power is set by
its loader rather than by the model, which is what being starved looks like.

The lesson is procedural and worth stating: an aggregate computed over a
directory of runs will quietly include the broken ones. Every analysis of this
campaign must filter on complete runs first, and `scripts/` should do it in one
place rather than in each script that reads `counters.csv`.


---

## 22. Four boundary errors in the manuscript, fixed against the code

These do not depend on the re-execution, so they were done while it ran. Each
was checked against what the pipeline actually computes rather than against what
the sentence sounded like.

**"Intel RAPL" on an AMD machine** (§2). The instrument paragraph introduced the
CPU counter as "Intel RAPL". The campaign host is AMD Zen 3. RAPL is Intel's
interface by origin, AMD exposes an equivalent set of model-specific registers,
and Linux presents both through `powercap` -- whose sysfs nodes are named
`intel-rapl` for either vendor. The text now says so, which also pre-empts a
reader who finds `intel-rapl:0` in our data and concludes we misreported the
hardware.

**Our own row in the related-work table** (§2) gave this study's boundary as
"GPU" where every other row reads "CPU + GPU". We measure the accelerator board
*and* the CPU package, and the row understated the work in a table built to
compare exactly that. Now "CPU + GPU".

**`\vAuditSpreadGpuOnly` described as "a counter boundary"** (§4). It is not:
`12_paper_numbers.py` computes it from CodeCarbon's `gpu_energy`, so it is the
estimator's own accelerator term. Its companion `\vAuditSpreadConfounded` is the
estimator's total. The point the sentence makes is stronger stated correctly --
both figures come from one instrument, and only the charged region changes -- so
it now says that.

**"Every figure we report is whole-machine"** (§7, limitations). The paper
defines whole-machine 1,800 lines earlier as what a *wall meter* sees, and
distinguishes it from counters precisely because counters do not see it. The
intended claim was that the figures are *un-baselined*: the counters include the
machine's static draw within their own boundary. Un-baselined and whole-machine
are now kept apart, with the boundary named.

Checked structurally rather than by compiling: `pdflatex` would occupy a core,
and CPU package energy is being measured right now. Braces balance at 1,110, and
all 228 `\v` macros the text uses are defined. A full build follows the campaign.

---

## 23. I overwrote the manuscript's inputs, and then built the guard

Fixing `\vOmnibusMaxP` (below) meant re-running `12_paper_numbers.py` to see the
new value. It regenerated all 274 macros -- from the campaign in flight. `\vRuns`
went 210 to 59, `\vRepetitions` 5 to 2, `\vBlocks` 12,600 to 3,540. Worse, an
earlier commit had already done the same to twenty-seven table files under
`results/revision/tables/`, because running `09`, `11` and `17` to check a
refactor also *writes*, and `git add -A` swept the result in.

Nothing about that was a bug. Every script did what it was asked. The defect is
that the analysis writes into the directory the manuscript reads, so any run of
it is a publication, and monitoring a live campaign is indistinguishable from
producing the final numbers.

Restored from git (`d2137867^`), verified: `n_runs` is 5 again in
`v2_between_run_statistics.csv`, and `\vRuns` is 210.

The fix is not to forbid monitoring runs -- they have already earned their keep
this afternoon -- but to send them somewhere else. `common.tables_dir()` returns
a `_partial` sibling while fewer than `EXPECTED_RUNS` runs are complete, and
`save_table`, `write_table_path`, `12_paper_numbers`'s `OUT` and
`13_paper_figures`'s `FIGDIR` all go through it. Reads are different from
writes: `TABLES / name` resolves to the partial copy when this run has produced
that name and to the committed one otherwise, so a partial pipeline reads its
own fresh v2 tables and the unchanged v1 tables, and overwrites neither.

Three tables and six figures had been bypassing `save_table` with a direct
`to_csv` or `savefig`, and the figures were the more dangerous of the two: a
plot carries no run count on its face, so a substituted figure looks exactly
like the right one. Both routes now go through the diversion. Verified by
running the whole v2 pipeline against the live campaign: `paper/generated/`,
`paper/figures/` and `results/revision/tables/` come out untouched, and 53
tables, six figures and a `numbers.tex` reading `\vRuns 67` land in the
`_partial` directories, which are gitignored.

One thing the partial run showed in passing, worth keeping: C++/ResNet-18 on
CIFAR-100 trains at **905.2 J** per epoch in the first campaign and **904.0 J**
in the second, 0.13% apart, on rebuilt binaries and re-encoded data. The
instrument reproduces across campaigns.

`15_convergence.py` fails on partial data -- `chi2_contingency` gets a zero
expected frequency with two repetitions. That is the same per-ecosystem
chi-squared the response already commits to dropping, so it is on the list
rather than a new problem.

## 24. An upper bound that rounded downward

`\vOmnibusMaxP` was built as `f"{om.p.max():.0e}"`, and the manuscript quotes it
as `$p \le \vOmnibusMaxP$`. The maximum is 1.1770e-5, which prints as
`1\times 10^{-5}`: a bound the data violate. Round-to-nearest is right for
reporting a value and wrong for reporting a limit -- a limit may only ever be
loosened by rounding.

`sci_upper()` ceilings the mantissa instead, to two significant figures, and the
value becomes `1.2\times 10^{-5}`. Checked against seven cases including
9.99e-6, which must round up to 1e-5 and does, and 4.7e-11, which the old
expression would have mangled outright: it built the exponent with
`.replace("e-0", ...)`, and past 1e-9 there is no leading zero left to match.

---

## 25. The exact collapse test, and a first sign the attribution was right

`15_convergence.py` tested homogeneity of the collapse rate across ecosystems
with a permutation test, and computed chi-square beside it because the
manuscript contrasts the two. Neither is the right answer for the table it has:
seven ecosystems, ten runs each, twelve collapses. Chi-square wants five
expected per cell and has 1.71. And the permutation statistic is
`max(rate) - min(rate)`, which sees only the extremes -- it cannot distinguish
0, 0, 0, 10, 20, 40, 50 per cent from 0 and 50 with five groups sitting at the
mean. That is why it returned p = 0.0676 where chi-square returned 0.0065: a
property of the statistic, not evidence about the data.

`common.freeman_halton_exact` now gives the exact answer. It is the r x c
generalisation of Fisher's test: condition on both margins, and sum the
multivariate hypergeometric probability of every table no more likely than the
observed one. For the first campaign's table that is **p = 0.0040**, from a few
thousand terms, with no approximation and no seed. Validated against
`scipy.stats.fisher_exact` on 2x2 tables, where the two agree to six decimals,
and against the requirement that the probabilities sum to 1.

Chi-square is kept, guarded: `chi2_contingency` raises on a zero expected
frequency, which a partial campaign produces, and a test we are arguing against
must not be able to stop the pipeline. It reports NaN instead.

### Zero collapses so far

Running it against the campaign in flight: **0 collapses in 20 VGG-16 runs on
the many-class datasets**, where the first campaign had 12 in 70, or 17.1%.

If the rate were unchanged, twenty runs would produce none with probability
0.023. That is suggestive and not yet conclusive -- the exact 95% upper bound on
the current rate is still 13.9%, so a rate near the old one is not excluded.
But the direction is what section 13 predicted: with every stack initialised
from the same exported weights, the phenomenon the first campaign distributed
across ecosystems has not appeared at all. Worth watching to 210 rather than
claiming now; if it holds, the collapse section becomes a much shorter and much
stronger argument.

---

## 26. The response document claimed a campaign that no longer stands

`REVIEWERS_RESPONSE.md` opened with:

> **Status: the replicated campaign has been executed in full --- 210 runs of
> 210, no failures.** Every item previously marked *REQUIRES RE-EXECUTION* is
> now closed with measured numbers.

That was true when written and is not true now. The campaign it describes did
complete; auditing it afterwards found the three confounds that forced the
re-execution, so its numbers cannot be quoted as closing anything. Sending this
document to the editor in that state would assert, in its first paragraph, a
completeness the work does not have.

The status paragraph now says what is actually the case: a campaign completed,
its audit found that TF32 policy, four different VGG-16 networks and four
different initialisers had confounded three headline claims, the stacks were
aligned, and the campaign is running again. It also says plainly that every
quantity below it is from the superseded campaign.

A closing table names exactly what changes and why -- the collapse counts and
their per-ecosystem spread, every VGG-16 energy figure, RQ1's mechanism, JAX's
inference ranking, the loader-configuration table (which describes the state
that was *found*, not the state that was measured, and still lists MATLAB), and
the permutation p-value now replaced by Freeman--Halton. What does not change is
also named: the instrument comparison, the padding and window-floor results, the
boundary argument and the source-level defects are properties of the tooling and
the code, not of the campaign's numbers.

The remaining numeric claims still need re-deriving from the macros one by one.
This closes the part that could mislead on its own.


---

## 27. Closing the campaign, and auditing the pipeline against it

Two days of finishing work. The campaign ended on Wednesday morning; everything
below happened on Thursday and Friday, and almost all of it was found by
attacking the pipeline rather than by reading it.

### The four replayed runs

The campaign's last run started at 05:47 UTC on Wednesday 2 September, 210 of
210. (Times in this section are UTC, which is what the manifests record; section
20's clock is local, two hours ahead.) Four of those runs were not part of the
interleaved schedule: the JAX/VGG-16 runs of section 20, which raised on the
Flax dropout PRNG before the fix. Their directories were deleted and the
configuration replayed on Thursday evening -- `tinyimagenet_rep0`,
`fashionmnist_rep0`, `cifar100_rep0` and `tinyimagenet_rep1`, starting at 18:22,
18:33, 18:42 and 18:52 on 4 September. All fifteen JAX VGG-16 runs now come from
one source revision, which is the property that mattered.

**What it costs.** Four runs of 210 sit two days outside the interleaved window,
and for them the design's central protection -- drift spread across conditions
rather than aliased onto whichever ecosystem ran last -- does not hold. That is
what the calibration is for, and it is why I did the calibration before writing
the threats section rather than after.

### The calibration was measuring a harness change and calling it drift

**Defect.** `results/calibration/` held five Python/PyTorch ResNet-18 /
Fashion-MNIST runs from 29 August, produced by the harness that pinned
`matmul.allow_tf32 = False`, `cudnn.allow_tf32 = False` and
`cudnn.deterministic = True` for the Python stacks. Every run of the second
campaign records `DEEPGREEN_TF32=1`. `17_window_calibration.py` joined the two
anyway: it matches on ecosystem, model, dataset and phase, and nothing in it
asked whether the two windows were produced by the same instrument.

**Evidence.** Against the first campaign the table read **-0.21 % at 0.61
standard deviations** of the within-window spread, and the manuscript quoted that
as drift "below the noise the design already carries". Against the re-executed
campaign the same five runs read **201 % apart at 293 standard deviations**.
Neither figure is drift. The first is drift plus a precision change that happened
to land on top of it; the second is the precision change alone, wearing a
between-window-drift label and a manuscript sentence saying it is small.

**Fix.** `comparability_objection()` refuses the comparison when the two windows
do not record the same precision policy, and when every calibration run predates
the campaign's earliest run -- a calibration that predates the campaign is a
comparison across harness versions, it cannot be re-executed to match a harness
that did not exist when it ran, so the campaign is the fixed side and the
calibration is the one that has to be produced again. Two details the refusal
needed. Run start times now come from `machine_state.utc`, written when the
harness opens the run, rather than from `counters.csv`'s mtime, which is the
*end* of the run and does not survive a copy or an rsync. And both roots are
walked through the same completeness gate, because one leftover directory under
either of them reads as a run recording no precision policy and would disable the
comparison permanently while blaming a key every real run does record.

The old calibration is kept as `results/calibration_first_harness/`. Five runs
under the current harness on Thursday, 19:15 to 19:50, immediately after the
replays above.

**Result.** Between-window drift is **-1.25 % at 1.62 sigma** on training and
-3.23 % at 3.95 sigma on inference, against within-window CVs of 0.59 % and
0.95 %. Small on training; on inference it is nearly four sigma of a very tight
spread, and I would rather quote it that way than round it to "negligible".

### Five pipeline defects, found by attacking the pipeline

The reviewers I set on the analysis were told to try to make it produce a wrong
number, not to read it for correctness. All five below came out of that, and
four of them had already reached the manuscript.

**Twelve macros were reading a superseded replication package.**
`12_paper_numbers.py` built two frames from `results/replication/` --
`metrics.csv.gz` for `collapse_mechanism_facts` (`\vCollapseLossDeviationMax`,
`\vCollapseDiscordantCells`, `\vCollapseSharedStacks`, `\vSigCollapseJava`) and
`codecarbon.csv.gz` for `reexecution_facts` (`\vLateRuns`, `\vLateConfigurations`,
`\vLateEcosystems`, `\vLateGapDays`, `\vLateBlocks`, `\vInterleavedRuns`). That package
is an *output* of the pipeline and nothing in `run_all.sh` rebuilt it, so for a
fortnight it held the first campaign while ten macros quoted it as the current
one. Worse than staleness in one case: in the first campaign Python/PyTorch built
torchvision fresh and loaded no shared module, so the discordance
`\vCollapseDiscordantCells` reports was measured between stacks that did not share
weights, under a sentence in the paper asserting that they did. Both frames come
from the campaign now, through `common.read_campaign_metrics` and
`09_campaign_v2.collect`, and the package is built at the *end* of `run_all.sh`
where an output belongs. It also refuses to build from a partial campaign, which
every other consumer of the raw tree had been given and it had not.

**And the checker was reading it too, which is where the other two macros came
from.** `check_consistency.py`'s three S5 metrics checks read the same
`metrics.csv.gz`. Java/DL4J's `test_loss` is NaN in all 900 of its rows in the
first campaign and in none of the second, so the checker reported a failure that
belonged to a campaign the paper does not describe -- and `12_paper_numbers`
calls `check_consistency.run()` to derive `\vConformanceFailName` and
`\vConformanceFailDetail`, so the manuscript disclosed "one failing check:
Java/DL4J `test_loss`" as a property of the campaign it reports. It was a
first-campaign fact. The checks read the campaign now, through the same
completeness gate the tables are built behind, and they emit a result whether or
not there is a campaign to read: the old `if records.exists()` made
`\vConformanceChecks` depend on whether a superseded artefact happened to be on
disk, 92 with it and 89 without.

**A SKIP counted as a PASS.** `\vConformancePassing` was `len(results) -
len(failing)`, which is "everything that did not fail". A checker run in which
three checks could not read their input would therefore have published 92 of 92
passing. The checker's own note says a check that stops running is
indistinguishable from one that passed; this is where that becomes a number in a
paper. `apparatus_facts` now raises rather than emitting the conformance macros
if any check skipped, and counts PASS explicitly. In the same pass I made
`macro()` refuse any value that formats as `nan`: with zero collapses,
`chi2_contingency` raises on a zero expected frequency, `15_convergence` records
NaN, and `paper.tex` had typeset "a chi-square test on that table returns
p = nan, which would license a claim that ecosystems differ in robustness".

**Stale-table fallbacks on zero rows.** `TABLES_RESOLVER` falls back to the
committed copy of a table when the current run has not produced it -- which is
correct for a partial campaign and silently wrong when a script *dies* before
writing. Three of `15_convergence`'s six tables were in that position: with an
empty quality table `homogeneity_test` reduced `max()` over an empty array and
`wasted_energy` divided by zero, so the script wrote three tables, raised, and
left `12_paper_numbers` to read the other three from the first campaign's
committed copies. `collapse_signature` had the same shape from the other
direction: it was guarded by `if len(sig)`, and with zero collapses it wrote
nothing at all, so `\vSigCollapseN` resolved to twelve runs that are not in this
campaign. Every table a script owns is written now, header and no rows when there
is nothing to report, and `only_row()` names the missing macro instead of raising
`IndexError` at position 0.

**`measure_idle.py` was measuring the idle host from inside the pipeline.** It
ran unconditionally in `run_all.sh`, so a full pipeline run measured the machine
while being the only load on it. The CPU package term came out 3.6 W above a
reading taken an hour earlier on a quiet host, and it wrote that straight over a
manuscript input. An idle baseline measured by the load is the defect this paper
catalogues, committed by this paper. The script records the one-minute load
average and a UTC timestamp in the file now, refuses to write above 0.5 without
`--force`, and writes through `write_table_path` like every other table;
`run_all.sh` runs it and `probe_reported_window.py` only under
`DEEPGREEN_MEASURE_HOST=1` and otherwise prints when the committed baseline was
measured. I re-measured on a quiet host this morning: 60.01 s at load 0.26,
**24.59 W accelerator, 38.91 W CPU package, 63.50 W together**, 2026-09-05
10:44:39 UTC.

### The precision numbers in this log had no script behind them

**Defect.** The four-cell table in section 1 -- 0.352 s / 78.3 J against
1.221 s / 407.6 J, the 4.81x -- has no script, no CSV and no commit anywhere in
this repository's history. Nor does section 19's Deeplearning4j bound, "13.3x the
energy and 16.7x the time", 100.3 J and 0.353 s against 1333.2 J and 5.874 s. I
went looking for the code that produced them and there is none.

**Evidence that they are also wrong.** Both tables are labelled per step. The
campaign's own per-step figures for the same two configurations, CIFAR-100
training at 390 steps per epoch, are about 0.0075 s and 1.9 J for C++/LibTorch
with TF32 on and about 0.022 s and 7.5 J for Python/PyTorch as the pre-fix
harness ran it. The logged figures are roughly 50 times those. They are almost
certainly 50-step totals with a per-step label, which is the same class of error
as kilowatt-hours labelled as joules, in a log written to catalogue it.

**Fix, in two parts, because the two campaigns already contain the experiment.**
`results/analysis/18_precision_ablation.py` reports the ablation that was sitting
on disk unread: PyTorch changed precision policy between the campaigns and six
stacks did not, on the same configurations, five repetitions and 150 training
epochs per cell. `scripts/probe_tf32.py` is the kernel-level mechanism check
underneath it -- six cells over `matmul.allow_tf32`, `cudnn.allow_tf32`,
`cudnn.deterministic` and `cudnn.enabled`, a wall-clock warm-up budget rather
than a step count so the GPU is out of its idle P-state, cells shuffled within
each repeat, and the counter window recorded in the table so a reader can check
it against NVML's 10 Hz tick instead of trusting it.

**What they reproduce.** Campaign contrast: **3.14x** GPU energy per training
epoch for Python/PyTorch on ResNet-18 / CIFAR-100, with C++, R and Rust as
controls at 1.00x, 0.99x and 1.01x -- at most 1.3 % of movement in three stacks
that did not change policy. Kernel probe: denying TF32 to cuDNN costs **3.62x**
the GPU energy and 3.12x the time, 4.05x with `deterministic` on top, and
disabling cuDNN outright costs **13.08x the energy and 16.30x the time**. The
kernel ratio sits a little above the end-to-end one, which is what it should do
and what section 1 claimed without being able to show.

**Comparability, which was the hard part.** Only CIFAR-100 supports a clean
contrast, and only for four stacks. VGG-16 is out entirely because it ran as four
different networks (section 2). TensorFlow is out on architecture -- Model
Garden's projection shortcut -- and Java on three axes of hand-built graph.
Fashion-MNIST and Tiny ImageNet are out because
`scripts/normalise_dataset_resolution.py` rewrote their *training* splits to
32x32 between the campaigns, so on those two every stack decodes different pixels
and does different loader work; I checked that against the trees each campaign
actually read rather than taking section 17's word for it, and 300 sampled
CIFAR-100 training files came back pixel-identical while 300 each of the other
two came back changed. JAX carries a caveat rather than an exclusion for
`drop_remainder` (390 steps against 391, 0.26 %), because a control group of one
is not a control group. One confound survives inside the comparable cells and is
stated rather than removed: v1 PyTorch built torchvision eagerly and v2 loads the
exported TorchScript module, so 3.14x is an upper bound on the precision effect.

### Persistence mode: sections 9 and 21 disagreed, and the manifests settle it

Section 9 records persistence mode as **Disabled** on this host and offers it as
the most plausible mechanism for a bimodal accelerator power state. Section 21,
written three weeks later from `nvidia-smi` during a run, says "persistence mode
is on, which keeps the driver resident". Both cannot be true of the same campaign
and I had left both standing.

The manifests answer it without either of us remembering: `machine_state.gpu` is
written per run, and all 210 runs of this campaign record **Enabled**, on driver
595.84, with a 350.00 W limit. Section 21 is right for the campaign the paper
reports; section 9's reading was taken before the driver fix
(`scripts/fix_nvidia_driver.sh`) and describes the machine as it was, not as it
measured. The bimodal power state therefore needs a different explanation than
the one section 9 offers, and the paper no longer offers that one. This is the
argument for recording machine state per run in one sentence: two entries in my
own log contradicted each other and the data resolved it in a minute.

### The manuscript rewrite, and what reviewing it in pieces caught

I rewrote `paper/paper.tex` yesterday across all sections -- 1,043 lines inserted
and 398 deleted, `git diff --stat` against the last commit -- and it builds to
36 pages. I did it in six clusters rather than end to end (apparatus and
instrument; the campaign and its statistics; RQ1 and the precision ablation;
quality, accuracy and the scenario; the collapse finding and the defect
catalogue; threats, related work and future work), and put each cluster through
an adversarial review before starting the next, on the theory that a reviewer
given one section and told to break it does better than one given a paper and
told to check it. Three catches worth recording, because none of them is a typo
and all three are the same shape -- a number whose *denominator* or *population*
was not what the sentence said.

**Accuracy per kilojoule is not per kilojoule of the run.** `09_campaign_v2.py`
computes it as final test accuracy over the run's **training** energy, evaluation
excluded. That is defensible -- the accuracy is what training bought -- but the
text read as though it were the run's total, and the ratio it quotes would move
if it were. Stated at the point the number appears now.

**A 28-sample probe was standing in for a 1 Hz record.** Section 21's "5.7 % GPU
utilisation over 28 samples" was sampled by hand during one R/torch run, and the
draft carried it into the results as R's utilisation. The 1 Hz record covers 157
of the 210 runs and gives **4.5-5.0 % on ResNet-18 and 12.1-13.4 % on VGG-16**,
which is a different number, a different sample and a different claim. The probe
identified the phenomenon; it could not measure it. `19_gpu_utilisation.py`
reports which runs the record covers rather than averaging over whatever happens
to be there, for the same reason an energy figure over an unstated subset is not
a figure.

**"Six control stacks" were three.** The precision ablation's control group is
the stacks whose policy did not change *and* whose cell is comparable, which on
CIFAR-100 / ResNet-18 is C++, R and Rust. The draft said six, counting every
stack that did not change policy and ignoring that TensorFlow and Java are
excluded on architecture and JAX on its measurement window. The same error in a
different place: `tab_control.tex` printed the LibTorch family size as "shared
module + 1", a literal wearing the shape of a derivation, where the group is
defined in `14_v2_statistics` and its size is in the table. Both are derived from
the data now, and the shared-module column is headed "shared module & build",
because that is the pair R shares neither of.

---

## Still open

Everything above this line is done. What is left is either administrative or is
a limitation of the design that no amount of re-running fixes.

- **CRediT roles, competing-interest declaration, Zenodo deposit, author
  photographs.** Unchanged from before, and all four are submission paperwork
  rather than work on the study.
- **The accelerator-saturation threat.** At 32x32 the workload does not saturate
  the card -- mean utilisation runs from 4.7 % (R/torch, ResNet-18) to 79.9 %
  (Java/DL4J, VGG-16) -- so the campaign compares whole pipelines rather than
  saturated kernels. This is the one open item that re-execution cannot close,
  because the workload is the thing at issue. I had been writing that a
  saturating workload would *compress* the ecosystem spread, on the grounds that
  the spread is dominated by host-side work; that does not follow and it is out
  of both documents now. Which way saturation moves the ranking depends on where
  the spread comes from, and a black-box comparison of this shape cannot
  establish that. It is Reviewer 1's comments 2 and 15 and it is declared rather
  than answered.
- **53 runs with no utilisation coverage.** `scripts/sample_gpu_utilisation.sh`
  was started on 31 August at 13:43 UTC, after the campaign had begun, so the
  1 Hz record covers 157 of 210 runs. `19_gpu_utilisation.py` says which, and no
  per-stack figure is averaged over an unstated subset -- but a full-coverage
  record would be a better number and this one is not it.
- **No harness provenance in the 210 manifests.** They record the environment,
  the CodeCarbon configuration, the precision policy, the GPU state, the
  governor and a UTC timestamp, and not the revision of the harness that wrote
  them. The claim that all 210 runs come from one source revision is therefore
  established by argument -- the four replayed runs of section 27 included --
  where it should be a field in the file. It is the same lesson as section 18:
  a clause enforced against argument rather than artefact is barely enforced.
