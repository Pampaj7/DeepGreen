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

## Still open

- Re-run the campaign (~57 h) and re-derive every number. *In flight since
  23:09:25 on Sunday 30 August; job 57 of 210 at 15:42 on Monday 31 August,
  3.4 jobs/h, finishing around midday on Wednesday 2 September. Four failures,
  all the one bug in section 20.*
- Replay the four JAX/VGG-16 runs that predate the dropout fix (section 20).
- Report per-stack GPU utilisation next to energy, and say why R draws half the
  power the others do (section 21).
- Give the analysis one shared "complete runs only" filter, so no aggregate can
  include a crashed run again (section 21).
- Rewrite the sections this invalidates: RQ1's mechanism, every VGG-16
  comparison, the collapse attribution, JAX's inference ranking, the omnibus *p* bound that rounds the wrong way, the within-cell
  ρ macros, the ε² saturation, the exact collapse test, the Welch-versus-rank
  discussion, the idle-subtraction sensitivity, the bimodal power state.
- Re-derive every numeric claim in `REVIEWERS_RESPONSE.md` from the generated
  macros; several overclaim against the current tree.
- Make the replication package round-trip (13,037 of ~13,230 files differ on
  restore: float truncation, manifest type coercion, list flattening, row order).
- Re-run the claim-evidence review that stopped on a rate limit.
- Unchanged from before: CRediT roles, competing-interest declaration, Zenodo
  deposit, author photographs.
