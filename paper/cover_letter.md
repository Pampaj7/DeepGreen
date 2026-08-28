# Cover letter

**Manuscript:** *Deep Green AI: Energy Efficiency of Deep Learning across
Language–Framework Ecosystems*

**Previous submission:** JSSOFTWARE-D-26-00842 (rejected, 15 August 2026). This
is a new submission of a re-executed study, declared here so the editors can
route it to the same reviewers if they wish.

Dear Editors,

We are submitting a manuscript that shares a title and a research question with
JSSOFTWARE-D-26-00842 and almost nothing else. The reviewers of that submission
identified faults that could not be answered by rewriting: the measurements
themselves were not sound. We agreed, discarded the campaign, rebuilt the
apparatus, and ran the study again. The numbers in this manuscript come from
that second campaign. None of them is carried over.

**What the reviewers found, and what we did about it.** Three faults were
decisive. Energy was reported in joules but recorded in kilowatt-hours, a factor
of 3.6 × 10⁶. Thirty epochs of one run were treated as thirty repeated
measurements, so the reported intervals described an effective sample size of
one. And the eight ecosystems were not running the same experiment: they
differed in learning rate, input shape, evaluation batching, data-loader
parallelism, and in two cases evaluated in training mode. Any one of these
invalidates a cross-ecosystem energy comparison; together they made the
submitted result an artefact of its own apparatus.

We also record two corrections we made to our own re-execution during internal
review, because both are the kind of thing this paper argues cannot be caught by
care alone. The headline energy tables were being computed from the software
estimator's total, modelled RAM term included, under captions saying hardware
counters --- the same class of error as the unit mislabelling, committed by us,
in the paper that catalogues it. And our model of the estimator's reported
duration was wrong twice before it was right: a floor, then a threshold, both
fitted to a predictor that does not govern the phenomenon and both carrying a
persuasive R². The manuscript now reports the consequence without any model.

We rebuilt accordingly. A written specification now states what "the same
experiment" means across stacks — model, optimisation, data pipeline, backend,
measurement, replication — and 63 automated checks enforce it against the source
code, so a divergence fails a check instead of quietly changing a number. Energy
is read from hardware counters (NVML's accumulated-energy register and the RAPL
package counters) with CodeCarbon running over the identical window as a second
reading, so every block carries two readings that can be held against each
other. We are careful in the manuscript about what that comparison establishes:
where both counters are exposed CodeCarbon reads the same registers, so the
agreement certifies the window and the arithmetic, not the accuracy. Each configuration runs five independent interleaved times
with distinct seeds, and every ecosystem records per-epoch test accuracy. The
result is 210 complete runs and 12,600 doubly instrumented blocks, with no
failed or partial run in the analysis.

**What that produced.** Two findings we did not expect, and would not have seen
under the original design.

The first is that the submitted paper's headline claim — that faster is not
greener — reproduces here as an artefact of the estimator rather than a property
of the ecosystems. The duration CodeCarbon reports beside each energy figure is
not the duration the energy was accumulated over: two thirds of blocks carry
seconds of tracker lifetime in which no energy was drawn, in three discrete
modes that block length predicts but does not determine. Any energy–time
analysis built on that field understates power by up to 11.2× on blocks under
half a second, and the bias falls precisely on the fastest stacks and the
inference phase. Measured
against the counters, energy and time correlate at ρ = 0.98 in both phases.

The second is a failure that only replication can see. VGG-16 converges to
exactly chance accuracy in 12 of 105 runs and never trains, in four ecosystems
independently and never for ResNet-18, consuming 10.5 % of the campaign's energy
for nothing. Excluding those runs collapses the cross-ecosystem accuracy spread
on VGG-16/CIFAR-100 from 20.2 to 1.3 percentage points — so almost all apparent
disagreement between stacks was runs that failed to train, not stacks computing
different things. A single-run design averages that in and reports it as an
ecosystem effect. This is, we think, the clearest possible vindication of
Reviewer 3's insistence on run-level repetitions.

**On the catalogue of defects.** The manuscript reports twenty-three measurement
defects that silently change a cross-ecosystem energy number. Five of them are
ours, introduced during this revision and caught by the checks we had just
built; they are marked as such in the table. We report them because a catalogue
of other people's mistakes would be worth less than one that includes the
authors', and because one of the five was produced by two changes that are each
correct in isolation and destructive together — which is the failure mode a
specification catches and review does not.

**What remains open.** One reviewer comment we cannot close: the workload does
not saturate the accelerator, and at 32 × 32 inputs it will not. The manuscript
states this as a limitation on external validity rather than working around it.
A wall-meter reference, which would relate our chip-level boundary to
whole-system energy, needed hardware we do not have; we report the chip boundary
consistently and decline to extrapolate.

**Artefacts.** The specification, the conformance checker, the shared
measurement bridge, the seven implementations, the per-block dual-instrument
records for all 210 runs, and the analysis pipeline are public. Every quantity
in the manuscript is generated from that pipeline and injected into the LaTeX
source; no number is typed by an author, which is how kilowatt-hours came to be
labelled as joules the first time. A single command rebuilds the manuscript from
the raw records.

A point-by-point response to every reviewer comment accompanies this letter.

Yours sincerely,

Leonardo Pampaloni, Marco Pagliocca, Enrico Vicario, Roberto Verdecchia
University of Florence
