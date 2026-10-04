# Cover letter

**To:** The Editors-in-Chief, *Empirical Software Engineering*

**Manuscript:** *Deep Green AI: Energy Efficiency of Deep Learning across
Language–Framework Ecosystems*

Dear Editors-in-Chief,

We submit the above manuscript for consideration as a regular article in
*Empirical Software Engineering*.

**What the paper contributes.** It is a controlled, replicated comparison of
the energy cost of deep-learning training and inference across seven
language–framework ecosystems: Python with PyTorch, TensorFlow and JAX; C++ with
LibTorch; Java with Deeplearning4j; R with `torch`; and Rust with `tch`. The
campaign is 210 runs on one GPU host. A written specification defines what "the
same experiment" means across stacks, and 110 executable checks enforce it
against the source and the built binaries. Every measurement block is read
twice, by hardware energy counters and by a widely used software estimator.

Training energy differs across ecosystems by 7.4×–9.8×, and by 1.1×–1.6× among
the three stacks that share one exported model and backend build. A declared
contrast cell of 70 runs at 224×224 shows the spread narrowing within the
LibTorch lineage (1.3×–2.8×) and widening across all seven stacks
(15.3×–19.1×). The estimator agrees with the counters on energy to 0.3 %, but
its reported duration understates derived power by up to 13.0×. The paper also
includes a catalogue of the defect classes we found, our own among them.

**Why EMSE.** The paper is an empirical software-engineering study before it is
a machine-learning one. Its central methodological claim is that a
cross-ecosystem comparison is only as good as the specification and enforcement
behind it. It treats the measurement apparatus as an object of study, reports
between-run uncertainty from independent replications, and states its threats
to validity in the terms the field uses. It also contributes to Green software
engineering. The study is built for open
science: every number in the manuscript is generated from the raw records by a
public pipeline, and one command rebuilds the paper.

**Prior submission.** An earlier version of this study was submitted to the
*Journal of Systems and Software* and rejected. We took the reviews as a reason
to redo the study rather than to revise the text:

- a new measurement campaign on new hardware;
- a written experiment specification with an automated conformance checker;
- dual-instrument measurement of every block;
- the accelerator-saturation contrast cell;
- an audit of the defects in the earlier work.

The earlier reviews and our point-by-point account of how each was addressed
are in the replication repository (`REVIEWERS_RESPONSE.md`), and we will supply
them directly on request.

**Data and code availability.** The implementations, the specification, the
conformance checker, the measurement bridge, the raw per-block records of the
campaign and of the contrast cell, and the analysis pipeline are public at
<https://github.com/Pampaj7/DeepGreen>. We will deposit an archived copy with a
DOI before publication.

**Declarations.** The manuscript is not under consideration elsewhere and has
not been published. All authors have read and approved the submission. We
declare no competing interests.

Yours sincerely,

Leonardo Pampaloni and Marco Pagliocca (corresponding authors), Enrico Vicario,
Roberto Verdecchia
University of Florence, Italy
