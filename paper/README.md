# Manuscript

`paper.tex` is the revised manuscript, in the Elsevier `cas-dc` class it was
originally submitted in.

## Build inputs that are not in this repository

`paper.tex` arrived here on its own, without the rest of its project. Three
things it needs have never been in the repository and must come from wherever
the submission was authored:

| Missing | Referenced as |
|---|---|
| `bibliography.bib` | `\bibliography{bibliography}` |
| the `cas-dc` class and `cas-common.sty` | `\documentclass[a4paper,fleqn]{cas-dc}` |
| author photographs | `\bio{bio/leo.jpg}` and three more |

**No new citation keys were introduced by the revision.** The rewrite uses 48
keys, all of which were already cited in the submitted version; three keys fell
out with removed text. So an existing `bibliography.bib` resolves the manuscript
unchanged. `citekeys.txt` lists every key the current text cites, for checking
that against a `.bib` before submitting.

## Nothing in the text is a typed number

Every quantity the manuscript quotes is generated from the analysis pipeline and
read in at build time. This is deliberate, and it is a direct response to how
the unit error in the submitted version happened: a number no author types is a
number no author can mislabel.

```bash
./results/analysis/run_all.sh            # the whole pipeline
python3 results/analysis/12_paper_numbers.py   # -> paper/generated/
python3 results/analysis/13_paper_figures.py   # -> paper/figures/
```

`generated/numbers.tex` holds one `\newcommand` per quoted value;
`generated/tab_*.tex` hold the result tables. `paper.tex` reads both. If a value
changes in the data it changes in the manuscript on the next build, and if a
value disappears the build fails rather than printing a stale figure.

## Where each claim comes from

| Claim | Produced by |
|---|---|
| Design: runs, blocks, configurations | `11_instrument_comparison.py` |
| Between-run intervals, repeatability | `09_campaign_v2.py` |
| Energy tables and spreads | `09_campaign_v2.py`, `12_paper_numbers.py` |
| Accuracy, accuracy per kJ, energy to target | `09_campaign_v2.py` |
| Two-instrument agreement, RAM share | `11_instrument_comparison.py` |
| The reported-window floor and its effect on power | `11_instrument_comparison.py` |
| Measurement coverage per ecosystem | `11_instrument_comparison.py` |
| Kruskal-Wallis, Holm, Cliff's delta, LibTorch control | `14_v2_statistics.py` |
| Energy against measured time, phase consistency | `14_v2_statistics.py` |
| Training collapses and conditional accuracy | `15_convergence.py` |
| Defect catalogue | the audit of the earlier campaign, `01`–`10` |
| Conformance: 56 checks | `scripts/check_consistency.py` |

## Figures

| Figure | What it shows |
|---|---|
| `fig_window_floor.png` | The estimator's reported duration is `max(phase, ~4 s)`, and what that does to power |
| `fig_energy_ci.png` | Training energy per ecosystem and block, with between-run intervals |
| `fig_energy_accuracy.png` | Energy spent against accuracy reached; collapsed runs marked |
| `fig_instrument.png` | Where the two instruments agree, and the part no counter can confirm |
| `fig_repeatability.png` | Between-run coefficient of variation, by ecosystem and phase |
| `fig_convergence.png` | What the collapsed VGG-16 runs were hiding |

## History

An intermediate draft under `main.tex` and `sections/` framed this work as a
standalone instrument-and-implementation audit, written before the replicated
campaign had run. Its material is now part of `paper.tex` — the defect catalogue
as Section "A catalogue of defect classes", the specification as part of the
methodology — and the draft was removed rather than left alongside the
manuscript it was folded into. It remains in the git history.

`reviewer.tex` holds the editorial correspondence. It is kept locally and
excluded from the repository: it is a private communication and names a
co-author's address.
