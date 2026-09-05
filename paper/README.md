# Manuscript

`paper.tex` is the revised manuscript, in the Elsevier `cas-dc` class it was
originally submitted in.

## Build inputs that are not in this repository

`paper.tex` arrived here on its own, without the rest of its project. Three
things it needs have never been in the repository and must come from wherever
the submission was authored:

| Missing | Referenced as |
|---|---|
| the `cas-dc` class and `cas-common.sty` | `\documentclass[a4paper,fleqn]{cas-dc}` |
| author photographs | `\bio{bio/leo.jpg}` and three more |

**No new citation keys were introduced by the revision.** The rewrite cites 48
keys, all of which were already cited in the submitted version; three keys fell
out with removed text. `citekeys.txt` lists them.

### `bibliography.bib` was reconstructed

The `.bib` was not in this repository either. Rather than leave the manuscript
unbuildable, every one of the 48 entries was looked up — against Consensus,
alphaXiv, Scholar Gateway, dblp and the publishers' own records — and written
with a DOI or arXiv identifier wherever the source exposes one. **Nothing was
written from memory.**

If you still have the original `.bib`, prefer it: it is the authors' own and
will match whatever formatting the journal expects. Use this one as a
cross-check, or as the build input if the original is lost.

**The two provisional keys have been resolved.** Both were inherited from the
submitted version: one named an author who is not on the paper, the other named
no paper at all. Each was checked against the published record and the key
renamed to the work actually cited.

| Old key | Resolved to | How it was confirmed |
|---|---|---|
| `patterson2025optimizer` | `almog2025optimizer` — Almog, *An Analysis of Optimizer Choice on Energy Efficiency and Performance in Neural Network Training*, arXiv:2509.13516 | arXiv record read directly: sole author Almog, 360 runs over eight optimizers on MNIST/CIFAR-10/CIFAR-100, energy tracked with CodeCarbon — exactly the claim the citing sentence makes. The old key's "patterson" matched nothing |
| `llmgreen2024` | `luccioni2024power` — Luccioni, Jernite & Strubell, *Power Hungry Processing: Watts Driving the Cost of AI Deployment?*, FAccT '24, pp. 85--99, DOI 10.1145/3630106.3658542 | ACM record for pages and DOI; the conference PDF confirms the instrument ("logs produced by Code Carbon"), which is what the citing sentence asserts |

**Every entry touched in review was checked against Crossref.** Not against a
search engine: a reviewer's verification agent and my own web search disagreed
about whether one entry had a journal version, and the search engine was wrong.
Crossref is the registry, so it is the instrument. It settled six entries,
including one where both the reviewer and I had the sixth author's given name
wrong (Fichel, Benjamin --- not Benoit).

The one entry with no Crossref record, `almog2025optimizer`, is arXiv-only; its
identifier was verified against the arXiv abstract page directly.

Every other entry was matched to a record where the title, the venue and the
claim made in the citing sentence all agree.

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
| Conformance: 92 checks | `scripts/check_consistency.py` |

## Figures

| Figure | What it shows |
|---|---|
| `fig_window_floor.png` | The estimator’s reported duration carries seconds of tracker lifetime, in three discrete modes, and what that does to derived power |
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
