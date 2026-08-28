# Submission checklist

Everything below is built by `./paper/build.sh` from the raw records. Rebuild
before uploading; nothing here should be edited by hand.

## Files to upload

| Elsevier item | File | State |
|---|---|---|
| Manuscript | `paper/paper.pdf` (27 pp., `cas-dc`) | ready |
| LaTeX source | `paper/paper.tex`, `paper/bibliography.bib`, `paper/generated/`, `paper/figures/` | ready |
| Highlights | `paper/highlights.txt` (5 bullets, all ≤ 85 characters) | ready |
| Cover letter | `paper/cover_letter.md` | ready |
| Response to reviewers | `REVIEWERS_RESPONSE.md` | ready |
| Declaration of competing interest | in the manuscript | **confirm** |
| CRediT statement | in the manuscript | **confirm** |
| Data availability | in the manuscript | ready |
| Author biographies | in the manuscript | ready |
| Author photographs | `paper/bio/{leo,marco,enrico,roberto}.jpg` | **missing, optional** |

## Four things only you can do

1. **Confirm the CRediT roles.** They are a claim about who did what, drafted
   from the evident division of labour and not from your say-so. Section
   *CRediT Authorship Contribution Statement* in `paper/paper.tex`.
2. **Confirm the competing-interest declaration.** It currently states the
   standard "no known competing financial interests or personal relationships".
   If any author has something to declare, it belongs there.
3. **Deposit a new replication package and cite its DOI.** The footnote in
   Section 1 now points at the GitHub repository, which is current, and marks
   <https://zenodo.org/records/17734884> as superseded — that record is the
   *first* campaign and contains none of the specification, the conformance
   checker, the shared bridge, or the 210 runs reported here. A journal wants
   an archived DOI rather than a repository URL, so deposit the repository plus
   `results/replication/` and swap the DOI into that footnote.
4. **Add the author photographs**, if you want them. Drop the four `.jpg` files
   into `paper/bio/` and rebuild; the manuscript typesets the biographies
   without photographs when the files are absent, so this is optional and
   Elsevier collects them with the final files anyway.

## What the manuscript claims, and where each claim comes from

Every quantity is a macro from `paper/generated/numbers.tex`, written by
`results/analysis/12_paper_numbers.py`. There are 213 of them. No number in the
text is typed by an author — including the count of conformance checks and the
size of the defect catalogue, both of which had already drifted (57 quoted against 63 run; five defects claimed as ours against four marked)
before they were made generated. A three-reviewer pass found more of the same
kind, all now generated: the chi-square p-value, the count and scope of the
runs executed outside the main window, the size of the defect catalogue, and
the number of runs that were interleaved.

Verify the whole chain with:

```bash
python3 scripts/check_consistency.py            # 67 pass, 0 fail
python3 scripts/consolidate_raw.py --check      # package matches the raw tree
./paper/build.sh                                # analysis + numbers + figures + PDF
```

## Known state of the manuscript build

* 0 undefined references, 0 undefined citations, 0 oversized floats.
* 1 overfull hbox, 123.6 pt, in the `cas-dc` front-matter e-mail block. It is
  the class's own box and does not print into the margin.
* 54 bibliography entries, all with a DOI or arXiv identifier where the source
  exposes one. The two provisional keys inherited from the submitted version
  (`patterson2025optimizer`, `llmgreen2024`) are resolved and renamed, and four
  miscitations found in review are corrected: CodeCarbon was cited to the paper
  introducing a different tool, Tiny ImageNet to a benchmark that does not use
  it, a language-energy claim to a study measuring no energy, and an electricity
  price to a paper containing none. See `paper/README.md`.
* The structured abstract runs to about 400 words. Elsevier's guidance is ~250
  for an unstructured one; JSS accepts structured abstracts, which run longer.
  Trim if the editor asks.

## One reviewer comment stays open

The workload does not saturate the accelerator, and at 32 × 32 inputs it will
not. This is stated in the manuscript as a threat to external validity rather
than worked around. Closing it means a different experiment, not a different
analysis.
