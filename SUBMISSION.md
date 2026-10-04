# Submission checklist: Empirical Software Engineering (Springer)

This is a **new submission** to EMSE. An earlier version of the study was
rejected by JSS; the cover letter says so. The manuscript itself does not
mention the earlier submission.

Everything below is built by `./paper/build.sh` from the raw records. Rebuild
before uploading, and edit nothing by hand.

## Files to upload

| EMSE item | File | State |
|---|---|---|
| Manuscript PDF | `paper/paper.pdf` | ready in `cas-dc`; **port to Springer `sn-jnl` pending** |
| LaTeX source (zip) | `paper/paper.tex`, `paper/bibliography.bib`, `paper/generated/`, `paper/figures/` | zip after the port |
| Cover letter | `paper/cover_letter.md` | ready |

EMSE uses neither highlights nor author photographs. The earlier reviews and
the point-by-point account (`REVIEWERS_RESPONSE.md`) are not uploaded. The
cover letter offers them on request.

## Declarations section (Springer template)

The `sn-jnl` template expects one *Declarations* section. Its content exists in
the manuscript under Elsevier headings; the port should move it as follows.

| Declaration | Source in `paper.tex` now | State |
|---|---|---|
| Funding | none stated; a commented-out LEAP note sits in the front matter | **authors: state funding or "none"** |
| Competing interests | *Declaration of Competing Interest* | **confirm** |
| Ethics approval | none; no human participants | write "Not applicable" |
| Consent to participate / publish | none | write "Not applicable" |
| Data availability | *Data Availability* | ready; add the DOI |
| Code availability | inside *Data Availability* | split out at the port |
| Author contributions | *CRediT Authorship Contribution Statement* | **confirm the roles** |
| Use of LLMs | *Declaration of Generative AI…* | ready |

## What remains for the authors

1. **Archive and cite a DOI before acceptance.** Section 1's footnote points at
   GitHub and marks <https://zenodo.org/records/17734884> as superseded, because
   that record holds an earlier campaign. Deposit the repository with
   `results/replication/` and `results/replication_saturation/`, and put the
   DOI into the footnote and the data-availability statement.
2. **Confirm the author contributions, the competing-interest statement and the
   funding statement** above.

## Verify the chain

```bash
python3 scripts/check_consistency.py            # 110 pass, 0 fail
python3 scripts/consolidate_raw.py --check      # package matches the raw tree
./paper/build.sh                                # analysis + numbers + figures + PDF
```

Every quantity in the text is a macro from `paper/generated/numbers.tex` (328)
or `paper/generated/numbers_saturation.tex` (327), written by the analysis
pipeline. No author types a number.

## Known state of the build

* 39 pages in `cas-dc`; 0 undefined references, citations or macros.
* The structured abstract (Context / Objective / Method / Results /
  Conclusions) is 246 words as typeset, within EMSE's 150–250.
* 1 overfull hbox in the `cas-dc` e-mail block (the class's own box). It goes
  away with the port.
* 57 bibliography entries, each with a DOI or arXiv identifier where the source
  has one; see `paper/README.md`.

## Fallback: Elsevier (IST or similar)

If the paper goes to an Elsevier journal instead, the `cas-dc` build is the
manuscript as it stands.

* `paper/highlights.txt` holds 5 bullets of at most 85 characters each. It is
  generated from the manuscript's `highlights` environment by
  `scripts/emit_highlights.py`.
* The declarations keep their Elsevier headings.
* Author photographs are optional. Drop `paper/bio/{leo,marco,enrico,roberto}.jpg`
  into place and rebuild.
