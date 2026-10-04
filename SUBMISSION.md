# Submission checklist: Empirical Software Engineering (Springer)

This is a **new submission** to EMSE. An earlier version of the study was
rejected by JSS; the cover letter says so. The manuscript itself does not
mention the earlier submission.

Everything below is built by `./paper/build.sh` from the raw records. Rebuild
before uploading, and edit nothing by hand.

## Files to upload

| EMSE item | File | State |
|---|---|---|
| Manuscript PDF | `paper/emse/main.pdf` | ready, Springer `sn-jnl`, 41 pages including references and Declarations |
| LaTeX source (zip) | `paper/emse/submission.zip` | ready; compiles standalone from an empty directory (`paper/emse/make_submission.sh`) |
| Electronic Supplementary Material: **Online Resource 1** | `paper/emse/Online_Resource_1.pdf` (a copy of `paper/emse/esm.pdf`) | ready, 39 pages; upload as a separate supplementary file named "Online Resource 1" |
| Cover letter | `paper/cover_letter.md` | ready |

The article was cut from 71 to 41 pages for EMSE. What left the main text went
to Online Resource 1 (`paper/appendix.tex`): the related-work table, the full
specification and execution detail, the collapsed-run and precision-policy
analyses, the reported-window mechanism, the instrument detail and coverage, the
saturation cell in full, the industrial scenario, the complete defect catalogue
and the extended threats to validity. The article points into it as "Online
Resource 1, Sect. D.2"; those pointers are resolved references (xr-hyper against
`esm.aux`, which the zip carries), so rebuild both together with
`./paper/build.sh` and upload the `Online_Resource_1.pdf` built beside the zip.
The Elsevier fallback carries the same material as an inline appendix.

EMSE uses neither highlights nor author photographs. The earlier reviews and
the point-by-point account (`REVIEWERS_RESPONSE.md`) are not uploaded. The
cover letter offers them on request.

## Declarations section (Springer template)

`paper/emse/main.tex` sets one *Declarations* section, as `sn-jnl` expects,
from the statements shared with the Elsevier build (`paper/statements.tex`).

| Declaration | In `paper/emse/main.tex` | State |
|---|---|---|
| Funding | `\FundingStatement`, a bold placeholder | **authors: state funding or "none"** |
| Competing interests | `\CompetingInterestStatement` | **confirm** |
| Ethics approval and consent to participate | "Not applicable" | ready |
| Consent for publication | "Not applicable" | ready |
| Data availability | `\DataAvailabilityStatement` | ready; add the DOI |
| Code availability | its own heading, pointing at the replication package | ready |
| Author contributions | `\CreditStatement` | **confirm the roles** |
| Use of generative AI and LLMs | `\GenerativeAIStatement` | ready |

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

* EMSE: `main.pdf` 41 pages, `esm.pdf` (Online Resource 1) 39 pages; 0
  undefined references, citations or macros in either, and no "??" in the
  typeset text (`make_submission.sh` checks the standalone compile for both).
* Elsevier fallback: `paper.pdf` 44 pages in `cas-dc`, with the supplementary
  material as an inline appendix; 0 undefined references, citations or macros.
* The structured abstract (Context / Objective / Method / Results /
  Conclusions) is 246 words as typeset, within EMSE's 150–250.
* 1 overfull hbox in the `cas-dc` e-mail block (the class's own box), in the
  Elsevier build only; the EMSE article has none.
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
