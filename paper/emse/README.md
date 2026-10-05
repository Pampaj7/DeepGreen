# EMSE (Springer Nature) front end

The submission to *Empirical Software Engineering* is built with Springer
Nature's own LaTeX template, `sn-jnl`, in the basic author–year reference
style.

## Template provenance

| File | Origin | SHA-256 |
|---|---|---|
| `sn-jnl.cls` | `sn-article-template/sn-jnl.cls` | `36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc` |
| `sn-basic.bst` | `sn-article-template/bst/sn-basic.bst` (`[2024/07/19 v1.1]`) | `4b368414cc5593169907933b417aacfdb0ce905866a39bdf55d21aad65e9d46c` |

* Source: Springer Nature LaTeX author support page,
  <https://www.springernature.com/gp/authors/campaigns/latex-author-support>,
  link "Download the Springer Nature LaTeX template", which resolves to
  <https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12>
  (zip SHA-256 `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`).
* Version: template **3.1, December 2024** (the header of the bundle's
  `sn-article.tex`). The class identifies itself as
  `\ProvidesClass{sn-jnl}[2019/11/18 v0.1]`; that string has not been updated
  across template releases, so the bundle version above is the one to cite.
* Fetched 2026-10-04. The files are copied unmodified (the class keeps its
  original CR line endings). The template is not on CTAN.
* Reference style: class option `sn-basic` ("Basic Springer Nature Reference
  Style"), which loads `natbib` with `authoryear` and selects `sn-basic.bst`.
  The documentclass line is `\documentclass[pdflatex,sn-basic]{sn-jnl}`.

## Files

| File | |
|---|---|
| `main.tex` | the EMSE front end: class, title and author block, abstract, Declarations; inputs `../preamble.tex`, `../abstract.tex`, `../statements.tex`, `../body.tex` |
| `esm.tex` | **Online Resource 1**, the Electronic Supplementary Material: same class and preamble, title "Online Resource 1 — Supplementary material for: …", sections A–J; inputs `../appendix.tex` |
| `sn-jnl.cls`, `sn-basic.bst` | the template, unmodified (above) |
| `make_submission.sh` | builds `submission.zip`, the flat, single-directory source package Springer's system asks for, verifies it, and copies `esm.pdf` to `Online_Resource_1.pdf` beside it |
| `flatten_submission.py` | the helper it calls: writes the flattened `main.tex` and the package's other files into `submission/`, and compares the package's PDF text with `main.pdf` |
| `main.pdf` | the built manuscript (`paper/build.sh`) |
| `esm.pdf`, `Online_Resource_1.pdf` | the built Online Resource 1, and the copy to upload |

## The article and Online Resource 1

EMSE's length is met by keeping the article to what each research question
needs and moving the rest to Online Resource 1 (`../appendix.tex`): the
related-work table, the specification clause by clause, execution detail, the
collapsed-run and precision-policy analyses, the reported-window mechanism and
its probe, instrument detail and coverage, the saturation cell in full, the
industrial scenario, the full defect catalogue and the extended threats to
validity. The article keeps a summary of each, with its headline numbers, and a
pointer.

Pointers are references, not typed text. The body writes `\esmsec{app:collapse}`,
`\esmtab{tab:mechanism}` or `\esmfig{…}`; `main.tex` defines them as "Online
Resource 1, Sect./Table/Fig. \ref{…}" and reads the numbers from `esm.aux`
through `xr-hyper` (`\externaldocument[][nocite]{esm}`), and `esm.tex` reads
`main.aux` the same way, so the supplement's "Section 6.4" is the article's. The
two documents' labels are disjoint, so no prefix is needed. Three details in
the front ends make this work under `sn-jnl`, which loads `hyperref` before
`xr-hyper` can be: the label import is installed by hand with every field kept
unexpanded (captions in the other document use `\si{\second}` and the number
macros, which would otherwise be expanded in the preamble); `nocite` keeps the
other document's citations out, since both cite one bibliography; and
hyperref's remote-link page is reset per link, so links to the other PDF do not
warn about page 0. The Elsevier front end (`../paper.tex`) inputs the same
`appendix.tex` after `\appendix`, and there the same macros read "Appendix D.2".

This is the local build only. The submitted source does not use `xr-hyper`:
`make_submission.sh` writes each pointer into the packaged `main.tex` as the
literal number `esm.aux` records ("Online Resource~1, Sect.~D.3"), so the
package needs neither `esm.aux` nor the label-import code, and the typeset text
is the same.

The Declarations carry no funding statement, by the authors' choice.

## Building

`paper/build.sh` compiles `main.tex`, then `esm.tex`, then `main.tex` again
(each reads the other's `.aux`; neither's numbering depends on the other, so
three passes settle it), then the Elsevier fallback, `paper/paper.tex`, and
then runs `make_submission.sh`. That script refuses to run if `esm.pdf` is
older than `esm.tex` or `../appendix.tex`, or `main.pdf` older than any of its
sources, and then builds the package to Springer's LaTeX rules (all files in one
directory, no subfolders; `\includegraphics` with local names; the `.bbl` and the
`.bib` with its `.bst`; the `pdflatex` class option; special characters as TeX
code). `submission.zip` holds exactly these files, at top level:

| File | |
|---|---|
| `main.tex` | **one** file: `main.tex` with `../preamble.tex`, `../abstract.tex`, `../statements.tex`, `../body.tex` and every generated macro and table file it reads inlined, comments stripped; no `\input` remains. Pointers into Online Resource 1 are literal numbers from `esm.aux`; the `xr-hyper` block, `\externaldocument`, `\paperroot`, `\gendir` and `\graphicspath` are gone, and figures are included by local name (`\includegraphics{fig_energy_ci}`) |
| `main.bbl` | the bibliography as BibTeX builds it from the two files below |
| `bibliography.bib` | `../bibliography.bib` with its accented letters as TeX accents (`Br{\'\i}tez`, `Jo{\~a}o`); the source file keeps UTF-8 |
| `sn-basic.bst`, `sn-jnl.cls` | the template, unmodified |
| `fig_*.pdf` | the six figures the article includes, and only those |

The script then unzips the package into an empty directory and compiles it
there with tectonic, once as shipped and once without `main.bbl` (tectonic runs
BibTeX itself, so it cannot typeset from the `.bbl` alone; instead the `.bbl`
BibTeX regenerates from the shipped `.bib` must equal the shipped one, byte for
byte). It fails if a compile fails or leaves an undefined reference or
citation, a "??" or "(?)"; if `main.tex` keeps an `\input` or `\include`; if the
zip holds a subdirectory or any file not listed above, or a text file with raw
non-ASCII characters; if the page count differs from `main.pdf`'s; or if the
PDF's text (`pdftotext`, whitespace, line-end hyphenation and ligatures
normalised) differs from `main.pdf`'s at all. The stage copy is in
`submission/`. Finally it copies `esm.pdf` to `Online_Resource_1.pdf` beside the
zip; Online Resource 1 is uploaded as a PDF, not as source, and so is the cover
letter (`cover_letter.pdf`, typeset by hand from `cover_letter.tex`).

## What sn-jnl needs that cas-dc did not

Found by compiling a minimal `sn-jnl` document with every package the
manuscript loads, and handled in `main.tex`:

* **`\usepackage[numbers]{natbib}` must not be loaded.** `sn-jnl` loads
  `natbib` itself (`authoryear` under `sn-basic`); a second load with
  `numbers` is an option-clash error. That line therefore lives in
  `paper/paper.tex`, not in the shared preamble.
* **`amsmath` is required**: the class calls `\allowdisplaybreaks` at
  `\begin{document}`. `booktabs` is loaded explicitly (cas-dc loads it).
* **`\setcounter{secnumdepth}{5}`** (shared preamble, for the body's
  `\subsubsubsection`) would number `\bmhead`, which is built on
  `\paragraph`; `main.tex` resets it to 3 before `\backmatter`.
* **Tables.** `sn-jnl` sets its `table` in 8 bp but leaves `table*` at body
  size, and its text block is 372 pt against cas-dc's two-column float width.
  `main.tex` sets every table float in the class's table font with a 4 pt
  column gap (through `\@floatboxreset`, which runs after any environment
  hook). The remaining overflows were fixed at the source so both layouts fit:
  see the generators (`tab_saturation`, `tab_instrument`,
  `tab_precision_contrast`) and the shared sources: `body.tex` (the framework,
  hyperparameter and energy-to-target tables) and `appendix.tex` (the related-
  work and protocol tables and the defect catalogue, four floats, all now in
  Online Resource 1).
* `caption` warns "Unknown document class, standard defaults will be used";
  harmless (`subcaption` needs it). hyperref is loaded by the class.
