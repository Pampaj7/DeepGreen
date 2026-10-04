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
| `sn-jnl.cls`, `sn-basic.bst` | the template, unmodified (above) |
| `make_submission.sh` | builds `submission.zip`, the standalone source package, and verifies it |
| `main.pdf` | the built manuscript (`paper/build.sh`) |

`\FundingStatement` at the top of `main.tex` is the one line the authors must
supply before submission; it typesets a bold placeholder until they do.

## Building

`paper/build.sh` compiles `main.tex` (and the Elsevier fallback,
`paper/paper.tex`), then runs `make_submission.sh`, which stages
`submission/` and writes `submission.zip`: `main.tex` with its one path line
(`\paperroot`) rewritten from `../` to nothing, the shared sources, `main.bbl`,
the class and style, `generated/` and the figures the body includes. It then
unzips the package into an empty directory and compiles it there with
tectonic, and fails if that compile fails or leaves an undefined reference or
citation.

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
  `tab_precision_contrast`) and `body.tex` (Tables 1, 2, the protocol table,
  the energy-to-target table, and the defect catalogue, now four floats).
* `caption` warns "Unknown document class, standard defaults will be used";
  harmless (`subcaption` needs it). hyperref is loaded by the class.
