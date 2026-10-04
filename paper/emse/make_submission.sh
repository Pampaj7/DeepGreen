#!/usr/bin/env bash
# Build the source package Springer Nature's submission system asks for: one
# zip that compiles standalone, with no path leaving it.
#
#   paper/emse/make_submission.sh          # -> paper/emse/submission.zip
#
# Online Resource 1 (esm.pdf, built from esm.tex) is uploaded separately as
# Electronic Supplementary Material, not inside the zip; this script copies it
# to paper/emse/Online_Resource_1.pdf, beside submission.zip.
#
# Layout inside the zip (and in paper/emse/submission/, the staging copy):
#   main.tex                  emse/main.tex with \paperroot rewritten to {}
#   preamble.tex abstract.tex statements.tex body.tex bibliography.bib
#   main.bbl                  the bibliography as built, so the source compiles
#                             even where the system does not run BibTeX
#   esm.aux                   Online Resource 1's labels as built, which main.tex
#                             reads through xr-hyper so that its "Online
#                             Resource 1, Sect. D.2" pointers resolve
#   sn-jnl.cls sn-basic.bst   the template, unmodified
#   generated/                every macro and table file the manuscript inputs
#   figures/                  the figures the body includes, and only those
#
# The package is verified before it is written: it is unzipped into an empty
# temporary directory and compiled there with tectonic, and the build fails if
# that compile does, or leaves an undefined reference or citation.
set -euo pipefail
cd "$(dirname "$0")"
EMSE="$PWD"
PAPER="$(cd .. && pwd)"
TECTONIC="${TECTONIC:-$HOME/miniforge3/envs/dg-tectonic/bin/tectonic}"

[ -f main.bbl ] || { echo "no paper/emse/main.bbl: build main.tex first (paper/build.sh)" >&2; exit 1; }
for f in esm.aux esm.pdf; do
  [ -f "$f" ] || { echo "no paper/emse/$f: build esm.tex first (paper/build.sh)" >&2; exit 1; }
done
# A stale Online Resource would give the article pointers into a document that
# no longer has that numbering.
for src in esm.tex ../appendix.tex; do
  [ esm.pdf -nt "$src" ] || { echo "paper/emse/esm.pdf is older than $src: rebuild (paper/build.sh)" >&2; exit 1; }
done

STAGE="$EMSE/submission"
rm -rf "$STAGE" && mkdir -p "$STAGE/generated" "$STAGE/figures"

# The one path rewrite: \paperroot from ../ to nothing. Refuse if the line is
# not there to rewrite, rather than shipping a main.tex that reaches outside.
grep -q '^\\newcommand{\\paperroot}{../}$' main.tex \
  || { echo "main.tex: \\paperroot line not found" >&2; exit 1; }
sed 's#^\\newcommand{\\paperroot}{../}$#\\newcommand{\\paperroot}{}#' main.tex > "$STAGE/main.tex"

cp main.bbl esm.aux sn-jnl.cls sn-basic.bst "$STAGE/"
for f in preamble.tex abstract.tex statements.tex body.tex bibliography.bib; do
  cp "$PAPER/$f" "$STAGE/"
done
cp "$PAPER"/generated/*.tex "$STAGE/generated/"
# Only the figures the body includes.
grep -o 'includegraphics\(\[[^]]*\]\)\?{figures/[^}]*}' "$PAPER/body.tex" \
  | sed 's/.*{figures\/\(.*\)}/\1/' | sort -u \
  | while read -r fig; do cp "$PAPER/figures/$fig" "$STAGE/figures/"; done

rm -f "$EMSE/submission.zip"
(cd "$STAGE" && zip -q -X -r "$EMSE/submission.zip" .)

# Verify: a clean directory, nothing but the zip's contents.
CHECK="$(mktemp -d)"
trap 'rm -rf "$CHECK"' EXIT
unzip -q "$EMSE/submission.zip" -d "$CHECK"
(cd "$CHECK" && "$TECTONIC" -X compile main.tex --keep-logs > build.out 2>&1) || {
  echo "submission.zip does not compile standalone:" >&2
  tail -n 20 "$CHECK/build.out" >&2
  exit 1
}
if grep -qiE "undefined (references|citations)|Citation .* undefined|Reference .* undefined" "$CHECK/main.log"; then
  echo "submission.zip compiles with undefined references or citations" >&2
  exit 1
fi
# Belt and braces: a pointer into Online Resource 1 that did not resolve is
# typeset as "??"; look for it in the text as well as in the log.
if command -v pdftotext >/dev/null && pdftotext "$CHECK/main.pdf" - | grep -q '??'; then
  echo "submission.zip compiles with an unresolved reference (??) in main.pdf" >&2
  exit 1
fi
pages=$(pdfinfo "$CHECK/main.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
echo "  paper/emse/submission.zip: $(unzip -l "$EMSE/submission.zip" | tail -1 | awk '{print $2}') files," \
     "compiles standalone (${pages:-?} pages)"

# Online Resource 1, beside the zip, under the name the upload form uses.
cp esm.pdf "$EMSE/Online_Resource_1.pdf"
esm_pages=$(pdfinfo esm.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
echo "  paper/emse/Online_Resource_1.pdf: Online Resource 1 (${esm_pages:-?} pages), upload as ESM"
