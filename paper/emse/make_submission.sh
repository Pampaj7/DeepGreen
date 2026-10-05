#!/usr/bin/env bash
# Build the source package Springer Nature's submission system asks for: one
# zip, one flat directory, that compiles standalone.
#
#   paper/emse/make_submission.sh          # -> paper/emse/submission.zip
#
# Springer's LaTeX rules (springernature.com, LaTeX author support): all files
# in a single directory with no subfolders; \includegraphics with local file
# names; the .bbl or the .bib with its .bst; the pdflatex class option; special
# characters as TeX code, not raw Unicode. The zip therefore holds exactly:
#
#   main.tex           ONE file: emse/main.tex with the shared preamble,
#                      abstract, statements, body and every generated macro and
#                      table file inlined and comments stripped -- no \input
#                      is left (flatten_submission.py). Pointers into Online
#                      Resource 1 ("Online Resource 1, Sect. D.3") are set as
#                      the literal numbers recorded in esm.aux, so the package
#                      needs neither xr-hyper nor esm.aux.
#   main.bbl           the bibliography, as BibTeX builds it from the two below
#   bibliography.bib   paper/bibliography.bib with accented letters as TeX
#                      accents ({\'a}); the source file is left as it is
#   sn-basic.bst sn-jnl.cls   the template, unmodified
#   fig_*.pdf          the figures the body includes, and only those
#
# The source tree is not touched: the two front ends, body.tex and the xr-hyper
# setup used for local builds stay as they are; only this packaged copy is
# flattened. Staged in paper/emse/submission/.
#
# The package is verified before the script succeeds. It is unzipped into an
# empty directory and compiled there with tectonic (XeTeX engine; it runs
# BibTeX itself), once as shipped and once without main.bbl, and the build
# fails if
#   * a compile fails, or leaves an undefined reference or citation, or "??";
#   * any \input or \include remains in main.tex, or a subdirectory or any
#     file outside the list above is in the zip, or a text file is not ASCII;
#   * the main.bbl BibTeX regenerates from the shipped .bib differs from the
#     shipped one;
#   * the PDF's text (pdftotext; whitespace, line-end hyphenation and ligatures
#     normalised) differs from paper/emse/main.pdf, or its page count does.
#
# Online Resource 1 (esm.pdf, built from esm.tex) is uploaded separately as
# Electronic Supplementary Material, not inside the zip; this script copies it
# to paper/emse/Online_Resource_1.pdf, beside submission.zip.
set -euo pipefail
cd "$(dirname "$0")"
EMSE="$PWD"
PAPER="$(cd .. && pwd)"
TECTONIC="${TECTONIC:-$HOME/miniforge3/envs/dg-tectonic/bin/tectonic}"
PY="${PYTHON:-python3}"
ZIP="$EMSE/submission.zip"

fail() { echo "make_submission: $*" >&2; exit 1; }

for f in esm.aux esm.pdf main.pdf; do
  [ -f "$f" ] || fail "no paper/emse/$f: build first (paper/build.sh)"
done
# A stale Online Resource would give the article pointers into a document that
# no longer has that numbering; a stale main.pdf would make the text comparison
# below meaningless.
for src in esm.tex ../appendix.tex; do
  [ esm.pdf -nt "$src" ] || fail "paper/emse/esm.pdf is older than $src: rebuild (paper/build.sh)"
done
for src in main.tex esm.aux ../preamble.tex ../abstract.tex ../statements.tex ../body.tex \
           ../bibliography.bib ../generated/*.tex ../figures/*.pdf; do
  [ main.pdf -nt "$src" ] || fail "paper/emse/main.pdf is older than $src: rebuild (paper/build.sh)"
done

compile() {  # <dir> [extra tectonic args]: compile main.tex there, or fail
  local dir="$1"; shift
  (cd "$dir" && "$TECTONIC" -X compile main.tex --keep-logs "$@" > build.out 2>&1) \
    || { tail -n 20 "$dir/build.out" >&2; fail "submission does not compile standalone ($dir)"; }
}

# --- stage ---------------------------------------------------------------
STAGE="$EMSE/submission"
rm -rf "$STAGE" && mkdir -p "$STAGE"
echo "  flattening paper/emse/main.tex:"
"$PY" "$EMSE/flatten_submission.py" stage "$STAGE"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
# main.bbl as BibTeX builds it from the shipped .bib and .bst.
mkdir "$SCRATCH/bbl" && cp "$STAGE"/* "$SCRATCH/bbl/"
compile "$SCRATCH/bbl" --keep-intermediates
cp "$SCRATCH/bbl/main.bbl" "$STAGE/"

rm -f "$ZIP"
(cd "$STAGE" && zip -q -X -D "$ZIP" ./*)

# --- verify --------------------------------------------------------------
expected="$( (printf '%s\n' main.tex main.bbl bibliography.bib sn-basic.bst sn-jnl.cls;
              grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' "$STAGE/main.tex" \
                | sed 's/.*{\(.*\)}/\1.pdf/') | sort -u)"
actual="$(unzip -Z1 "$ZIP" | sort)"
if grep -q / <<<"$actual"; then
  fail "submission.zip contains a subdirectory: $(grep / <<<"$actual" | head -3 | tr '\n' ' ')"
fi
[ "$actual" = "$expected" ] || {
  diff <(echo "$expected") <(echo "$actual") >&2
  fail "submission.zip does not hold exactly the expected files (< expected, > in zip)"
}

check() {  # <dir> <what>: the checks every compile of the package must pass
  local dir="$1" what="$2"
  if grep -qiE "undefined (references|citations)|(Citation|Reference) .* undefined|There were undefined" "$dir/main.log"; then
    grep -iE "undefined" "$dir/main.log" | head -5 >&2
    fail "$what: undefined references or citations"
  fi
  if pdftotext "$dir/main.pdf" - | grep -q '??'; then
    fail "$what: an unresolved reference (??) in the typeset text"
  fi
  if grep -q '(?)' <(pdftotext "$dir/main.pdf" -); then
    fail "$what: an unresolved citation (?) in the typeset text"
  fi
  local pages ref_pages
  pages=$(pdfinfo "$dir/main.pdf" | awk '/^Pages:/{print $2}')
  ref_pages=$(pdfinfo "$EMSE/main.pdf" | awk '/^Pages:/{print $2}')
  [ "$pages" = "$ref_pages" ] || fail "$what: $pages pages, paper/emse/main.pdf has $ref_pages"
  "$PY" "$EMSE/flatten_submission.py" compare "$EMSE/main.pdf" "$dir/main.pdf" \
    || fail "$what: typeset text differs from paper/emse/main.pdf"
  echo "  $what: compiles, $pages pages, no undefined reference or citation"
}

mkdir "$SCRATCH/zip"
unzip -q "$ZIP" -d "$SCRATCH/zip"
if find "$SCRATCH/zip" -mindepth 1 -type d | grep -q .; then
  fail "submission.zip unpacks into subdirectories"
fi
# No \input or \include survives (outside comments), and nothing raw: Springer
# wants special characters as TeX code.
if sed 's/\(^\|[^\\]\)%.*$/\1/' "$SCRATCH/zip/main.tex" \
     | grep -nE '\\(input|include|InputIfFileExists)([^A-Za-z@]|$)' >&2; then
  fail "main.tex in submission.zip still has an \\input or \\include"
fi
for f in main.tex main.bbl bibliography.bib sn-basic.bst sn-jnl.cls; do
  if LC_ALL=C grep -nP '[^\x00-\x7F]' "$SCRATCH/zip/$f" | head -3 >&2; then
    fail "$f in submission.zip has raw non-ASCII characters"
  fi
done

# 1. As shipped. (tectonic runs BibTeX whenever the .aux asks for it, so this
#    also exercises the .bib and .bst.)
compile "$SCRATCH/zip"
check "$SCRATCH/zip" "submission.zip as shipped"

# 2. Without main.bbl: BibTeX from bibliography.bib and sn-basic.bst alone, and
#    the .bbl it writes must be the shipped one, byte for byte -- so a system
#    that typesets from main.bbl without running BibTeX gets the same pages.
mkdir "$SCRATCH/nobbl"
unzip -q "$ZIP" -d "$SCRATCH/nobbl" && rm "$SCRATCH/nobbl/main.bbl"
compile "$SCRATCH/nobbl" --keep-intermediates
check "$SCRATCH/nobbl" "submission.zip without main.bbl"
cmp -s "$SCRATCH/nobbl/main.bbl" "$STAGE/main.bbl" \
  || fail "BibTeX on the shipped bibliography.bib does not reproduce the shipped main.bbl"

echo "  paper/emse/submission.zip: $(wc -l <<<"$actual") files, one flat directory:"
sed 's/^/      /' <<<"$actual"

# Online Resource 1, beside the zip, under the name the upload form uses.
cp esm.pdf "$EMSE/Online_Resource_1.pdf"
esm_pages=$(pdfinfo esm.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
echo "  paper/emse/Online_Resource_1.pdf: Online Resource 1 (${esm_pages:-?} pages), upload as ESM"
# The cover letter is uploaded as a PDF and built by hand (see SUBMISSION.md).
if [ ! -f cover_letter.pdf ] || [ cover_letter.tex -nt cover_letter.pdf ]; then
  echo "  WARNING: paper/emse/cover_letter.pdf is missing or older than cover_letter.tex" >&2
fi
