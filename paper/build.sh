#!/usr/bin/env bash
# Regenerate every number and figure, then build the manuscript.
#
#   ./paper/build.sh            # analysis + compile
#   ./paper/build.sh --no-data  # compile only, reusing paper/generated/
#
# The manuscript reads its numbers from paper/generated/, written by the
# analysis pipeline, so a build after new measurements picks them up and a
# build after a deleted table fails loudly instead of printing a stale figure.
set -euo pipefail
cd "$(dirname "$0")/.."

TECTONIC="${TECTONIC:-$HOME/miniforge3/envs/dg-tectonic/bin/tectonic}"
PY="${PYTHON:-./.venv-deepgreen/bin/python}"

if [[ "${1:-}" != "--no-data" ]]; then
  echo "=== regenerating tables, numbers and figures ==="
  PYTHON="$PY" ./results/analysis/run_all.sh
fi

# The author photographs are not in this repository and are not needed: the
# manuscript wraps \bio in \IfFileExists, so a missing photograph typesets the
# biography without one. Drop the real files into paper/bio/ before sending the
# final files to Elsevier and they appear on the next build.

# The cas-dc class draws small icons next to \ead and \ead[url] from
# thumbnails/, which ship with Elsevier's template bundle rather than with the
# class on CTAN. Generate stand-ins so the paper builds from this repository
# alone; replace them from the official bundle before submitting if the icons
# matter to you.
mkdir -p paper/thumbnails
"$PY" - <<'PY'
from PIL import Image, ImageDraw
import pathlib
out = pathlib.Path("paper/thumbnails")
icons = {
    "cas-email": [(2, 4, 22, 18), "envelope"],
    "cas-url": [(2, 4, 22, 18), "globe"],
    "cas-facebook": [(2, 2, 22, 22), "box"],
    "cas-twitter": [(2, 2, 22, 22), "box"],
    "cas-gplus": [(2, 2, 22, 22), "box"],
    "cas-instagram": [(2, 2, 22, 22), "box"],
    "cas-linkedin": [(2, 2, 22, 22), "box"],
    "cas-orcid": [(2, 2, 22, 22), "circle"],
    "cas-mendeley": [(2, 2, 22, 22), "box"],
}
for name, (bbox, kind) in icons.items():
    path = out / f"{name}.jpeg"
    if path.exists():
        continue
    img = Image.new("RGB", (24, 24), "white")
    d = ImageDraw.Draw(img)
    if kind == "envelope":
        d.rectangle(bbox, outline=(70, 70, 70), width=2)
        d.line([bbox[0], bbox[1], (bbox[0] + bbox[2]) // 2, bbox[3] - 4], fill=(70, 70, 70), width=2)
        d.line([(bbox[0] + bbox[2]) // 2, bbox[3] - 4, bbox[2], bbox[1]], fill=(70, 70, 70), width=2)
    elif kind == "circle":
        d.ellipse(bbox, outline=(70, 70, 70), width=2)
    elif kind == "globe":
        d.ellipse(bbox, outline=(70, 70, 70), width=2)
        d.line([bbox[0], (bbox[1] + bbox[3]) // 2, bbox[2], (bbox[1] + bbox[3]) // 2], fill=(70, 70, 70), width=1)
    else:
        d.rectangle(bbox, outline=(70, 70, 70), width=2)
    img.save(path, quality=92)
PY

# Two front ends over one body (paper/body.tex, appendix.tex, preamble.tex,
# abstract.tex, statements.tex): paper/emse/main.tex for Empirical Software Engineering
# (Springer Nature sn-jnl) -- the submission -- and paper/paper.tex for Elsevier
# (cas-dc), kept building as the fallback.
compile() {  # <directory> <file.tex> <what failed, for the message>
  (cd "$1" && "$TECTONIC" -X compile "$2" --keep-intermediates --synctex 2>&1 | tail -n 25) || {
    echo
    echo "Build of $1/$2 failed. The usual causes, in order:"
    echo "  * $3 -- fetched or vendored, see the README beside it; a first"
    echo "    build needs network access for tectonic's bundle."
    echo "  * bibliography.bib -- reconstructed here, see paper/README.md."
    echo "  * paper/generated/ -- run without --no-data to rebuild it."
    exit 1
  }
}

# The EMSE article and its Online Resource 1 (emse/esm.tex, the content of
# appendix.tex) refer to each other through xr-hyper, each reading the other's
# .aux: main first (esm reads its section numbers), then esm, then main again
# so that its "Online Resource 1, Sect. D.2" pointers resolve. Neither
# document's numbering depends on the other, so three passes settle it.
echo "=== compiling: EMSE (Springer Nature) ==="
compile paper/emse main.tex "sn-jnl.cls (vendored in paper/emse/)"
echo "=== compiling: EMSE Online Resource 1 ==="
compile paper/emse esm.tex "sn-jnl.cls (vendored in paper/emse/)"
echo "=== compiling: EMSE (Springer Nature), against Online Resource 1 ==="
compile paper/emse main.tex "sn-jnl.cls (vendored in paper/emse/)"
echo "=== compiling: Elsevier fallback (cas-dc) ==="
compile paper paper.tex "cas-dc.cls (TeX Live bundle)"

# The flat source package EMSE's system wants, compiled once more from an empty
# directory before it is written (paper/emse/make_submission.sh).
echo "=== EMSE source package ==="
paper/emse/make_submission.sh

# Elsevier collects highlights through the submission form as a separate file,
# not from the PDF. Expand the manuscript's own environment so the uploaded text
# cannot drift from the typeset one.
"$PY" scripts/emit_highlights.py

echo
echo "Built paper/emse/main.pdf (EMSE submission), paper/emse/esm.pdf"
echo "(Online Resource 1, also copied to paper/emse/Online_Resource_1.pdf),"
echo "paper/emse/submission.zip and paper/paper.pdf (Elsevier fallback, with"
echo "the supplementary material as an inline appendix)"
