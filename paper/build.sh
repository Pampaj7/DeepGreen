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

echo "=== compiling ==="
cd paper
"$TECTONIC" -X compile paper.tex --keep-intermediates --synctex 2>&1 | tail -n 25 || {
  echo
  echo "Build failed. The usual causes, in order:"
  echo "  * cas-dc.cls -- fetched from the TeX Live bundle by tectonic; a first"
  echo "    build needs network access."
  echo "  * bibliography.bib -- reconstructed here, see paper/README.md."
  echo "  * paper/generated/ -- run without --no-data to rebuild it."
  exit 1
}
cd ..
# Elsevier collects highlights through the submission form as a separate file,
# not from the PDF. Expand the manuscript's own environment so the uploaded text
# cannot drift from the typeset one.
"$PY" scripts/emit_highlights.py

echo
echo "Built paper/paper.pdf"
