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

# The author photographs are not in this repository. Rather than fail, stand in
# neutral placeholders so the layout can be checked; drop the real files into
# paper/bio/ and rebuild before submitting.
missing=()
for f in leo marco enrico roberto; do
  [[ -f "paper/bio/$f.jpg" ]] || missing+=("$f")
done
if (( ${#missing[@]} )); then
  echo "=== ${#missing[@]} author photograph(s) missing: generating placeholders ==="
  mkdir -p paper/bio
  "$PY" - "${missing[@]}" <<'PY'
import sys
from PIL import Image, ImageDraw
for name in sys.argv[1:]:
    img = Image.new("RGB", (300, 400), (232, 232, 232))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 299, 399], outline=(190, 190, 190), width=3)
    d.text((18, 186), f"photo missing:\n{name}.jpg", fill=(120, 120, 120))
    img.save(f"paper/bio/{name}.jpg", quality=88)
    print(f"  placeholder paper/bio/{name}.jpg")
PY
fi

echo "=== compiling ==="
cd paper
"$TECTONIC" -X compile paper.tex --keep-intermediates --synctex 2>&1 | tail -n 25 || {
  echo
  echo "Build failed. The usual causes, in order:"
  echo "  * cas-dc.cls -- fetched from the TeX Live bundle by tectonic; a first"
  echo "    build needs network access."
  echo "  * bibliography.bib -- reconstructed here, see paper/README.md. Two"
  echo "    entries are marked VERIFY."
  echo "  * paper/generated/ -- run without --no-data to rebuild it."
  exit 1
}
echo
echo "Built paper/paper.pdf"
