#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expand the manuscript's highlights environment into plain text.

Elsevier collects highlights through the submission form as a separate file,
not from the PDF. Typing them there by hand would put five sentences beyond the
reach of the generated-numbers discipline that covers everything else in this
paper, so expand the manuscript's own environment instead, and check the
85-character limit while we are here.

  python3 scripts/emit_highlights.py   ->  paper/highlights.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIMIT = 85  # Elsevier: at most 85 characters per bullet, spaces included.

REPLACEMENTS = [
    (r"$\times$", "\u00d7"),
    (r"\,", ""),
    (r"\%", "%"),
    ("``", "\u201c"),
    ("''", "\u201d"),
    ("--", "\u2013"),
]


def main() -> int:
    numbers = dict(re.findall(
        r"\\newcommand\{\\(\w+)\}\{(.*?)\}\n",
        (REPO_ROOT / "paper" / "generated" / "numbers.tex").read_text()))
    tex = (REPO_ROOT / "paper" / "paper.tex").read_text()
    block = tex[tex.index(r"\begin{highlights}"):tex.index(r"\end{highlights}")]

    bullets, over = [], []
    for line in block.split("\n"):
        if not line.startswith(r"\item"):
            continue
        text = line[len(r"\item "):]
        text = re.sub(r"\\v(\w+?)\{\}|\\v(\w+)",
                      lambda m: numbers.get("v" + (m.group(1) or m.group(2)), "?"),
                      text)
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        text = re.sub(r"\\[a-zA-Z]+", "", text).strip()
        bullets.append(text)
        if len(text) > LIMIT:
            over.append((len(text), text))

    out = REPO_ROOT / "paper" / "highlights.txt"
    out.write_text("\n".join(bullets) + "\n")
    print(f"  wrote paper/highlights.txt ({len(bullets)} bullets)")
    for length, text in over:
        print(f"  !! {length} characters, Elsevier allows {LIMIT}: {text}",
              file=sys.stderr)
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
