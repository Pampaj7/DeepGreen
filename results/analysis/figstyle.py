#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One visual identity for every figure of the manuscript.

Each figure used to carry its own colours: some coloured the ecosystems, some
used two blues for whatever the panel's two series were, and the per-stack
colours paired a green JAX with a red Java that a deuteranope cannot tell
apart. This module is the single place that says what a stack looks like, and
the figure scripts import it rather than restate it.

**The stacks.**  Order is ``common.ECOSYSTEM_ORDER``, the pipeline's canonical
order and the one ``tab_saturation`` prints its rows in, held in every figure
so a stack sits in the same row wherever it appears. Each stack has a
colour *and* a marker, and the marker is not decoration: seven hues cannot all
be told apart pairwise under colour-vision deficiency, so wherever two stacks
can sit side by side without a label (a scatter), shape carries the identity
that hue alone cannot.

The hues are the eight-hue categorical palette of the dataviz method this
redesign followed, less its orange, assigned by enumeration rather than by eye
(``node scripts/validate_palette.js`` from that method, light mode, white
surface):

* adjacent pairs in that order: worst CVD delta-E 16.2, worst normal-vision
  delta-E 19.6 -- every gate passes;
* the four LibTorch-lineage stacks (Rust, C++, PyTorch, R), which Section 6.6
  compares among themselves, pass the all-pairs gates on their own (CVD 13.0,
  normal 15.6);
* over all 21 pairs of seven, three sit in the 6-8 CVD band (Rust-TensorFlow
  6.1, Rust-Java 6.9, PyTorch-Java 7.2) and one under the normal-vision floor
  (TensorFlow-Java 13.2). Seven series are past the all-pairs cap of any
  categorical palette; those four pairs get maximally different markers, and
  the key stacks of the paper's argument (C++, R, Java) get the three
  high-contrast hues. Aqua, yellow and magenta sit under 3:1 against white,
  which is why every stack is also named on an axis or in a legend and every
  value is in a table of the manuscript.

**Everything that is not a stack is grey** (or, where a panel needs a second
non-stack series, the palette's unused orange), so a colour in this paper
always means an ecosystem.

**Size.**  Figures are drawn at the size they are printed. ``TEXT_WIDTH_IN`` is
sn-jnl's ``\\textwidth`` (372 pt, probed by compiling paper/emse/main.tex), the
width every ``figure*`` is set at in the EMSE build; ``COLUMN_WIDTH_IN`` is
cas-dc's ``\\columnwidth`` (238.25 pt), the narrower of the two places a plain
``figure`` lands. Type is 8 pt at that size, in a Helvetica-metric sans
(Springer's artwork guidelines ask for Helvetica or Arial lettering). Output is
vector PDF with the fonts embedded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from common import ECOSYSTEM_ORDER  # noqa: E402

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
PT_PER_IN = 72.27
TEXT_WIDTH_IN = 372.0 / PT_PER_IN       # sn-jnl \textwidth  (13.1 cm)
COLUMN_WIDTH_IN = 238.25 / PT_PER_IN    # cas-dc \columnwidth (8.4 cm)

# --------------------------------------------------------------------------
# Ink (light chart surface; print is always light)
# --------------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text, dark non-stack marks
MUTED = "#898781"        # axis lines, ticks, reference lines
GRID = "#e1e0d9"         # hairline grid
LIGHT = "#c3c2b7"        # light non-stack marks
BAND = "#f0efec"         # a row or region set apart (the diverging midpoint grey)
ACCENT = "#eb6834"       # a second non-stack series; no stack wears it

# --------------------------------------------------------------------------
# The stacks
# --------------------------------------------------------------------------
SHORT = {
    "Python/PyTorch": "PyTorch", "Python/TensorFlow": "TensorFlow",
    "Python/JAX": "JAX", "Cpp/LibTorch": "C++", "C++/LibTorch": "C++",
    "Java/DL4J": "Java", "R/torch": "R", "Rust/tch": "Rust",
}

#: Table order, in short names (MATLAB is in ECOSYSTEM_ORDER but not measured).
STACKS = [SHORT[e] for e in ECOSYSTEM_ORDER if e in SHORT]

#: stack -> (colour, marker). See the module docstring for how these were chosen.
STACK_STYLE = {
    "Rust":       ("#1baf7a", "D"),   # aqua, diamond
    "C++":        ("#2a78d6", "o"),   # blue, circle
    "PyTorch":    ("#008300", "s"),   # green, square
    "JAX":        ("#eda100", "v"),   # yellow, triangle down
    "TensorFlow": ("#e87ba4", "^"),   # magenta, triangle up
    "R":          ("#4a3aa7", "P"),   # violet, filled plus
    "Java":       ("#e34948", "X"),   # red, filled cross
}
assert list(STACK_STYLE) == STACKS, "STACK_STYLE must follow ECOSYSTEM_ORDER"
COLOR = {k: v[0] for k, v in STACK_STYLE.items()}
MARKER = {k: v[1] for k, v in STACK_STYLE.items()}

# Markers differ in optical size at one nominal size; these even them out.
MARKER_SCALE = {"o": 1.0, "s": 0.88, "D": 0.78, "v": 1.05, "^": 1.05,
                "P": 1.0, "X": 1.0}

LIBTORCH_LINEAGE = ["Rust", "C++", "PyTorch", "R"]

DATASET = {"fashionmnist": "Fashion-MNIST", "cifar100": "CIFAR-100",
           "tinyimagenet": "Tiny ImageNet"}
DATASETS = ["fashionmnist", "cifar100", "tinyimagenet"]
MODEL = {"resnet18": "ResNet-18", "vgg16": "VGG-16"}
MODELS = ["resnet18", "vgg16"]

FONT_PT = 8.0
SMALL_PT = 7.0


def apply() -> None:
    """Set matplotlib's defaults for print at the final size."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans",
                            "DejaVu Sans"],
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": SMALL_PT + 0.5,
        "ytick.labelsize": SMALL_PT + 0.5,
        "legend.fontsize": SMALL_PT + 0.5,
        "mathtext.default": "regular",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.handletextpad": 0.3,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,
        "legend.handlelength": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 600,          # only rasterised layers use it
        "pdf.fonttype": 42,          # embed TrueType, keep text as text
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.04,
        "figure.constrained_layout.wspace": 0.04,
    })


def figure(width_in: float, height_in: float, **kw):
    return plt.figure(figsize=(width_in, height_in), **kw)


def ms(stack: str, size: float = 4.5) -> float:
    """Marker size (points) for ``stack``, optically evened."""
    return size * MARKER_SCALE[MARKER[stack]]


def stack_handle(stack: str, size: float = 4.5, filled: bool = True) -> Line2D:
    """An opaque legend handle for ``stack``, whatever alpha the data used."""
    c = COLOR[stack]
    return Line2D([], [], ls="none", marker=MARKER[stack], ms=ms(stack, size),
                  markerfacecolor=c if filled else SURFACE,
                  markeredgecolor=c, markeredgewidth=0.8 if not filled else 0.0,
                  label=stack)


def panel_label(ax, letter: str, x: float | None = None) -> None:
    """``(a)`` at the top left of the axes, outside the plot area."""
    ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=FONT_PT,
                 x=x if x is not None else 0.0, ha="left")


def log_ticks(axis, values, fmt=None) -> None:
    """Fixed, labelled ticks on a log axis, no minor labels.

    Matplotlib labels only decades by default, and a panel that spans less than
    one carries no labelled tick at all.
    """
    axis.set_major_locator(FixedLocator(list(values)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(fmt or compact))


def compact(v, _pos=None) -> str:
    """1, 2, 5, 10, 20 ... 1k, 2k, 10k, 1M; trailing zeros dropped."""
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v / div:g}{suffix}"
    return f"{v:g}"


def times(v, _pos=None) -> str:
    return f"{v:g}×"


def save(fig, path: Path, max_mb: float = 2.0) -> Path:
    """Write ``fig`` as PDF at its own size, and remove a stale PNG of the stem.

    No ``bbox_inches='tight'``: that crops to the ink and changes the width, and
    the manuscript then rescales the figure and its type. Constrained layout
    keeps everything inside the canvas instead. The creation date is left out
    so an unchanged figure regenerates byte-identically.
    """
    path = path.with_suffix(".pdf")
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)
    size_mb = path.stat().st_size / 1e6
    if size_mb > max_mb:
        print(f"  !! {path.name} is {size_mb:.1f} MB; rasterise its densest layer")
    stale = path.with_suffix(".png")
    if stale.exists():
        stale.unlink()
    return path
