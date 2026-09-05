#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel-level confirmation of the TF32 x determinism precision ablation.

The number the manuscript cites for this effect is *not* in this file's
output: it is the campaign's own v1-vs-v2 contrast for Python/PyTorch
ResNet-18 -- 150 training epochs per cell, real data, six other stacks moving
<=1.05% as a control -- written by the analysis pipeline to
``results/analysis/tables/v2_tf32_campaign_contrast``. That is an end-to-end,
on-instrument measurement and is the primary evidence. This script is the
mechanism check underneath it: it isolates the two cuDNN flags at the kernel
level, on synthetic data, so the campaign contrast can be attributed to
convolution algorithm selection rather than argued into it. Agreement between
the two is corroboration; disagreement would mean the campaign contrast has a
confound this probe does not share.

It measures kernels, not accuracy: the input is synthetic random noise rather
than CIFAR-100, because the cells differ only in which convolution and GEMM
algorithms cuDNN selects, and that choice does not depend on what the pixels
are. What has to match the campaign is the *shape* of the work -- the
network, the resolution, the batch size, the optimiser -- not the data.

Model: ``torchvision.models.resnet18(weights=None)`` with ``model.fc``
replaced by ``nn.Linear(in_features, 100)``, the definition in
``scripts/export_torchscript_models.py:build()`` that every stack's
ResNet-18 module is exported from -- this script imports that ``build()``
rather than duplicating it. **Divergence from the campaign, stated plainly:**
the campaign's Python/PyTorch stack does not run this eager module -- it
calls ``tools.deepgreen_bench.load_shared_module``, which is
``torch.jit.load`` on the exported artefact, and TorchScript's executor
fuses ops the eager graph does not. This probe therefore measures cuDNN
algorithm selection under the eager path; it does not measure whatever
TorchScript is separately worth. 100 classes and 32x32 input are CIFAR-100's,
the dataset every other campaign dataset is also pre-resized to. Batch 128
and Adam at ``tools.deepgreen_bench.DEFAULT_LR`` (1e-4) match the harness;
precision is fp32.

Six cells, ``(matmul_allow_tf32, cudnn_allow_tf32, deterministic,
cudnn_enabled)``. The first four hold ``matmul.allow_tf32`` at ``False`` --
the pre-fix (v1) harness's setting, and every binding's framework default --
and vary only the two cuDNN flags, which is the confound REVISION_LOG.md
section 1 traces the v1 PyTorch-vs-C++ gap to. The fifth, ``(True, True,
False, True)``, is the re-executed (v2) campaign's actual configuration:
``DEEPGREEN_TF32=1`` ties ``matmul.allow_tf32`` to ``cudnn.allow_tf32`` in
``tools.deepgreen_bench.set_precision_policy``, and ``NVIDIA_TF32_OVERRIDE=1``
does the same for the four stacks that never import it. The sixth sets
``torch.backends.cudnn.enabled = False`` on top of the v2 configuration --
its ``cudnn_allow_tf32``/``deterministic`` values are moot and are recorded
as the v2-default values so the row is unambiguous -- and bounds what
Deeplearning4j's absent cuDNN path (REVISION_LOG.md section 19) can cost: an
upper bound on losing cuDNN, not an estimate of nd4j's own im2col path, since
that is not PyTorch's non-cuDNN fallback either. Ratios are reported against
both the first cell (the v1 baseline) and the fifth (the v2 default), so a
reader can read off any of the three contrasts directly.

Per cell: build a fresh model and optimiser, warm up on a wall-clock budget
(``WARMUP_SECONDS``, not a step count -- a fixed step count is tens of
milliseconds in the fast cells, too short for the GPU to leave its idle
P-state, which is the same cold-clock bias that made an earlier, undocumented
version of this table's baseline read low), call
``torch.cuda.synchronize()``, take a ``HardwareCounters`` snapshot, run
``--steps`` training steps (forward, backward, optimiser step) on one fixed
synthetic batch, synchronize again, and take the closing snapshot -- the same
two-snapshot pattern ``scripts/measure_idle.py`` uses. NVML's energy register
updates at roughly 10 Hz, so a window of only a few ticks is quantised; each
cell's window is checked against ``MIN_WINDOW_S`` and a short window is
reported, not silently accepted, and ``window_s`` is written into the table
so a reader can check the margin themselves rather than take it on trust.
Each cell is repeated ``--repeats`` times (default 3); the cells are shuffled
independently within each repeat, using ``--seed``, so thermal drift over the
run does not alias onto one cell and the schedule is reproducible.

What this probe should agree with -- the campaign's own per-step figures,
first campaign, cifar100 train (390 steps/epoch): C++/LibTorch (TF32 on,
cuDNN default) approximately 0.0075 s and 1.9 J per step; Python/PyTorch
(true fp32, cuDNN pinned deterministic, as the pre-fix harness ran it)
approximately 0.022 s and 7.5 J per step. A kernel-only ratio should sit at
or a little above that end-to-end one, not below it. It should not agree with
the 0.352 s / 78.3 J and 1.221 s / 407.6 J figures that once appeared in
REVISION_LOG.md's table: those have no script, CSV, or commit behind them
anywhere in this repository's history, and are roughly 50x the campaign's own
per-step measurement of the same two configurations -- most likely 50-step
totals mislabelled as per-step. They are not reproduced here and should not
be cited.

Run:

    .venv-deepgreen/bin/python scripts/probe_tf32.py
    .venv-deepgreen/bin/python scripts/probe_tf32.py --steps 1000 --repeats 3 --seed 1000

The no-cuDNN cell is far slower per step (REVISION_LOG.md section 19 implies
roughly 17x the baseline), so it runs a separate, shorter step count,
``--steps-nocudnn`` (default 300), while still holding to the same >=5 s
window floor.

Writes results/revision/tables/v2_tf32_ablation.{csv,md} through
results/analysis/common.save_table (which diverts to a ``_partial`` sibling
while the campaign on disk is incomplete -- see that module).

Refuses to run if the accelerator is not idle, or if the hardware GPU energy
counter is unavailable: both make the measurement meaningless rather than
merely noisy, the same standard the campaign holds itself to.
"""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "results" / "analysis"))

from tools.deepgreen_bench import DEFAULT_BATCH_SIZE, DEFAULT_LR  # noqa: E402
from tools.hardware_counters import HardwareCounters  # noqa: E402
from common import save_table  # noqa: E402

try:
    # The exact definition every stack's ResNet-18 module is exported from.
    from export_torchscript_models import build as _campaign_resnet18  # noqa: E402
except ImportError as e:  # pragma: no cover -- repo layout guarantees this
    print(f"cannot import the campaign's model definition: {e}", file=sys.stderr)
    raise

try:
    # Same whole-machine check the campaign driver refuses to start without.
    from run_campaign import _assert_accelerator_idle  # noqa: E402
except ImportError:
    def _assert_accelerator_idle() -> None:
        """Fallback copy of run_campaign.py's check, for an isolated checkout."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            print("warning: could not query the accelerator; proceeding unchecked.",
                  file=sys.stderr)
            return
        busy = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        if busy:
            print(
                "error: the accelerator is already in use:\n  "
                + "\n  ".join(busy)
                + "\nWhole-machine energy measurement attributes every watt to the run\n"
                  "being tracked, so a second workload silently inflates it. Stop those\n"
                  "processes first.",
                file=sys.stderr,
            )
            raise SystemExit(3)

#: CIFAR-100's class count and the resolution every campaign dataset is
#: pre-resized to (scripts/run_campaign.py, scripts/normalise_dataset_resolution.py).
NUM_CLASSES = 100
INPUT_HW = (32, 32)

#: Warm up on a wall-clock budget, not a step count -- see the module docstring.
WARMUP_SECONDS = 2.0

#: NVML's energy register updates at roughly 10 Hz (REVISION_LOG.md section 9);
#: a window this short or shorter is a handful of ticks and quantisation-bound.
MIN_WINDOW_S = 5.0

#: (matmul_allow_tf32, cudnn_allow_tf32, deterministic, cudnn_enabled). The
#: first four are the v1 confound's configuration space (matmul held at its
#: framework default, False; cuDNN enabled); the fifth is the re-executed
#: campaign's actual policy, where DEEPGREEN_TF32=1 ties matmul to cudnn; the
#: sixth disables cuDNN entirely on top of the fifth's flags, bounding
#: REVISION_LOG.md section 19's Deeplearning4j gap. See the module docstring.
CELLS = [
    (False, True, False, True),   # v1 baseline: C++ / Rust / R / Java's defaults
    (False, True, True, True),
    (False, False, False, True),
    (False, False, True, True),   # v1, as the pre-fix Python/PyTorch harness pinned it
    (True, True, False, True),    # v2 default: every stack, DEEPGREEN_TF32=1
    (True, True, False, False),   # v2 default, cuDNN disabled: bounds section 19
]
BASELINE_CELL = (False, True, False, True)
V2_DEFAULT_CELL = (True, True, False, True)
NOCUDNN_CELL = (True, True, False, False)


def _cell_label(cell: tuple[bool, bool, bool, bool]) -> str:
    matmul_allow_tf32, cudnn_allow_tf32, deterministic, cudnn_enabled = cell
    return (f"(matmul={matmul_allow_tf32}, cudnn={cudnn_allow_tf32}, "
            f"det={deterministic}, cudnn_enabled={cudnn_enabled})")


def _warmup_by_time(model, optimizer, criterion, x, y, seconds: float) -> None:
    """Warm up until real time has passed, not until a step count is reached.

    A fixed step count is tens of milliseconds in the fast cells -- too short
    for the GPU to leave its idle P-state -- so the measured region could
    straddle the clock ramp and read a deflated, cell-dependent baseline.
    Looping on wall time instead gives the clock room to settle regardless of
    how cheap a step in this particular cell happens to be.
    """
    import torch

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimizer.step()
        torch.cuda.synchronize()


def run_cell(matmul_allow_tf32: bool, cudnn_allow_tf32: bool, deterministic: bool,
            cudnn_enabled: bool, steps: int, batch: int,
            counters: HardwareCounters, device) -> tuple[float, float, float]:
    """One measured cell: per-step wall seconds, per-step GPU joules, window_s."""
    import torch
    import torch.nn as nn

    torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.enabled = cudnn_enabled

    model = _campaign_resnet18("resnet18", NUM_CLASSES).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(batch, 3, *INPUT_HW, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch,), device=device)

    _warmup_by_time(model, optimizer, criterion, x, y, WARMUP_SECONDS)
    torch.cuda.synchronize()

    a = counters.snapshot()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimizer.step()
    torch.cuda.synchronize()
    b = counters.snapshot()

    d = counters.delta(a, b)
    window_s = d["duration_s"]
    if window_s < MIN_WINDOW_S:
        cell = (matmul_allow_tf32, cudnn_allow_tf32, deterministic, cudnn_enabled)
        print(f"warning: {_cell_label(cell)} "
              f"window is {window_s:.2f} s, below the {MIN_WINDOW_S:.0f} s floor -- "
              "NVML's ~10 Hz register makes this delta quantisation-bound; "
              "raise --steps.", file=sys.stderr)

    del model, optimizer, x, y
    torch.cuda.empty_cache()
    return d["duration_s"] / steps, d["gpu_j"] / steps, window_s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1000,
                    help="measured training steps per cell, per repeat (default "
                         "1000, chosen so the fastest cell's window clears "
                         f"{MIN_WINDOW_S:.0f} s)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent repeats per cell; the table reports the "
                         "median (default 3)")
    ap.add_argument("--steps-nocudnn", type=int, default=300,
                    help="measured steps for the cudnn.enabled=False cell only "
                         "(default 300): that cell is far slower per step "
                         "(REVISION_LOG.md section 19 implies roughly 17x), so "
                         "it runs fewer of them while still clearing "
                         f"{MIN_WINDOW_S:.0f} s")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"batch size (default {DEFAULT_BATCH_SIZE}, the campaign's)")
    ap.add_argument("--seed", type=int, default=1000,
                    help="seeds the synthetic batch and the cell-order shuffle, "
                         "for a reproducible run (default 1000)")
    args = ap.parse_args()

    _assert_accelerator_idle()

    import torch

    if not torch.cuda.is_available():
        print("error: no CUDA device visible; this ablation is GPU-only.",
              file=sys.stderr)
        return 1
    device = torch.device("cuda:0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    counters = HardwareCounters()
    if not counters.available.get("gpu"):
        print(f"error: GPU energy counter unavailable ({counters.available}); "
              "the GPU-joules half of this table cannot be measured honestly.",
              file=sys.stderr)
        return 1

    # Repeats x cells, shuffled independently per repeat (from --seed) so
    # thermal drift over the whole run does not alias onto one cell. The
    # no-cuDNN cell uses its own, shorter step count -- see --steps-nocudnn.
    results: dict[tuple[bool, bool, bool, bool], dict[str, list[float]]] = {
        cell: {"wall_s": [], "gpu_j": [], "window_s": []} for cell in CELLS
    }
    steps_used: dict[tuple[bool, bool, bool, bool], int] = {}
    for rep in range(args.repeats):
        order = list(CELLS)
        random.shuffle(order)
        for cell in order:
            steps = args.steps_nocudnn if cell == NOCUDNN_CELL else args.steps
            steps_used[cell] = steps
            wall_s, gpu_j, window_s = run_cell(*cell, steps, args.batch,
                                               counters, device)
            results[cell]["wall_s"].append(wall_s)
            results[cell]["gpu_j"].append(gpu_j)
            results[cell]["window_s"].append(window_s)
            print(f"  rep {rep + 1}/{args.repeats}  {_cell_label(cell)}  "
                  f"{wall_s:.4f} s/step  {gpu_j:.2f} J/step  "
                  f"window {window_s:.2f} s")

    baseline_wall = statistics.median(results[BASELINE_CELL]["wall_s"])
    baseline_gpu = statistics.median(results[BASELINE_CELL]["gpu_j"])
    v2_wall = statistics.median(results[V2_DEFAULT_CELL]["wall_s"])
    v2_gpu = statistics.median(results[V2_DEFAULT_CELL]["gpu_j"])

    rows = []
    for cell in CELLS:
        matmul_allow_tf32, cudnn_allow_tf32, deterministic, cudnn_enabled = cell
        wall_runs = results[cell]["wall_s"]
        gpu_runs = results[cell]["gpu_j"]
        window_runs = results[cell]["window_s"]
        wall_med = statistics.median(wall_runs)
        gpu_med = statistics.median(gpu_runs)
        rows.append({
            "matmul_allow_tf32": matmul_allow_tf32,
            "cudnn_allow_tf32": cudnn_allow_tf32,
            "cudnn_deterministic": deterministic,
            "cudnn_enabled": cudnn_enabled,
            "wall_s_median": round(wall_med, 4),
            "gpu_j_median": round(gpu_med, 2),
            "window_s_median": round(statistics.median(window_runs), 2),
            "ratio_wall_s_vs_v1_baseline": round(wall_med / baseline_wall, 3),
            "ratio_gpu_j_vs_v1_baseline": round(gpu_med / baseline_gpu, 3),
            "ratio_wall_s_vs_v2_default": round(wall_med / v2_wall, 3),
            "ratio_gpu_j_vs_v2_default": round(gpu_med / v2_gpu, 3),
            "wall_s_runs": ",".join(f"{v:.4f}" for v in wall_runs),
            "gpu_j_runs": ",".join(f"{v:.2f}" for v in gpu_runs),
            "window_s_runs": ",".join(f"{v:.2f}" for v in window_runs),
            "steps": steps_used[cell],
            "batch": args.batch,
            "seed": args.seed,
        })

    df = pd.DataFrame(rows)
    save_table(
        df, "v2_tf32_ablation",
        "Precision ablation (matmul.allow_tf32 x cudnn.allow_tf32 x "
        "cudnn.deterministic x cudnn.enabled), RTX 3090, ResNet-18, batch 128, "
        "fp32, median of three -- kernel-level confirmation of "
        "REVISION_LOG.md sections 1 and 19; the citable numbers are the "
        "campaign's own v1-vs-v2 contrast and section 19's Deeplearning4j bound",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
