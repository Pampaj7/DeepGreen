#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One measurement bridge for every non-Python ecosystem.

Why this exists
---------------
C++, Java, R, Rust and MATLAB all drive CodeCarbon through a Python helper, and
in the first campaign each shipped its OWN helper with its OWN configuration:

    rust/scripts/tracker_control.py        measure_power_secs=1, machine
    R/scripts/tracker_control.py           measure_power_secs=1, tracking_mode=process
    cpp/py_script/tracker/...              defaults (15 s, machine)
    Java/.../resources/tracker/...         defaults (15 s, machine)
    matlab/tracker/tracker_control.py      defaults, with the 1 s line commented out

Five copies, four configurations. The Rust daemon additionally spawned `python3`
from PATH, so the CodeCarbon *version* measuring that stack was whatever the
ambient shell resolved -- which is how one campaign mixed 2.8.4 and 3.0.4.

Every stack now uses this file, so the instrument is identical by construction
rather than by discipline.

The run contract (environment variables)
----------------------------------------
    DEEPGREEN_RUN_DIR   where to write emissions_<phase>_epoch<N>.csv and metrics.csv
    DEEPGREEN_ECOSYSTEM e.g. "Rust/tch"
    DEEPGREEN_MODEL     resnet18 | vgg16
    DEEPGREEN_DATASET   fashionmnist | cifar100 | tinyimagenet
    DEEPGREEN_REP       repetition index
    DEEPGREEN_SEED      seed for this repetition
    DEEPGREEN_EPOCHS    epochs (default 30)
    DEEPGREEN_DATA      dataset root (default ./data)
    DEEPGREEN_MODELS    shared TorchScript modules (default ./models)

Environment variables rather than CLI flags: adding argument parsing to a C++
binary, a Maven exec target, an Rscript and a Rust binary is four different
pieces of plumbing, and every one of them is a place for the stacks to drift
apart again.

Usage
-----
    # daemon over stdin, one command per line (Rust, C++, Java)
    python3 tools/deepgreen_tracker.py --daemon
        START train 1
        STOP
        METRIC epoch=1 train_loss=0.44 test_loss=0.35 test_acc=87.11
        EXIT

    # module (anything that can import Python)
    from tools.deepgreen_tracker import start, stop, metric
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.deepgreen_bench import CODECARBON_CONFIG, _check_codecarbon_version  # noqa: E402

_tracker = None
_counters = None
_hw_before = None
_hw_block = None
_metrics_path: Path | None = None
_metrics_fields = ["ecosystem", "model", "dataset", "repetition", "seed", "epoch",
                   "train_loss", "train_acc", "test_loss", "test_acc"]


def run_dir() -> Path:
    d = os.environ.get("DEEPGREEN_RUN_DIR")
    if not d:
        raise SystemExit(
            "DEEPGREEN_RUN_DIR is not set. The campaign driver sets it; see the run "
            "contract at the top of tools/deepgreen_tracker.py."
        )
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ctx() -> dict[str, str]:
    return {
        "ecosystem": os.environ.get("DEEPGREEN_ECOSYSTEM", "unknown"),
        "model": os.environ.get("DEEPGREEN_MODEL", "unknown"),
        "dataset": os.environ.get("DEEPGREEN_DATASET", "unknown"),
        "repetition": os.environ.get("DEEPGREEN_REP", "0"),
        "seed": os.environ.get("DEEPGREEN_SEED", ""),
    }


def write_manifest(extra: dict | None = None) -> None:
    """Record the resolved environment, as the Python harness does."""
    from importlib.metadata import version

    try:
        cc = version("codecarbon")
    except Exception:
        cc = None
    manifest = {
        "run": _ctx() | {"epochs": os.environ.get("DEEPGREEN_EPOCHS", "30")},
        "codecarbon_config": CODECARBON_CONFIG,
        "codecarbon_version": cc,
        "python": sys.version,
        "interpreter": sys.executable,
        "env": {k: v for k, v in os.environ.items() if k.startswith("DEEPGREEN_")},
        "hardware_counters": (_hardware().describe() if _hardware() is not None else None),
    }
    if extra:
        manifest |= extra
    (run_dir() / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _hardware() -> object | None:
    """The second instrument: NVML and RAPL counters, if reachable."""
    global _counters
    if _counters is None:
        try:
            from tools.hardware_counters import HardwareCounters

            hc = HardwareCounters()
            _counters = hc if any(hc.available.values()) else False
        except Exception as e:
            print(f"[deepgreen] hardware counters unavailable: {e}",
                  file=sys.stderr, flush=True)
            _counters = False
    return _counters or None


def start(phase: str, epoch: int) -> None:
    """Begin tracking one epoch of one phase, with both instruments."""
    global _tracker, _hw_before, _hw_block
    from codecarbon import EmissionsTracker

    _check_codecarbon_version()
    if _tracker is not None:
        print(f"[deepgreen] tracker already active, ignoring START {phase} {epoch}",
              file=sys.stderr, flush=True)
        return
    _tracker = EmissionsTracker(
        output_dir=str(run_dir()),
        output_file=f"emissions_{phase}_epoch{epoch}.csv",
        project_name=os.environ.get("DEEPGREEN_RUN_DIR", "deepgreen").split("/")[-1],
        **CODECARBON_CONFIG,
    )
    _hw_block = (phase, epoch)
    _tracker.start()
    # Inside the tracker window: see the note in tools/deepgreen_bench.py.
    hc = _hardware()
    _hw_before = hc.snapshot() if hc is not None else None


def stop() -> None:
    global _tracker, _hw_before
    if _tracker is None:
        print("[deepgreen] STOP with no active tracker", file=sys.stderr, flush=True)
        return
    hc = _hardware()
    hw_after = hc.snapshot() if hc is not None else None
    _tracker.stop()
    _tracker = None
    if hc is not None and _hw_before is not None and _hw_block is not None:
        _write_counters(_hw_block[0], _hw_block[1], hc.delta(_hw_before, hw_after))
    _hw_before = None


def _write_counters(phase: str, epoch: int, delta: dict) -> None:
    path = run_dir() / "counters.csv"
    row = {"phase": phase, "epoch": epoch} | {k: round(v, 6) for k, v in delta.items()}
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
_counters = None
_hw_before = None
_hw_block = None


def metric(**values) -> None:
    """Append one row to metrics.csv.

    Every ecosystem computed accuracy in the first campaign and every one of them
    only printed it, so energy could never be normalised by the useful work
    produced -- and a stack evaluating with batch-norm pinned to training mode
    went unnoticed for a whole campaign.
    """
    global _metrics_path
    path = run_dir() / "metrics.csv"
    new = not path.exists() or path.stat().st_size == 0
    row = _ctx() | {k: values.get(k) for k in
                    ("epoch", "train_loss", "train_acc", "test_loss", "test_acc")}
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_metrics_fields)
        if new:
            w.writeheader()
        w.writerow(row)
    _metrics_path = path


def _daemon() -> int:
    """Line protocol on stdin: START <phase> <epoch> | STOP | METRIC k=v... | EXIT."""
    write_manifest()
    print("[deepgreen] tracker ready", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            if line.startswith("START"):
                _, phase, epoch = line.split()
                start(phase, int(epoch))
                print(f"[deepgreen] START {phase} {epoch}", flush=True)
            elif line == "STOP":
                stop()
                print("[deepgreen] STOP", flush=True)
            elif line.startswith("METRIC"):
                kv = dict(tok.split("=", 1) for tok in line.split()[1:])
                metric(**{k: _num(v) for k, v in kv.items()})
                print("[deepgreen] METRIC", flush=True)
            elif line == "EXIT":
                break
            else:
                print(f"[deepgreen] unknown command: {line}", file=sys.stderr, flush=True)
        except Exception as e:  # a bad line must not kill a running campaign
            print(f"[deepgreen] ERROR on '{line}': {e}", file=sys.stderr, flush=True)
    if _tracker is not None:
        stop()
    return 0


def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        raise SystemExit(_daemon())
    print(__doc__)
