#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The machine's idle draw at the same boundary the campaign reports.

Every energy figure in this study is whole-machine and un-baselined: the
counters accumulate whatever the silicon draws, including the host's static
consumption, and nothing in the campaign records what that static term is. A
reviewer is entitled to ask how much of a reported block is the machine simply
being on, and the answer should be measured rather than argued.

This samples the same two counters the campaign uses, over an idle host, and
writes the result where the analysis can read it.

  python3 scripts/measure_idle.py [--seconds 60]

Run it on an idle machine, as the campaign requires of itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from hardware_counters import HardwareCounters  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--settle", type=float, default=10.0,
                    help="discarded before measuring, so the tail of whatever "
                         "ran last is not charged to the idle figure")
    args = ap.parse_args()

    counters = HardwareCounters()
    if not all(counters.available.values()):
        print(f"counters unavailable: {counters.available}", file=sys.stderr)
        return 1

    print(f"settling for {args.settle:.0f} s ...")
    time.sleep(args.settle)
    print(f"measuring idle for {args.seconds:.0f} s ...")
    a = counters.snapshot()
    time.sleep(args.seconds)
    b = counters.snapshot()
    d = counters.delta(a, b)

    seconds = d["duration_s"]
    record = {
        "seconds": round(seconds, 2),
        "gpu_w": round(d.get("gpu_j", 0.0) / seconds, 2),
        "cpu_package_w": round(d.get("cpu_package_total_j", 0.0) / seconds, 2),
        "total_w": round(d.get("hw_total_j", 0.0) / seconds, 2),
    }
    out = REPO_ROOT / "results" / "revision" / "tables" / "v2_idle_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
