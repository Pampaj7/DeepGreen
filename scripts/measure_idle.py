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

Run it on an idle machine, as the campaign requires of itself -- and the script
now checks rather than trusts. It was in results/analysis/run_all.sh
unconditionally, so a full pipeline run measured the idle host while being the
load on it: the CPU package term came out 3.6 W above a reading taken an hour
earlier on a quiet machine, and the file it overwrote is a manuscript input. An
idle baseline measured by the pipeline that is itself the load is the defect
this paper catalogues, committed by the paper. So the load average is recorded
in the file, the measurement refuses to write above 0.5 without --force, and
run_all.sh runs this only under DEEPGREEN_MEASURE_HOST=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "results" / "analysis"))

from common import write_table_path  # noqa: E402
from hardware_counters import HardwareCounters  # noqa: E402

#: Above this one-minute load average the host is not idle and the reading
#: would be of whatever else is running. 0.5 admits the sampler itself and the
#: usual background daemons and little else.
MAX_LOAD = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--settle", type=float, default=10.0,
                    help="discarded before measuring, so the tail of whatever "
                         "ran last is not charged to the idle figure")
    ap.add_argument("--force", action="store_true",
                    help=f"measure even above a {MAX_LOAD} load average; the "
                         f"result is not an idle baseline and the file will "
                         f"say so")
    args = ap.parse_args()

    counters = HardwareCounters()
    if not all(counters.available.values()):
        print(f"counters unavailable: {counters.available}", file=sys.stderr)
        return 1

    load = os.getloadavg()[0]
    if load > MAX_LOAD and not args.force:
        print(f"one-minute load average is {load:.2f}, above {MAX_LOAD}: this "
              f"host is not idle and the reading would be of whatever else is "
              f"running. Refusing to overwrite the baseline (--force to "
              f"measure anyway).", file=sys.stderr)
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
        # When, and how idle. Without these a contaminated reading is
        # indistinguishable from a clean one after the fact, which is exactly
        # what happened: the difference between the two readings on record is
        # 3.6 W of CPU package and nothing in either file says which host state
        # produced it.
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "load_average_1min": round(load, 2),
        "forced": bool(args.force),
    }
    # Through the diversion, like every other table: this file is a manuscript
    # input and a monitoring run must not be able to replace it.
    out = write_table_path("v2_idle_baseline.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
