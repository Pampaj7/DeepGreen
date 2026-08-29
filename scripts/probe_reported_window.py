#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why CodeCarbon's reported duration exceeds the interval it measured.

The campaign shows it: two thirds of measured blocks are filed with a window
seconds longer than the phase the energy was accumulated over, and the excess
falls in three tight modes. This script establishes the cause, on the same
machine, in about two minutes and without a GPU.

``EmissionsTracker.stop()`` calls ``_prepare_emissions_data()``, which resolves
cloud metadata and then the host's geolocation over the network -- two blocking
HTTP requests, made *after* the final energy reading. That is why the energy is
unaffected and only the duration is wrong.

Both results are cached on the tracker instance. A background thread also calls
``_prepare_emissions_data()`` once ``api_call_interval`` measurements have
accumulated (default 8, at ``measure_power_secs=1``), but only when an output
handler is attached. So:

  * a block shorter than the background call pays for both lookups at stop();
  * a block that outlives one of them pays for the remaining one;
  * a block that outlives both pays nothing.

Three modes, and a threshold near eleven seconds. Run it and see:

    python3 scripts/probe_reported_window.py

Writes results/revision/tables/v2_window_mechanism.{md,csv}.

A note on getting this wrong. Our first probe passed ``save_to_file=False``,
which detaches the output handler, which disables the background call, which
makes stop() pay both lookups at every duration -- and produced a flat 4.5 s
with no threshold at all. The campaign runs with the file handler attached, so
the probe has to as well.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "results" / "analysis"))
from common import save_table  # noqa: E402

# Spans the region where the campaign's blocks fall, either side of the
# background call's ~8 measurement threshold.
DURATIONS = (2.0, 5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0)


def main() -> int:
    logging.disable(logging.CRITICAL)
    from codecarbon import EmissionsTracker

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)                       # the handler writes emissions.csv here
        try:
            for work in DURATIONS:
                tracker = EmissionsTracker(
                    measure_power_secs=1, tracking_mode="machine",
                    save_to_file=True, log_level="error", allow_multiple_runs=True)
                tracker.start()
                time.sleep(work)
                t0 = time.perf_counter()
                tracker.stop()
                cost = time.perf_counter() - t0
                rows.append({
                    "work_s": work,
                    "stop_cost_s": round(cost, 3),
                    "lookups_outstanding": 2 if cost > 4.0 else (1 if cost > 1.0 else 0),
                })
                print(f"  work {work:5.1f} s -> stop() {cost:6.3f} s")
        finally:
            os.chdir(cwd)

    out = pd.DataFrame(rows)
    save_table(out, "v2_window_mechanism",
               "Cost of EmissionsTracker.stop() against the length of the block "
               "it closes, on this host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
