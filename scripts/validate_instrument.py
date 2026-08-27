#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare CodeCarbon against hardware energy counters over the same window.

Reviewer comment 9 of the study under audit asks for the measurement setup to be
validated against a hardware meter. This is that validation, run on the
replication machine: a known GPU workload of controlled duration, measured
simultaneously by CodeCarbon and by NVML's energy counter plus Intel RAPL.

Both instruments bracket the *same* window: the hardware snapshots are taken
immediately inside the CodeCarbon tracker, so the comparison is not confounded by
CodeCarbon's own start-up cost.

    python3 scripts/validate_instrument.py [--repeats 3]

Why it matters here: the inference blocks in the campaign under audit last
1.6-13 s, and the accuracy of a sampling-based estimator depends on how many
samples fit in the window.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.deepgreen_bench import CODECARBON_CONFIG  # noqa: E402
from tools.hardware_counters import HardwareCounters  # noqa: E402

J_PER_KWH = 3.6e6


def gpu_burn(seconds: float) -> None:
    """A steady GPU load, so the comparison is not about workload shape."""
    import torch

    dev = torch.device("cuda")
    a = torch.randn(4096, 4096, device=dev)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        for _ in range(20):
            a = torch.mm(a, a).clamp_(-1.0, 1.0)
        torch.cuda.synchronize()


def one_block(hc: HardwareCounters, seconds: float, out_dir: Path, tag: str) -> dict:
    from codecarbon import EmissionsTracker
    import pandas as pd

    tracker = EmissionsTracker(output_dir=str(out_dir),
                               output_file=f"{tag}.csv", **CODECARBON_CONFIG)
    tracker.start()
    before = hc.snapshot()          # inside the tracker: same window
    gpu_burn(seconds)
    after = hc.snapshot()
    tracker.stop()

    hw = hc.delta(before, after)
    cc = pd.read_csv(out_dir / f"{tag}.csv").iloc[-1]
    return {
        "target_s": seconds,
        "hw_s": hw["duration_s"],
        "cc_s": float(cc["duration"]),
        "hw_gpu_j": hw.get("gpu_j", float("nan")),
        "cc_gpu_j": float(cc["gpu_energy"]) * J_PER_KWH,
        "hw_cpu_j": hw.get("cpu_package_total_j", float("nan")),
        "cc_cpu_j": float(cc["cpu_energy"]) * J_PER_KWH,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--durations", type=float, nargs="*",
                    default=[2.0, 5.0, 10.0, 20.0, 40.0])
    args = ap.parse_args()

    hc = HardwareCounters()
    if not hc.available["gpu"]:
        print("no GPU energy counter available", file=sys.stderr)
        return 1
    print("counters:", hc.available, "| CodeCarbon:", CODECARBON_CONFIG)

    out_dir = REPO / "results" / "revision" / "instrument_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'target':>7} {'window':>16} {'GPU energy (J)':>26} {'CPU package (J)':>26}")
    print(f"{'':>7} {'hw':>7} {'cc':>8} {'hw':>9} {'cc':>9} {'cc/hw':>7} {'hw':>9} {'cc':>9} {'cc/hw':>7}")
    ratios: dict[float, list[float]] = {}
    for d in args.durations:
        for r in range(args.repeats):
            row = one_block(hc, d, out_dir, f"block_{d:g}s_{r}")
            gr = row["cc_gpu_j"] / row["hw_gpu_j"] if row["hw_gpu_j"] else float("nan")
            cr = row["cc_cpu_j"] / row["hw_cpu_j"] if row["hw_cpu_j"] else float("nan")
            ratios.setdefault(d, []).append(gr)
            print(f"{d:7.0f} {row['hw_s']:7.2f} {row['cc_s']:8.2f} "
                  f"{row['hw_gpu_j']:9.0f} {row['cc_gpu_j']:9.0f} {gr:7.2f} "
                  f"{row['hw_cpu_j']:9.0f} {row['cc_cpu_j']:9.0f} {cr:7.2f}")

    print("\nGPU energy reported by CodeCarbon as a fraction of the counter:")
    for d, rs in sorted(ratios.items()):
        med = statistics.median(rs)
        print(f"  {d:5.0f} s block: {med:5.2f}   ({(med - 1) * 100:+.0f}% error)")
    print("\nA sampling estimator needs samples. The shorter the block, the fewer")
    print("it gets, and the inference phase of the campaign under audit is 1.6-13 s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
