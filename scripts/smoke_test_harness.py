#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end check of the measurement harness, without a GPU and without a real
training run.

Verifies what the first campaign got wrong (spec S5):
  * CodeCarbon is >= 3.0 and refuses to run under 2.x;
  * the pinned configuration actually reaches the tracker;
  * one CSV is produced per epoch per phase;
  * energy is in kWh and converts to a plausible power;
  * per-epoch quality metrics land in metrics.csv;
  * manifest.json records the resolved environment.

    python3 scripts/smoke_test_harness.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.deepgreen_bench import CODECARBON_CONFIG, Harness, RunContext  # noqa: E402

J_PER_KWH = 3.6e6
EPOCHS = 2
failures: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def main() -> int:
    print("=" * 72)
    print("HARNESS SMOKE TEST (spec S5)")
    print("=" * 72)

    import codecarbon
    import pandas as pd

    ver = tuple(int(p) for p in codecarbon.__version__.split(".")[:2])
    check(ver >= (3, 0), "codecarbon >= 3.0", codecarbon.__version__)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = RunContext(ecosystem="smoke", model="noop", dataset="none",
                         repetition=0, epochs=EPOCHS)
        out = Path(tmp)
        with Harness(ctx, out_dir=out) as bench:
            bench.set_seeds()
            for epoch in range(1, EPOCHS + 1):
                with bench.track("train", epoch):
                    _burn(0.6)
                with bench.track("eval", epoch):
                    _burn(0.3)
                bench.log_metrics(epoch, train_loss=1.0 / epoch,
                                  test_loss=1.1 / epoch, test_acc=10.0 * epoch)

        emissions = sorted(out.glob("emissions_*.csv"))
        check(len(emissions) == 2 * EPOCHS,
              "one CSV per epoch per phase", f"{len(emissions)} files")

        frames = [pd.read_csv(p) for p in emissions]
        df = pd.concat(frames, ignore_index=True)
        check("energy_consumed" in df.columns, "energy_consumed column present")

        power = (df["energy_consumed"] * J_PER_KWH / df["duration"]).dropna()
        plausible = bool(len(power)) and bool(((power > 0.5) & (power < 2000)).all())
        check(plausible, "energy reads as kWh (implied power in a sane range)",
              f"{power.min():.1f}-{power.max():.1f} W" if len(power) else "no rows")

        check(bool((df["duration"] > 0).all()), "durations positive")

        metrics = pd.read_csv(out / "metrics.csv")
        check(len(metrics) == EPOCHS, "one metrics row per epoch", f"{len(metrics)} rows")
        for col in ("train_loss", "test_loss", "test_acc", "seed", "precision"):
            check(col in metrics.columns, f"metrics.csv has {col}")

        manifest = json.loads((out / "manifest.json").read_text())
        check(manifest["codecarbon_config"] == CODECARBON_CONFIG,
              "pinned CodeCarbon config recorded")
        check(manifest["codecarbon_config"]["tracking_mode"] == "machine",
              "tracking_mode = machine")
        check(manifest["codecarbon_config"]["measure_power_secs"] == 1,
              "measure_power_secs = 1")
        check("framework_versions" in manifest, "framework versions recorded")

    print("-" * 72)
    if failures:
        print(f"  {len(failures)} check(s) failed: " + ", ".join(failures))
        return 1
    print("  all checks passed")
    return 0


def _burn(seconds: float) -> None:
    """A little CPU work so the tracked block has non-zero duration."""
    import time

    t0 = time.time()
    x = 0
    while time.time() - t0 < seconds:
        x += sum(i * i for i in range(1000))


if __name__ == "__main__":
    raise SystemExit(main())
