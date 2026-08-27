#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy from hardware counters, measured alongside CodeCarbon.

Why
---
CodeCarbon's GPU term reads the NVML energy counter and is sound. Its CPU and
RAM terms are *models*, and where RAPL is unavailable the CPU term is a hardcoded
constant. On the machine that produced the campaign under study this mattered a
great deal: for four of eight ecosystems the modelled host terms were
≈231 W of constant power, roughly two thirds of the reported energy, and the RAM
term was derived from *installed* memory rather than from use.

This module reads the counters directly instead:

  * **GPU** -- ``nvmlDeviceGetTotalEnergyConsumption``, a millijoule counter
    integrated by the board. No sampling, no interval to choose, no model.
  * **CPU** -- Intel RAPL under ``/sys/class/powercap``, per domain
    (``package-0``, ``core``, and ``dram`` where the platform exposes it).

Recording both instruments in the same run turns "is CodeCarbon accurate enough?"
from an assumption into a measurement, which is what reviewer comment 9 of the
study under audit asked for.

Two properties worth knowing:

  * NVML's counter is **per device**: work on other devices, or CPU-only work
    elsewhere on the host, does not enter it. RAPL's is **per package**, so it
    behaves like CodeCarbon's machine mode and the host must still be idle.
  * RAPL counters **wrap**. The range is exposed as ``max_energy_range_uj``; on
    this platform it is ≈65.5 kJ, which at a few tens of watts is under half an
    hour. A reader that ignores wraparound silently loses whole epochs.

Permissions
-----------
RAPL is root-readable only on most current kernels (a side-channel mitigation).
Grant read access once per boot::

    sudo chmod -R a+r /sys/class/powercap/intel-rapl

If it is unreadable the CPU domains are reported as unavailable rather than
guessed, and the GPU counter is used on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

RAPL_ROOT = Path("/sys/class/powercap")
J_PER_UJ = 1e-6
J_PER_MJ = 1e-3


@dataclass
class Snapshot:
    """One reading of every available counter."""

    t: float
    gpu_mj: int | None = None
    rapl_uj: dict[str, int] = field(default_factory=dict)


class HardwareCounters:
    """Direct energy counters for one GPU and the host package domains."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._nvml = None
        self._handle = None
        self._rapl: dict[str, Path] = {}
        self._rapl_range: dict[str, int] = {}
        self._init_gpu()
        self._init_rapl()

    # -- discovery --------------------------------------------------------
    def _init_gpu(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            # probe: not every device exposes the energy counter
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self._nvml = pynvml
        except Exception:
            self._nvml = None
            self._handle = None

    def _init_rapl(self) -> None:
        if not RAPL_ROOT.is_dir():
            return
        for entry in sorted(RAPL_ROOT.glob("intel-rapl*")):
            name_file = entry / "name"
            energy_file = entry / "energy_uj"
            if not (name_file.is_file() and energy_file.is_file()):
                continue
            try:
                name = name_file.read_text().strip()
                energy_file.read_text()  # permission probe
            except OSError:
                continue  # unreadable: report as unavailable, never guess
            key = f"{name}@{entry.name}"
            self._rapl[key] = energy_file
            try:
                self._rapl_range[key] = int(
                    (entry / "max_energy_range_uj").read_text().strip())
            except (OSError, ValueError):
                self._rapl_range[key] = 0

    # -- reading ----------------------------------------------------------
    @property
    def available(self) -> dict[str, bool]:
        return {"gpu": self._nvml is not None, "cpu": bool(self._rapl)}

    def snapshot(self) -> Snapshot:
        s = Snapshot(t=time.monotonic())
        if self._nvml is not None:
            try:
                s.gpu_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            except Exception:
                s.gpu_mj = None
        for key, path in self._rapl.items():
            try:
                s.rapl_uj[key] = int(path.read_text().strip())
            except (OSError, ValueError):
                pass
        return s

    def delta(self, a: Snapshot, b: Snapshot) -> dict[str, float]:
        """Energy in Joules between two snapshots, wraparound-corrected."""
        out: dict[str, float] = {"duration_s": b.t - a.t}

        if a.gpu_mj is not None and b.gpu_mj is not None:
            # 64-bit millijoule counter; no practical wraparound
            out["gpu_j"] = (b.gpu_mj - a.gpu_mj) * J_PER_MJ

        for key in a.rapl_uj:
            if key not in b.rapl_uj:
                continue
            raw = b.rapl_uj[key] - a.rapl_uj[key]
            if raw < 0:
                # RAPL wraps at max_energy_range_uj. Without this correction a
                # wrap turns into a large negative energy, or is dropped, and a
                # whole epoch is lost from the record.
                rng = self._rapl_range.get(key, 0)
                if rng <= 0:
                    continue  # cannot correct honestly: omit rather than invent
                raw += rng
            out[f"cpu_{key}_j"] = raw * J_PER_UJ

        pkg = [v for k, v in out.items() if k.startswith("cpu_package")]
        if pkg:
            out["cpu_package_total_j"] = sum(pkg)
        if "gpu_j" in out and pkg:
            out["hw_total_j"] = out["gpu_j"] + out["cpu_package_total_j"]
        return out

    def describe(self) -> dict[str, object]:
        return {
            "gpu_counter": "nvmlDeviceGetTotalEnergyConsumption" if self._nvml else None,
            "gpu_index": self.gpu_index if self._nvml else None,
            "rapl_domains": sorted(self._rapl),
            "rapl_wrap_uj": dict(self._rapl_range),
            "note": "RAPL is per package and wraps; GPU counter is per device.",
        }


if __name__ == "__main__":
    import json

    hc = HardwareCounters()
    print(json.dumps(hc.describe(), indent=2))
    print("available:", hc.available)
    a = hc.snapshot()
    time.sleep(2.0)
    b = hc.snapshot()
    d = hc.delta(a, b)
    for k, v in sorted(d.items()):
        if k == "duration_s":
            print(f"  {k:26} {v:8.2f} s")
        else:
            print(f"  {k:26} {v:8.2f} J   ({v / d['duration_s']:6.1f} W)")
