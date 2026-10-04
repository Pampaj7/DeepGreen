#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check that every job in the campaign plan can actually run.

A campaign is hours to days long. Discovering at job 40 that one binary was never
built, or that a training script has a different name than the driver assumes,
costs the whole run its uniformity: the fixed jobs then execute in a different
machine state from the rest.

    source scripts/campaign_env.sh
    python3 scripts/preflight.py --repetitions 5

The accelerator-saturation cell (spec S7) is a separate campaign and is
preflighted the same way the driver runs it -- by naming it, with a campaign
directory of its own:

    DEEPGREEN_CAMPAIGN_DIR=results/campaign_saturation \
        python3 scripts/preflight.py --datasets imagenette
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAIL: list[str] = []
WARN: list[str] = []


def check(ok: bool, label: str, detail: str = "", warn_only: bool = False) -> None:
    mark = "ok  " if ok else ("warn" if warn_only else "FAIL")
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        (WARN if warn_only else FAIL).append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--datasets", nargs="*",
                    help="preflight this grid instead of the default three; "
                         "pass 'imagenette' for the accelerator-saturation cell")
    args = ap.parse_args()

    print("=" * 78)
    print("CAMPAIGN PREFLIGHT")
    print("=" * 78)

    # --- the shared artefacts every stack depends on ---------------------
    print("\nshared artefacts")
    data = Path(os.environ.get("DEEPGREEN_DATA", REPO / "data"))
    # imagenette_png is the saturation cell: ten ImageNet classes at 224x224,
    # written once by dataloader/download_convert_imagenette.py. It is listed
    # here whichever grid is being preflighted, because a missing conversion is
    # worth knowing about before the cell is scheduled rather than at job 1.
    for ds, want in (("cifar100_png", 32), ("fashion_mnist_png", 32),
                     ("tiny_imagenet_png", 32), ("imagenette_png", 224)):
        d = data / ds
        n = sum(1 for _ in d.rglob("*.png")) if d.is_dir() else 0
        check(n > 0, f"dataset {ds}", f"{n} images at {want}x{want}")
    models = Path(os.environ.get("DEEPGREEN_MODELS", REPO / "models"))
    n_pt = len(list(models.glob("*.pt"))) if models.is_dir() else 0
    # Eight: two architectures x four datasets. The saturation cell's two --
    # resnet18_imagenette.pt and vgg16_imagenette.pt -- are ten-class modules
    # like the Fashion-MNIST pair and carry the same parameter counts.
    check(n_pt == 8, "shared TorchScript modules", f"{n_pt}/8")
    py = Path(os.environ.get("DEEPGREEN_PYTHON", ""))
    check(py.exists(), "measurement interpreter", str(py))

    # --- the counters -----------------------------------------------------
    print("\ninstruments")
    try:
        from tools.hardware_counters import HardwareCounters

        hc = HardwareCounters()
        check(hc.available["gpu"], "NVML energy counter")
        check(hc.available["cpu"], "RAPL domains readable",
              "sudo chmod -R a+r /sys/class/powercap/intel-rapl" if not hc.available["cpu"] else
              ", ".join(hc.describe()["rapl_domains"]),
              warn_only=not hc.available["cpu"])
    except Exception as e:
        check(False, "hardware counters importable", str(e))

    # --- every job's executable ------------------------------------------
    print("\njobs")
    plan_cmd = [sys.executable, "scripts/run_campaign.py",
                "--repetitions", str(args.repetitions), "--print-plan"]
    if args.datasets:
        plan_cmd += ["--datasets", *args.datasets]
    done = subprocess.run(plan_cmd, cwd=REPO, capture_output=True, text=True)
    # The plan lands wherever the driver writes runs, so preflighting the
    # saturation cell reads that cell's plan and not campaign_v2's.
    campaign = Path(os.environ.get("DEEPGREEN_CAMPAIGN_DIR",
                                   REPO / "results" / "campaign_v2"))
    plan_path = campaign / "plan.json"
    if done.returncode or not plan_path.exists():
        check(False, "campaign plan", (done.stderr.strip().splitlines() or
                                       [f"{plan_path} not written"])[0])
        print("\n" + "-" * 78)
        print(f"  no plan to preflight, {len(FAIL)} blocking")
        print("-" * 78)
        return 1
    plan = json.loads(plan_path.read_text())

    seen: set[str] = set()
    for job in plan:
        cmd = job["command"]
        if cmd in seen:
            continue
        seen.add(cmd)
        first = cmd.split()[0]
        if first.endswith("python") or "/bin/python" in first:
            ok, detail = Path(first).exists(), first
        elif first in ("mvn", "Rscript"):
            ok = shutil.which(first) is not None
            detail = shutil.which(first) or "not on PATH"
            if ok and first == "Rscript":
                script = cmd.split()[1]
                ok = (REPO / script).exists()
                detail = script if ok else f"missing script {script}"
        else:
            p = REPO / first
            ok, detail = p.exists() and os.access(p, os.X_OK), first
        check(ok, f"{job['ecosystem']:<18} {job['model']}/{job['dataset']}", detail)

    print("\n" + "-" * 78)
    print(f"  {len(plan)} jobs, {len(seen)} distinct commands, "
          f"{len(FAIL)} blocking, {len(WARN)} warnings")
    if FAIL:
        print("\n  blocking:")
        for f in FAIL:
            print(f"    - {f}")
    print("-" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
