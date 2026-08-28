#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate the raw campaign output into a replication package.

``results/campaign_v2/`` holds one directory per run and, inside it, one
CodeCarbon file per measured block: 13,231 files and 54 MB for 210 runs. That
layout is right for the campaign driver, which appends as it goes and must
survive a crash mid-run, and wrong for anyone who wants to check our numbers --
nobody should have to walk twelve thousand two-line CSVs to do that.

This script rewrites the same records as four gzipped tables, adding the run
identity to every row so each table stands on its own:

  results/replication/codecarbon.csv.gz  one row per measured block, the
                                         software estimator's own output
  results/replication/counters.csv.gz    one row per measured block, NVML and
                                         RAPL as read by the shared bridge
  results/replication/metrics.csv.gz     one row per epoch, per-stack quality
  results/replication/manifests.csv.gz   one row per run: versions, seeds,
                                         environment, as recorded at launch

Nothing is aggregated, filtered or rounded here. The analysis pipeline reads
``results/campaign_v2/`` directly; this package is what it reads, flattened, so
that a reader can reproduce the pipeline's input without the pipeline.

  python3 scripts/consolidate_raw.py [--check]

``--check`` verifies an existing package against the raw tree instead of
rewriting it, which is what CI wants.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "campaign_v2"
OUT = REPO_ROOT / "results" / "replication"

# A run directory is named <ecosystem>_<model>_<dataset>_rep<n>, with the
# ecosystem's slash written as a dash. metrics.csv carries the canonical
# spelling, so the directory name is only ever used for the identity columns.
ECOSYSTEM = {
    "Python-PyTorch": "Python/PyTorch",
    "Python-TensorFlow": "Python/TensorFlow",
    "Python-JAX": "Python/JAX",
    "Cpp-LibTorch": "C++/LibTorch",
    "Java-DL4J": "Java/DL4J",
    "R-torch": "R/torch",
    "Rust-tch": "Rust/tch",
}


def identity(run_dir: Path) -> dict[str, object]:
    stem, rep = run_dir.name.rsplit("_rep", 1)
    eco, model, dataset = stem.split("_", 2)
    if eco not in ECOSYSTEM:
        raise SystemExit(f"unknown ecosystem in {run_dir.name!r}: {eco!r}")
    return {
        "run": run_dir.name,
        "ecosystem": ECOSYSTEM[eco],
        "model": model,
        "dataset": dataset,
        "repetition": int(rep),
    }


def flatten(obj, prefix: str = "") -> dict[str, object]:
    """manifest.json nests three levels deep; the table wants one."""
    out: dict[str, object] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        elif isinstance(value, list):
            out[name] = " ".join(str(v) for v in value)
        else:
            out[name] = value
    return out


def collect() -> dict[str, pd.DataFrame]:
    runs = sorted(d for d in RAW.iterdir() if d.is_dir())
    if not runs:
        raise SystemExit(f"no run directories under {RAW}")

    codecarbon, counters, metrics, manifests = [], [], [], []
    for run_dir in runs:
        ident = identity(run_dir)

        for path in sorted(run_dir.glob("emissions_*.csv")):
            phase, _, epoch = path.stem[len("emissions_"):].partition("_epoch")
            frame = pd.read_csv(path)
            frame.insert(0, "epoch", int(epoch))
            frame.insert(0, "phase", phase)
            for column, value in reversed(ident.items()):
                frame.insert(0, column, value)
            codecarbon.append(frame)

        path = run_dir / "counters.csv"
        if path.exists():
            frame = pd.read_csv(path)
            for column, value in reversed(ident.items()):
                frame.insert(0, column, value)
            counters.append(frame)

        path = run_dir / "metrics.csv"
        if path.exists():
            # metrics.csv already carries ecosystem/model/dataset/repetition;
            # only the directory name is new information.
            frame = pd.read_csv(path)
            frame.insert(0, "run", ident["run"])
            metrics.append(frame)

        path = run_dir / "manifest.json"
        if path.exists():
            record = flatten(json.loads(path.read_text()))
            manifests.append({**ident, **record})

    sort_by = ["ecosystem", "model", "dataset", "repetition"]
    return {
        "codecarbon": pd.concat(codecarbon, ignore_index=True)
        .sort_values(sort_by + ["phase", "epoch"], kind="stable")
        .reset_index(drop=True),
        "counters": pd.concat(counters, ignore_index=True)
        .sort_values(sort_by + ["phase", "epoch"], kind="stable")
        .reset_index(drop=True),
        "metrics": pd.concat(metrics, ignore_index=True)
        .sort_values(sort_by + ["epoch"], kind="stable")
        .reset_index(drop=True),
        "manifests": pd.DataFrame(manifests).sort_values(sort_by, kind="stable")
        .reset_index(drop=True),
    }


def write(tables: dict[str, pd.DataFrame]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, frame in tables.items():
        path = OUT / f"{name}.csv.gz"
        # mtime=0 so the same data produces the same bytes, and the checksum
        # below means something across machines.
        with gzip.GzipFile(path, "wb", mtime=0) as fh:
            frame.to_csv(fh, index=False, lineterminator="\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
        print(f"  {path.name:<20} {len(frame):>7,} rows  "
              f"{path.stat().st_size / 1e6:>5.1f} MB")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def check(tables: dict[str, pd.DataFrame]) -> int:
    """Re-derive the package in memory and compare it to what is on disk."""
    failures = 0
    for name, frame in tables.items():
        path = OUT / f"{name}.csv.gz"
        if not path.exists():
            print(f"  MISSING  {path.name}")
            failures += 1
            continue
        on_disk = pd.read_csv(path)
        if len(on_disk) != len(frame):
            print(f"  STALE    {path.name}: {len(on_disk):,} rows on disk, "
                  f"{len(frame):,} in {RAW.name}")
            failures += 1
        elif list(on_disk.columns) != list(frame.columns):
            print(f"  STALE    {path.name}: column set differs")
            failures += 1
        else:
            print(f"  ok       {path.name} ({len(frame):,} rows)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the package against the raw tree, do not write")
    args = ap.parse_args()

    print(f"reading {RAW.relative_to(REPO_ROOT)} ...")
    tables = collect()
    if args.check:
        failures = check(tables)
        print("package is current" if not failures
              else f"{failures} table(s) out of date -- rerun without --check")
        return 1 if failures else 0
    write(tables)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
