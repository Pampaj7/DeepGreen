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

Every field is carried as text, because this is a copy and a copy should be
exact: parsing to float and writing back moves energy readings by one unit in
the last place and turns "0" into "0.0", which was four thousand files that
failed to round-trip for no reason at all. Consumers cast what they need.
``scripts/restore_from_replication.py --check-roundtrip`` proves it.

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
    """manifest.json nests three levels deep; the table wants one.

    Lossy on purpose, for readability: a list becomes a space-joined string and
    a value's type is whatever pandas infers for its column, so "1000" in a
    column of numbers comes back as 1000.0 -- and an environment variable cannot
    be a float. The exact record is kept alongside in `manifest_json`, which is
    what restore_from_replication.py reads; these columns are for looking at.
    """
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
    metrics_columns: dict[str, str] = {}
    for run_dir in runs:
        ident = identity(run_dir)

        for path in sorted(run_dir.glob("emissions_*.csv")):
            phase, _, epoch = path.stem[len("emissions_"):].partition("_epoch")
            # As text. These files are the estimator's own output, copied
            # verbatim -- nothing here does arithmetic on them, and parsing them
            # as numbers loses the difference between "0" and "0.0", which is
            # four thousand files that fail to round-trip for no reason.
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.insert(0, "row_index", range(len(frame)))
            frame.insert(0, "epoch", int(epoch))
            frame.insert(0, "phase", phase)
            for column, value in reversed(ident.items()):
                frame.insert(0, column, value)
            codecarbon.append(frame)

        path = run_dir / "counters.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            # The tables below are sorted for readability, which reorders
            # counters.csv out of execution order -- train,1 / eval,1 / train,2
            # becomes every eval then every train. This is how a run gets back
            # the order it was written in.
            frame.insert(0, "row_index", range(len(frame)))
            for column, value in reversed(ident.items()):
                frame.insert(0, column, value)
            counters.append(frame)

        path = run_dir / "metrics.csv"
        if path.exists():
            # metrics.csv already carries ecosystem/model/dataset/repetition;
            # only the directory name is new information.
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.insert(0, "row_index", range(len(frame)))
            frame.insert(0, "run", ident["run"])
            # Which columns this run actually had. The concatenated table takes
            # the union, so a stack that never wrote `precision` would get an
            # empty one back on restore.
            metrics_columns[ident["run"]] = ",".join(
                c for c in frame.columns if c not in ("run", "row_index"))
            metrics.append(frame)

        path = run_dir / "manifest.json"
        if path.exists():
            raw = path.read_text()
            record = flatten(json.loads(raw))
            manifests.append({**ident, **record,
                              # The lossless copy. Everything above is a view.
                              "manifest_json": raw,
                              "metrics_columns": metrics_columns.get(ident["run"], "")})

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
        # below means something across machines. Reading uses
        # float_precision="round_trip": pandas' default parser is fast and not
        # exactly round-tripping, and it moved energy readings by one unit in
        # the last place -- 1.5169734358444487e-05 came back as ...488e-05.
        with gzip.GzipFile(path, "wb", mtime=0) as fh:
            # %.17g round-trips a float64 exactly; pandas' default drops the
            # last digit or two, so a restored file differed from its original
            # in the sixteenth significant figure of every energy reading.
            # No float_format: pandas writes the shortest representation that
            # round-trips, which is what the original files contain. %.17g
            # writes 3.4290280000000002 where the source says 3.429028 --
            # the same float64, and a different file.
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
