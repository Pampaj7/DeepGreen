#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild run directories from the replication package.

``scripts/consolidate_raw.py`` flattens ``results/campaign_v2/`` into four
gzipped tables. This is the inverse: it reconstructs a run's directory from
those tables, byte-for-byte where the tables carry every field, so the package
is a round trip rather than a one-way export.

That property is not decorative. A calibration re-execution wrote over three of
the campaign's own run directories, because one of the two code paths that
choose an output directory ignored the run contract the other honoured. The raw
tree is not distributed and not in version control; the package is, and it is
what the campaign was restored from.

  python3 scripts/restore_from_replication.py --runs <slug> [<slug> ...]
  python3 scripts/restore_from_replication.py --all --dry-run

Existing directories are refused unless --force is given, so restoring cannot
itself become the thing it repairs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "results" / "replication"
CAMPAIGN = REPO_ROOT / "results" / "campaign_v2"

# Columns consolidate_raw.py added; everything else is the original file.
IDENTITY = ["run", "ecosystem", "model", "dataset", "repetition"]


def restore(run: str, tables: dict[str, pd.DataFrame], force: bool,
            dry_run: bool) -> bool:
    out = CAMPAIGN / run
    if out.exists() and not force and not dry_run:
        print(f"  refusing {run}: directory exists (use --force)")
        return False

    counters = tables["counters"][tables["counters"].run == run]
    codecarbon = tables["codecarbon"][tables["codecarbon"].run == run]
    metrics = tables["metrics"][tables["metrics"].run == run]
    manifests = tables["manifests"][tables["manifests"].run == run]
    if counters.empty:
        print(f"  {run}: not in the package")
        return False

    n_blocks = len(codecarbon)
    print(f"  {run}: {len(counters)} counter rows, {n_blocks} blocks, "
          f"{len(metrics)} metric rows")
    if dry_run:
        return True

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    counters.drop(columns=IDENTITY).to_csv(out / "counters.csv", index=False)
    if not metrics.empty:
        metrics.drop(columns=["run"]).to_csv(out / "metrics.csv", index=False)

    for _, row in codecarbon.iterrows():
        phase, epoch = row["phase"], int(row["epoch"])
        block = row.drop(labels=IDENTITY + ["phase", "epoch"])
        block.to_frame().T.to_csv(
            out / f"emissions_{phase}_epoch{epoch}.csv", index=False)

    if not manifests.empty:
        # The manifest was flattened with dotted keys; nest it back.
        flat = manifests.iloc[0].drop(labels=IDENTITY).dropna().to_dict()
        nested: dict = {}
        for key, value in flat.items():
            node = nested
            *parents, leaf = str(key).split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        (out / "manifest.json").write_text(json.dumps(nested, indent=2) + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="replace directories that already exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tables = {name: pd.read_csv(PACKAGE / f"{name}.csv.gz")
              for name in ("counters", "codecarbon", "metrics", "manifests")}
    runs = sorted(tables["counters"].run.unique()) if args.all else args.runs
    if not runs:
        print("nothing to restore: pass --runs <slug> ... or --all")
        return 2

    print(f"restoring {len(runs)} run(s) from "
          f"{PACKAGE.relative_to(REPO_ROOT)}/")
    ok = sum(restore(r, tables, args.force, args.dry_run) for r in runs)
    print(f"{ok} of {len(runs)} restored" if not args.dry_run
          else f"{ok} of {len(runs)} would be restored")
    return 0 if ok == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
