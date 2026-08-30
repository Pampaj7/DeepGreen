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
#: Added to preserve within-file ordering; dropped on the way back out.
ORDER = "row_index"


def _restore_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Put a run's rows back in the order they were written.

    The package is sorted for readability, which groups counters.csv by phase --
    every eval, then every train -- where the campaign wrote them interleaved.
    """
    if ORDER in frame.columns:
        frame = frame.sort_values(ORDER, key=lambda c: c.astype(int),
                                  kind="stable")
    return frame.drop(columns=[ORDER] if ORDER in frame.columns else [])


def restore(run: str, tables: dict[str, pd.DataFrame], force: bool,
            dry_run: bool, out_root: Path | None = None) -> bool:
    out = (out_root or CAMPAIGN) / run
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

    _restore_order(counters.drop(columns=IDENTITY, errors="ignore")).to_csv(
        out / "counters.csv", index=False)
    if not metrics.empty:
        frame = _restore_order(metrics.drop(columns=["run"], errors="ignore"))
        # Only the columns this run actually had. The package holds the union
        # across stacks, so restoring all of them hands a run an empty column
        # it never wrote.
        want = ""
        if not manifests.empty and "metrics_columns" in manifests.columns:
            want = str(manifests.iloc[0].get("metrics_columns") or "")
        if want:
            cols = [c for c in want.split(",") if c in frame.columns]
            if cols:
                frame = frame[cols]
        frame.to_csv(out / "metrics.csv", index=False)

    for (phase, epoch), block in codecarbon.groupby(["phase", "epoch"], sort=False):
        frame = _restore_order(
            block.drop(columns=IDENTITY + ["phase", "epoch"], errors="ignore"))
        frame.to_csv(out / f"emissions_{phase}_epoch{int(epoch)}.csv",
                     index=False)

    if not manifests.empty:
        raw = manifests.iloc[0].get("manifest_json")
        if isinstance(raw, str) and raw.strip():
            # The exact bytes. Rebuilding from the dotted columns loses types --
            # a seed of "1000" comes back as 1000.0, and an environment variable
            # cannot be a float -- and flattens every list to a joined string.
            (out / "manifest.json").write_text(raw)
        else:
            flat = manifests.iloc[0].drop(
                labels=[c for c in IDENTITY + ["manifest_json", "metrics_columns"]
                        if c in manifests.columns]).dropna().to_dict()
            nested: dict = {}
            for key, value in flat.items():
                node = nested
                *parents, leaf = str(key).split(".")
                for part in parents:
                    node = node.setdefault(part, {})
                node[leaf] = value
            (out / "manifest.json").write_text(json.dumps(nested, indent=2) + "\n")
    return True


def check_roundtrip(tables: dict[str, pd.DataFrame]) -> int:
    """Restore every run to a temporary tree and diff it against the raw one.

    consolidate_raw.py --check verifies raw -> package. Nothing verified
    package -> raw, and it did not hold: 13,037 of roughly 13,230 files differed,
    from float truncation in the sixteenth digit, manifest values silently
    retyped, lists joined into strings, and counters.csv handed back sorted by
    phase rather than in execution order.
    """
    import filecmp
    import tempfile

    runs = sorted(tables["counters"].run.unique())
    same = differ = missing = eol = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for run in runs:
            restore(run, tables, force=True, dry_run=False, out_root=root)
        for run in runs:
            original, restored = CAMPAIGN / run, root / run
            if not original.is_dir():
                continue
            for f in sorted(original.iterdir()):
                other = restored / f.name
                if not other.exists():
                    missing += 1
                    print(f"  missing: {run}/{f.name}")
                elif filecmp.cmp(f, other, shallow=False):
                    same += 1
                elif (f.read_bytes().replace(b"\r\n", b"\n")
                      == other.read_bytes().replace(b"\r\n", b"\n")):
                    # Python's csv module defaults to the excel dialect, which
                    # is CRLF; everything that reads these files back writes LF.
                    # The harness writes LF now, so this can only appear on runs
                    # recorded before that -- same content, different bytes.
                    eol += 1
                else:
                    differ += 1
                    if differ <= 5:
                        print(f"  differs: {run}/{f.name}")
    total = same + differ + missing + eol
    print(f"\n  {same:,} of {total:,} files identical, {differ:,} differ, "
          f"{missing:,} missing")
    if eol:
        print(f"  {eol:,} identical apart from line terminators, written before "
              f"the harness was\n  fixed to emit LF; a campaign recorded since "
              f"round-trips byte for byte.")
    return 0 if differ == 0 and missing == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="replace directories that already exist")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path,
                    help="restore here instead of results/campaign_v2, so a "
                         "round trip can be checked without overwriting the "
                         "campaign -- the failure this script exists to repair")
    ap.add_argument("--check-roundtrip", action="store_true",
                    help="restore to a temporary directory and diff against "
                         "the raw tree, file by file")
    args = ap.parse_args()

    # Text throughout, as the package stores it. Parsing to float and writing
    # back is not the identity: it moves a reading by one unit in the last place
    # and turns "0" into "0.0".
    tables = {name: pd.read_csv(PACKAGE / f"{name}.csv.gz", dtype=str,
                                keep_default_na=False)
              for name in ("counters", "codecarbon", "metrics", "manifests")}
    if args.check_roundtrip:
        return check_roundtrip(tables)

    runs = sorted(tables["counters"].run.unique()) if args.all else args.runs
    if not runs:
        print("nothing to restore: pass --runs <slug> ... or --all")
        return 2

    print(f"restoring {len(runs)} run(s) from "
          f"{PACKAGE.relative_to(REPO_ROOT)}/")
    ok = sum(restore(r, tables, args.force, args.dry_run, args.out_dir)
             for r in runs)
    print(f"{ok} of {len(runs)} restored" if not args.dry_run
          else f"{ok} of {len(runs)} would be restored")
    return 0 if ok == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
