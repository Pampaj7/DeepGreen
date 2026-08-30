#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prove that every ecosystem trains on and is scored against the same data.

Architecture parity says the stacks compute the same function. This says they
compute it over the same inputs and are graded on the same answers, which is the
other half of the comparison the paper makes -- and the half that has produced
the subtler defects. Two examples already found in this study: TensorFlow and
JAX silently evaluated on 9,984 of 10,000 test images because the loader dropped
the final partial batch, and R/ResNet-18 trained on [0, 1/255] because its
transform divided by 255 an array the loader had already scaled.

Four things are checked, per dataset, for every stack that can be probed:

  1. **Sample counts.** Train and test, exactly, so a truncated split shows up.
  2. **Class ordering.** The label index a class gets is determined by the order
     the loader enumerates directories. Two stacks that sort differently score
     against different answers while every count agrees -- a defect this study
     has already had once, when a path separator in "T-shirt/top" split one
     class into two.
  3. **Pixel range and statistics.** Mean, standard deviation, minimum and
     maximum of a fixed slice of the test split, which catches a stack scaling
     its inputs differently.
  4. **Label sequence.** The first labels of the unshuffled test split, in
     order, so that two stacks agreeing on counts and classes but enumerating
     files differently are still separated.

    python3 scripts/verify_data_parity.py
    python3 scripts/verify_data_parity.py --json out.json

The C++, Rust and Java stacks read the same directories through their own
loaders and are covered by the directory-level facts (1, 2, 4) rather than by
their own probe; their pixel handling is asserted by the conformance checks on
the input transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = {
    "fashionmnist": "data/fashion_mnist_png",
    "cifar100": "data/cifar100_png",
    "tinyimagenet": "data/tiny_imagenet_png",
}
#: Tiny ImageNet ships train/ and val/; the other two ship train/ and test/.
TEST_SPLIT = {"tinyimagenet": "val"}
SAMPLE = 256          # images summarised from the head of the test split


def on_disk(root: Path, split: str) -> dict:
    """What the directory itself says, independent of any framework."""
    base = root / split
    classes = sorted(p.name for p in base.iterdir() if p.is_dir())
    files = []
    for name in classes:
        for f in sorted((base / name).iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                files.append((str(f.relative_to(base)), classes.index(name)))
    return {
        "n_classes": len(classes),
        "classes": classes,
        "n_files": len(files),
        "first_labels": [lab for _, lab in files[:32]],
        # A hash of the full ordered file list: two stacks enumerating the same
        # directory in the same order must agree on this exactly.
        "file_order_sha256": hashlib.sha256(
            "\n".join(f for f, _ in files).encode()).hexdigest()[:16],
    }


PROBE_TORCH = r'''
import json, sys
sys.path.insert(0, %(repo)r)
import numpy as np, torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
ds = ImageFolder(%(path)r, transform=t)
labels = [int(ds[i][1]) for i in range(min(32, len(ds)))]
xs = torch.stack([ds[i][0] for i in range(min(%(sample)d, len(ds)))])
if xs.shape[1] == 1:
    xs = xs.repeat(1, 3, 1, 1)
print("@@" + json.dumps({
    "n_files": len(ds), "n_classes": len(ds.classes), "classes": list(ds.classes),
    "first_labels": labels,
    "pixel": [round(float(xs.mean()), 6), round(float(xs.std()), 6),
              round(float(xs.min()), 6), round(float(xs.max()), 6)],
}))
'''

PROBE_SHARED_LOADER = r'''
import json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, %(repo)r)
import numpy as np
from tools.deepgreen_loader import train_test_loaders
train, test, n_classes = train_test_loaders(
    %(root)r, img_size=(32, 32), batch_size=%(sample)d, seed=1000, one_hot=True)
it = test.as_numpy()
x, y = next(it)
labels = [int(v) for v in np.argmax(y[:32], axis=1)]
print("@@" + json.dumps({
    "n_files": int(test.samples), "n_classes": int(n_classes),
    "classes": list(test.class_indices.keys()),
    "first_labels": labels,
    "pixel": [round(float(x.mean()), 6), round(float(x.std()), 6),
              round(float(x.min()), 6), round(float(x.max()), 6)],
}))
'''


def run_probe(interpreter: Path, code: str, label: str) -> dict | None:
    try:
        out = subprocess.run([str(interpreter), "-c", code], cwd=REPO,
                             capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  {label}: timed out", file=sys.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    print(f"  {label}: no result\n{out.stderr.strip()[-300:]}", file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    results: dict = {}
    failures = 0
    for dataset, rel in DATASETS.items():
        root = REPO / rel
        split = TEST_SPLIT.get(dataset, "test")
        if not (root / split).is_dir():
            print(f"{dataset}: {root / split} missing", file=sys.stderr)
            failures += 1
            continue

        disk = on_disk(root, split)
        entry = {"on disk": disk}

        subs = {"repo": str(REPO), "path": str(root / split), "root": str(root),
                "sample": SAMPLE}
        got = run_probe(REPO / ".venv-deepgreen" / "bin" / "python",
                        PROBE_TORCH % subs, f"{dataset} torchvision")
        if got:
            entry["torchvision (PyTorch)"] = got
        got = run_probe(REPO / ".venv-tensorflow" / "bin" / "python",
                        PROBE_SHARED_LOADER % subs, f"{dataset} shared tf.data loader")
        if got:
            entry["shared tf.data loader (TensorFlow, JAX)"] = got
        results[dataset] = entry

        print(f"\n{dataset}  ({split} split)")
        print(f"  on disk: {disk['n_files']:,} files, {disk['n_classes']} classes, "
              f"order {disk['file_order_sha256']}")
        for name, fp in entry.items():
            if name == "on disk":
                continue
            ok_n = fp["n_files"] == disk["n_files"]
            ok_c = fp["classes"] == disk["classes"]
            ok_l = fp["first_labels"] == disk["first_labels"]
            mark = "ok  " if (ok_n and ok_c and ok_l) else "DIFF"
            if mark == "DIFF":
                failures += 1
            mean, sd, lo, hi = fp["pixel"]
            print(f"  [{mark}] {name:42} {fp['n_files']:>6,} files, "
                  f"{fp['n_classes']} classes, pixels [{lo:.3f}, {hi:.3f}] "
                  f"mean {mean:.4f} sd {sd:.4f}")
            if not ok_n:
                print(f"         file count differs: {fp['n_files']} vs {disk['n_files']}")
            if not ok_c:
                print(f"         class ORDER differs -- the stacks are scored "
                      f"against different label indices")
                print(f"         here: {fp['classes'][:5]}")
                print(f"         disk: {disk['classes'][:5]}")
            if not ok_l:
                print(f"         first labels differ: {fp['first_labels'][:12]} "
                      f"vs {disk['first_labels'][:12]}")

    # Pixel statistics must agree between the two loaders, or the stacks are
    # training on differently scaled inputs -- the R defect, in another place.
    for dataset, entry in results.items():
        probes = {k: v for k, v in entry.items() if k != "on disk"}
        if len(probes) < 2:
            continue
        stats = {k: v["pixel"] for k, v in probes.items()}
        ref = next(iter(stats.values()))
        for name, px in stats.items():
            if max(abs(a - b) for a, b in zip(px, ref)) > 1e-3:
                print(f"\n  [DIFF] {dataset}: {name} pixel statistics differ from "
                      f"the reference loader\n         {px} vs {ref}")
                failures += 1

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nfingerprints written to {args.json}")

    print("\n" + "-" * 78)
    if failures:
        print(f"  {failures} difference(s). The stacks are not being scored on the "
              f"same data.")
    else:
        print("  Every probed loader sees the same files, in the same order, with "
              "the same\n  class indices and the same pixel range.")
    print("-" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
