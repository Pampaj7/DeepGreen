#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resize every dataset to the training resolution once, offline, so no stack does it.

Seven ecosystems resize with four different implementations -- torchvision on
PIL, `tf.image.resize`, `tch::vision::image::resize`, DataVec's
`ImageRecordReader`, R's `transform_resize` -- and they do not agree. Measured
over the whole Tiny ImageNet test split, 30,720,000 values apiece:

    C++ / Java / R        sd 0.263919 - 0.263923
    Rust                  sd 0.258406
    PyTorch / TensorFlow  sd 0.256103 - 0.256110

The means agree to within 0.1%; only the variance moves. That is the signature
of a resampling filter rather than of different content, and on 600 images the
choice spans 0.2483 (bilinear, antialiased) to 0.2699 (nearest) -- 8.7%. It bites
only where an image is actually downsampled: Tiny ImageNet is 64x64 halved to
32x32, Fashion-MNIST is 28x28 enlarged, CIFAR-100 is already 32x32 and has never
had the problem.

Aligning four resamplers across four libraries would mean expressing the same
filter in each, and DataVec and R may not offer it. Resizing once, here, removes
the question: every stack then decodes a 32x32 PNG and its own resize is a
no-op, which is the position CIFAR-100 has been in all along.

The trade is that resizing leaves the measured path. The paper counts the data
pipeline as part of what it measures, and this takes one step out of it -- but
it takes out the one step the seven stacks were performing differently, and
decode, batching and host-to-device transfer all remain.

    python3 scripts/normalise_dataset_resolution.py            # -> data/*_png32
    python3 scripts/normalise_dataset_resolution.py --check    # verify only

Bilinear with antialiasing, which is what `torchvision.transforms.Resize` does
and therefore what the reference stacks were already getting. Written to a new
directory so the originals stay as provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
TARGET = (32, 32)
#: source -> destination, relative to data/
DATASETS = {
    "fashion_mnist_png": "fashion_mnist_png32",
    "cifar100_png": "cifar100_png32",
    "tiny_imagenet_png": "tiny_imagenet_png32",
}


def convert(src_root: Path, dst_root: Path, check: bool) -> tuple[int, int]:
    """Resize every image under `src_root` into `dst_root`. Returns (done, skipped)."""
    done = skipped = 0
    for src in sorted(src_root.rglob("*")):
        if src.is_dir():
            continue
        dst = dst_root / src.relative_to(src_root)
        if src.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            # Everything that is not an image travels verbatim. Tiny ImageNet
            # ships a classes.json that the C++ loader reads, and leaving it
            # behind made that stack the only one that could not start.
            if not check and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
            continue
        if check:
            if not dst.exists():
                print(f"  missing: {dst.relative_to(REPO)}")
                skipped += 1
            else:
                with Image.open(dst) as im:
                    if im.size != TARGET:
                        print(f"  wrong size {im.size}: {dst.relative_to(REPO)}")
                        skipped += 1
                    else:
                        done += 1
            continue
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            # convert() before resize: a palette or 1-bit source resampled in its
            # own mode is not the same picture, and Fashion-MNIST ships greyscale.
            im = im.convert("RGB")
            if im.size != TARGET:
                im = im.resize(TARGET, Image.BILINEAR)
            im.save(dst)
        done += 1
    return done, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the converted tree instead of writing it")
    ap.add_argument("--data", type=Path, default=REPO / "data")
    args = ap.parse_args()

    failures = 0
    for src_name, dst_name in DATASETS.items():
        src, dst = args.data / src_name, args.data / dst_name
        if not src.is_dir():
            print(f"{src_name}: missing at {src}", file=sys.stderr)
            failures += 1
            continue
        verb = "checking" if args.check else "converting"
        print(f"{verb} {src_name} -> {dst_name} ...", flush=True)
        done, skipped = convert(src, dst, args.check)
        if args.check:
            print(f"  {done:,} at {TARGET[0]}x{TARGET[1]}, {skipped:,} wrong or missing")
            failures += skipped
        else:
            print(f"  {done:,} written, {skipped:,} already present")

    if args.check:
        print("\n" + ("  every image is 32x32" if not failures
                      else f"  {failures:,} images are not"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
