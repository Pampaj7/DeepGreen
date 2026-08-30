#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared image-folder input pipeline for the TensorFlow and JAX ecosystems.

Spec S3 requires one data-loader parallelism setting across every ecosystem,
because the audit showed the pipeline dominates the measurement: the GPU runs at
24-56% of its power limit and epoch duration accounts for essentially the whole
energy spread, while loader threads ranged from 0 (R) to 96 (Rust, rayon over
all cores) with Spearman -0.73 against duration.

PyTorch, C++ and Java each expose a worker count, so pinning them to 2 was a
one-line change. TensorFlow and JAX did not: both used
``ImageDataGenerator.flow_from_directory``, which decodes with PIL in the
calling thread. Their effective concurrency was neither 2 nor knowable from the
source, which is why the first version of the protocol had to list it as a
manual check.

This module replaces that generator with an explicit ``tf.data`` pipeline whose
parallelism is a stated number, so the same value applies to all ecosystems and
the specification is enforceable rather than aspirational.

Preprocessing matches the other stacks exactly: decode, resize to ``img_size``,
replicate grayscale to three channels, scale to ``[0, 1]``, no mean/std
normalisation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: Loader worker threads. Must equal the value used by every other ecosystem.
DEFAULT_NUM_WORKERS = int(os.environ.get("DEEPGREEN_LOADER_THREADS", "2"))


@dataclass
class FolderLoader:
    """A tf.data pipeline over an ImageFolder-style directory.

    Exposes the handful of attributes the existing training code reads from a
    Keras generator (``samples``, ``class_indices``), so call sites do not have
    to change shape.
    """

    dataset: "object"          # tf.data.Dataset yielding (x, y_onehot)
    samples: int
    class_indices: dict[str, int]
    batch_size: int
    num_workers: int

    @property
    def num_classes(self) -> int:
        return len(self.class_indices)

    @property
    def steps_per_epoch(self) -> int:
        return self.samples // self.batch_size

    def __iter__(self):
        return iter(self.dataset)

    def as_numpy(self) -> Iterator:
        """Numpy batches, for the JAX stack."""
        return self.dataset.as_numpy_iterator()


def _class_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def folder_loader(
    directory: str | Path,
    img_size: tuple[int, int] = (32, 32),
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    class_names: list[str] | None = None,
    one_hot: bool = True,
) -> FolderLoader:
    import tensorflow as tf

    directory = Path(directory)
    if class_names is None:
        class_names = _class_names(directory)
    class_indices = {name: i for i, name in enumerate(class_names)}

    paths: list[str] = []
    labels: list[int] = []
    for name in class_names:
        class_dir = directory / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class directory: {class_dir}")
        # rglob, not iterdir: a class label containing a path separator (Fashion-MNIST
        # class 0 is "T-shirt/top") produces a nested directory, and a loader that
        # lists only direct children sees that class as empty. torchvision's
        # ImageFolder and Keras' flow_from_directory both walk recursively, so
        # listing only direct children would make this stack disagree with the others
        # about the contents of the dataset.
        found = sorted(
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not found:
            raise FileNotFoundError(f"class directory has no images: {class_dir}")
        for f in found:
            paths.append(str(f))
            labels.append(class_indices[name])

    n_classes = len(class_names)
    height, width = img_size

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [height, width])
        img = tf.cast(img, tf.float32) / 255.0        # [0,1], no mean/std
        img.set_shape([height, width, 3])
        y = tf.one_hot(label, n_classes) if one_hot else label
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    # The single knob this module exists for.
    ds = ds.map(_load, num_parallel_calls=num_workers, deterministic=True)
    # Keep the final partial batch. This was drop_remainder=True, and with
    # steps computed as samples // batch_size it meant TensorFlow and JAX took
    # fewer gradient steps per epoch than the other five stacks and evaluated
    # on 9,984 of 10,000 test images -- always the same 16, since the test
    # loader is unshuffled over a sorted file list. Recoverable from the
    # recorded accuracies alone: test_acc * N / 100 is an integer only for
    # N = 9,984 in those two stacks and only for N = 10,000 in the rest.
    # Accuracy is the denominator of this study's energy-per-unit-quality
    # metric, so the two stacks were being scored on a different test set.
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(1)
    # Options are set explicitly so tf.data's autotuning cannot silently give
    # this stack more parallelism than the others.
    options = tf.data.Options()
    options.autotune.enabled = False
    options.threading.private_threadpool_size = num_workers
    options.deterministic = True
    ds = ds.with_options(options)

    return FolderLoader(
        dataset=ds,
        samples=len(paths),
        class_indices=class_indices,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def train_test_loaders(
    dataset_path: str | Path,
    img_size: tuple[int, int] = (32, 32),
    batch_size: int = 128,
    seed: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    one_hot: bool = True,
) -> tuple[FolderLoader, FolderLoader, int]:
    """Train and test loaders sharing one class order."""
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    test_dir = dataset_path / ("test" if (dataset_path / "test").is_dir() else "val")

    names = _class_names(train_dir)
    train = folder_loader(train_dir, img_size, batch_size, shuffle=True,
                          seed=seed, num_workers=num_workers,
                          class_names=names, one_hot=one_hot)
    test = folder_loader(test_dir, img_size, batch_size, shuffle=False,
                         seed=seed, num_workers=num_workers,
                         class_names=names, one_hot=one_hot)
    return train, test, len(names)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/cifar100_png"
    tr, te, n = train_test_loaders(root)
    print(f"{root}: {n} classes, train {tr.samples} / test {te.samples}, "
          f"{tr.num_workers} worker threads, {tr.steps_per_epoch} steps/epoch")
    x, y = next(iter(tr))
    print("batch:", x.shape, x.dtype, "range", float(x.numpy().min()), "-", float(x.numpy().max()),
          "| labels", y.shape)
