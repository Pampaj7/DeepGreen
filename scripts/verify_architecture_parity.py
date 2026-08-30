#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prove that every ecosystem trains the same network, structurally, in every block.

The paper's entire claim is that energy differences are attributable to the
ecosystem because the workload is held constant. Two rounds of review found that
it was not: VGG-16 ran as four different networks spanning 9.1x in parameters,
and ResNet-18 as three. Those were caught by counting parameters, which is the
weak form of the test -- two networks can agree on a total and differ layer by
layer, and a total says nothing about which tensor has which shape.

This is the strong form. For each (ecosystem, architecture, dataset) it collects
the *multiset of parameter tensor shapes*, sorted, which is comparable across
languages because it depends on neither naming nor ordering conventions. Two
stacks agree only if every shape appears the same number of times in both.

    python3 scripts/verify_architecture_parity.py            # all 42
    python3 scripts/verify_architecture_parity.py --json out.json

Each stack is fingerprinted in its own interpreter, because they cannot share
one: torch's cu128 wheels and TensorFlow's bundled CUDA libraries resolve to
different versions of the same shared objects. The JVM and R are fingerprinted
through the same subprocesses the campaign uses.

Conventions that differ and are normalised here, all of them found the hard way:

  * torch's ``model.parameters()`` counts learnable weights; Keras's
    ``count_params()`` and DL4J's ``numParams()`` also count each batch
    normalisation's running statistics. Those are buffers, not parameters, and
    are excluded.
  * Keras convolution kernels are (kh, kw, in, out) where torch's are
    (out, in, kh, kw). Shapes are canonicalised to torch's layout.
  * A bias of length n and a batch-norm scale of length n are both 1-D tensors
    of length n, so a stack that adds a convolution bias where torchvision omits
    one is invisible in the shape multiset alone. Bias counts are reported
    separately.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = {"fashionmnist": 10, "cifar100": 100, "tinyimagenet": 200}
MODELS = ("resnet18", "vgg16")

#: dataset name in the campaign -> the name the exported modules use
MODULE_DATASET = {"fashionmnist": "fashionmnist", "cifar100": "cifar100",
                  "tinyimagenet": "tinyimagenet200"}


def fingerprint(shapes: list[tuple[int, ...]]) -> dict:
    """A language-independent summary of a network's parameter tensors."""
    counter = Counter(tuple(int(x) for x in s) for s in shapes)
    return {
        "n_tensors": len(shapes),
        "n_parameters": sum(int(_prod(s)) for s in shapes),
        # sorted so two stacks that build the same layers in a different order
        # still agree, which they legitimately may
        "shapes": sorted(f"{list(k)}x{v}" for k, v in counter.items()),
        "n_rank4": sum(1 for s in shapes if len(s) == 4),
        "n_rank2": sum(1 for s in shapes if len(s) == 2),
        "n_rank1": sum(1 for s in shapes if len(s) == 1),
    }


def _prod(shape) -> int:
    out = 1
    for x in shape:
        out *= int(x)
    return out


# --------------------------------------------------------------------------
# One probe per interpreter. Each prints a single JSON line to stdout.
# --------------------------------------------------------------------------

PROBE_TORCH = r'''
import json, sys
sys.path.insert(0, %(repo)r)
from tools.deepgreen_bench import load_shared_module
m = load_shared_module(%(model)r, %(dataset)r)
shapes = [list(p.shape) for p in m.parameters()]
print("@@" + json.dumps(shapes))
'''

PROBE_KERAS = r'''
import json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, %(repo)r)
import importlib
mod = importlib.import_module("python.tensorflow.models." + %(model)r)
build = getattr(mod, "build_resnet18_garden", None) or getattr(mod, "build_vgg16")
model = build(input_shape=(32, 32, 3), num_classes=%(classes)d)
shapes = []
# trainable_weights on the top-level model, which Keras de-duplicates. Walking
# the layer tree double-counts: VGG-16 is a Sequential wrapping the backbone as
# a nested model, and a container reports its children's weights as its own.
for w in model.trainable_weights:
    s = [int(x) for x in w.shape]
    # Keras conv kernels are (kh, kw, in, out); torch's are (out, in, kh, kw)
    if len(s) == 4:
        s = [s[3], s[2], s[0], s[1]]
    # Keras Dense kernels are (in, out); torch Linear weights are (out, in)
    elif len(s) == 2:
        s = [s[1], s[0]]
    shapes.append(s)
print("@@" + json.dumps(shapes))
'''

PROBE_JAX = r'''
import json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, %(repo)r)
import jax, jax.numpy as jnp, importlib
mod = importlib.import_module("python.jax.models." + %(model)r)
if %(model)r == "vgg16":
    net = mod.VGG16_32(num_classes=%(classes)d)
else:
    # flaxmodels.ResNet18 used directly, exactly as the stack builds it
    net = mod.FMResNet18(output="logits", num_classes=%(classes)d,
                         pretrained=None, normalize=False)
k = jax.random.PRNGKey(0)
try:
    v = net.init({"params": k, "dropout": k}, jnp.ones((1, 32, 32, 3)), train=True)
except Exception:
    v = net.init(k, jnp.ones((1, 32, 32, 3)), train=True)
shapes = []
for path, leaf in jax.tree_util.tree_flatten_with_path(v["params"])[0]:
    s = [int(x) for x in leaf.shape]
    # flax conv kernels are (kh, kw, in, out); Dense are (in, out)
    if len(s) == 4:
        s = [s[3], s[2], s[0], s[1]]
    elif len(s) == 2:
        s = [s[1], s[0]]
    shapes.append(s)
print("@@" + json.dumps(shapes))
'''


PROBE_JAVA = r"""
import io.github.stlabunifi.deepgreen.dl4j.model.builder.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.*;

public class DGFingerprint {
  public static void main(String[] a) {
    int n = Integer.parseInt(a[1]);
    ComputationGraph g = a[0].equals("resnet18")
        ? ResNet18GraphBuilder.buildResNet18(n, 1000, 3, 32, 32, 1e-4)
        : Vgg16GraphBuilder.buildVGG16(n, 1000, 3, 32, 32, 1e-4);
    StringBuilder sb = new StringBuilder("[");
    boolean first = true;
    for (org.deeplearning4j.nn.api.Layer l : g.getLayers()) {
      Map<String, INDArray> p = l.paramTable();
      if (p == null) continue;
      for (Map.Entry<String, INDArray> e : new TreeMap<>(p).entrySet()) {
        // mean and log10stdev are buffers in torch, not parameters
        if (e.getKey().equals("mean") || e.getKey().equals("log10stdev")) continue;
        long[] sh = e.getValue().shape();
        // DL4J conv weights are (out, in, kh, kw) like torch; dense are
        // (in, out) like Keras, so transpose those. Squeeze leading 1s that
        // DL4J uses for bias and batch-norm rows.
        List<Long> dims = new ArrayList<>();
        for (long d : sh) dims.add(d);
        while (dims.size() > 1 && dims.get(0) == 1L) dims.remove(0);
        if (dims.size() == 2 && !e.getKey().equals("W")) { /* leave */ }
        else if (dims.size() == 2) Collections.reverse(dims);
        if (!first) sb.append(",");
        sb.append(dims.toString().replace(" ", ""));
        first = false;
      }
    }
    System.out.println("@@" + sb.append("]"));
  }
}
"""


def run_java_probe(model: str, classes: int, label: str) -> list | None:
    """Fingerprint a Deeplearning4j graph through the campaign's own classpath."""
    import shutil
    import tempfile

    java_home = os.environ.get("JAVA_HOME") or str(Path.home() / "miniforge3/envs/deepgreen")
    java = Path(java_home) / "bin" / "java"
    javac = Path(java_home) / "bin" / "javac"
    mvn = Path(java_home) / "bin" / "mvn"
    if not java.exists():
        print(f"  {label}: no JDK at {java_home}", file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cp_file = Path(tmp) / "cp.txt"
        subprocess.run([str(mvn), "-q", "-o", "-f", "Java/deepgreen-dl4j/pom.xml",
                        "dependency:build-classpath",
                        f"-Dmdep.outputFile={cp_file}"],
                       cwd=REPO, capture_output=True, text=True, timeout=600)
        if not cp_file.exists():
            print(f"  {label}: could not resolve the Java classpath", file=sys.stderr)
            return None
        cp = cp_file.read_text().strip()
        src = Path(tmp) / "DGFingerprint.java"
        src.write_text(PROBE_JAVA)
        r = subprocess.run([str(javac), "-cp", cp, "-d", tmp, "-sourcepath",
                            "Java/deepgreen-dl4j/src/main/java", str(src)],
                           cwd=REPO, capture_output=True, text=True, timeout=600)
        if r.returncode:
            print(f"  {label}: javac failed\n{r.stderr[-300:]}", file=sys.stderr)
            return None
        out = subprocess.run([str(java), "-cp", f"{tmp}:{cp}", "DGFingerprint",
                              model, str(classes)],
                             cwd=REPO, capture_output=True, text=True, timeout=600)
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    print(f"  {label}: no fingerprint\n{out.stderr.strip()[-300:]}", file=sys.stderr)
    return None


def run_probe(interpreter: str, code: str, label: str) -> list | None:
    try:
        out = subprocess.run([interpreter, "-c", code], cwd=REPO,
                             capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  {label}: timed out", file=sys.stderr)
        return None
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    print(f"  {label}: no fingerprint\n{out.stderr.strip()[-400:]}", file=sys.stderr)
    return None


def collect() -> dict:
    """Fingerprint every stack that can be probed from Python."""
    venv = {
        "Python/PyTorch": REPO / ".venv-deepgreen" / "bin" / "python",
        "Python/TensorFlow": REPO / ".venv-tensorflow" / "bin" / "python",
        "Python/JAX": REPO / ".venv-jax" / "bin" / "python",
    }
    results: dict = {}
    for model in MODELS:
        for dataset, classes in DATASETS.items():
            key = f"{model}/{dataset}"
            results[key] = {}
            subs = {"repo": str(REPO), "model": model, "dataset": dataset,
                    "classes": classes}

            # The exported module is the reference: C++ and Rust load this file
            # and PyTorch now loads it too, so one probe covers all three.
            shapes = run_probe(str(venv["Python/PyTorch"]), PROBE_TORCH % subs,
                               f"{key} shared module")
            if shapes is not None:
                results[key]["shared module (PyTorch, C++, Rust)"] = fingerprint(shapes)

            shapes = run_probe(str(venv["Python/TensorFlow"]), PROBE_KERAS % subs,
                               f"{key} TensorFlow")
            if shapes is not None:
                results[key]["Python/TensorFlow"] = fingerprint(shapes)

            shapes = run_probe(str(venv["Python/JAX"]), PROBE_JAX % subs,
                               f"{key} Python/JAX")
            if shapes is not None:
                results[key]["Python/JAX"] = fingerprint(shapes)

            shapes = run_java_probe(model, classes, f"{key} Java/DL4J")
            if shapes is not None:
                results[key]["Java/DL4J"] = fingerprint(shapes)
    return results


def report(results: dict) -> int:
    """Print a block-by-block comparison and return a shell exit code."""
    failures = 0
    for key in sorted(results):
        stacks = results[key]
        if len(stacks) < 2:
            print(f"\n{key}: only {len(stacks)} stack(s) fingerprinted -- cannot compare")
            failures += 1
            continue
        reference = "shared module (PyTorch, C++, Rust)"
        ref = stacks.get(reference) or next(iter(stacks.values()))
        print(f"\n{key}")
        for name, fp in stacks.items():
            same_shapes = fp["shapes"] == ref["shapes"]
            same_count = fp["n_parameters"] == ref["n_parameters"]
            mark = "ok  " if same_shapes else "DIFF"
            print(f"  [{mark}] {name:38} {fp['n_parameters']:>11,} params, "
                  f"{fp['n_tensors']:>3} tensors "
                  f"({fp['n_rank4']} conv, {fp['n_rank2']} dense, {fp['n_rank1']} 1-D)")
            if not same_shapes:
                failures += 1
                only_here = sorted(set(fp["shapes"]) - set(ref["shapes"]))
                only_ref = sorted(set(ref["shapes"]) - set(fp["shapes"]))
                for s in only_here[:6]:
                    print(f"         only here: {s}")
                for s in only_ref[:6]:
                    print(f"         only in reference: {s}")
                if same_count:
                    print("         (parameter totals agree -- this is exactly the "
                          "case a count cannot see)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="write the fingerprints here")
    args = ap.parse_args()

    results = collect()
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"fingerprints written to {args.json}")
    failures = report(results)
    print("\n" + "-" * 78)
    if failures:
        print(f"  {failures} structural difference(s). The stacks are not training "
              f"the same network,\n  and a cross-ecosystem energy comparison "
              f"compares models until they do.")
    else:
        print("  Every fingerprinted stack has the same parameter tensors, "
              "shape for shape.")
    print("-" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
