#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify every ecosystem against results/analysis/experiment_spec.md.

The first campaign had no such check, and the eight stacks silently drifted
apart: two learning rates, four different loader-parallelism settings, one stack
normalising its inputs and three different LibTorch builds behind the group the
paper calls a shared backend.

    python3 scripts/check_consistency.py            # report
    python3 scripts/check_consistency.py --strict   # exit 1 on any FAIL

Checks are grep-level: they read the source that the campaign actually runs.
They cannot see runtime values, so the spec's "manual checks" section still
applies.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Result:
    ecosystem: str
    check: str
    status: str
    detail: str


def read(rel: str) -> str:
    p = REPO / rel
    return p.read_text(errors="replace") if p.exists() else ""


def glob_read(*patterns: str, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """Read every file matching any of `patterns`, minus `exclude`d basenames.

    Several patterns rather than brace expansion, because pathlib does not
    expand braces and would silently match nothing -- which in a checker turns
    into a SKIP that reads like a pass.

    `exclude` is for files that match a glob but carry none of the behaviour it
    is about -- a Rust `mod.rs` that only re-exports, say. Use it sparingly: a
    checker whose globs are narrowed until they pass is the thing this script
    exists to prevent.
    """
    out: dict[str, str] = {}
    for pattern in patterns:
        for p in sorted(REPO.glob(pattern)):
            if p.is_file() and p.name not in exclude:
                out[str(p.relative_to(REPO))] = p.read_text(errors="replace")
    return out


def strip_comments(files: dict[str, str]) -> dict[str, str]:
    """Blank out // and /* */ comments, keeping line numbers intact.

    A checker that matches inside comments cannot be used to document the very
    defect it guards against -- the first version of the per-binary-transform
    check failed on the comment explaining why the check exists.
    """
    out = {}
    for path, text in files.items():
        res, i, n = [], 0, len(text)
        while i < n:
            two = text[i:i + 2]
            if two == "//":
                j = text.find("\n", i)
                j = n if j == -1 else j
                res.append(" " * (j - i))
                i = j
            elif two == "/*":
                j = text.find("*/", i + 2)
                j = n if j == -1 else j + 2
                res.append("".join(c if c == "\n" else " " for c in text[i:j]))
                i = j
            else:
                res.append(text[i])
                i += 1
        out[path] = "".join(res)
    return out


def strip_hash_comments(files: dict[str, str]) -> dict[str, str]:
    """Blank out `#` comments in Python, R and shell, keeping line numbers.

    The counterpart of strip_comments for the languages that use `#`. Needed for
    the same reason: a check that documents the defect it guards against will
    match its own explanation. Three of the checks added here failed on their
    own comments the first time they ran.

    Quotes are tracked so a `#` inside a string is not mistaken for a comment.
    """
    out = {}
    for path, text in files.items():
        res, i, n, quote = [], 0, len(text), None
        while i < n:
            ch = text[i]
            if quote:
                res.append(ch)
                if ch == "\\" and i + 1 < n:
                    res.append(text[i + 1]); i += 2; continue
                if ch == quote:
                    quote = None
                i += 1
            elif ch in "\"'":
                quote = ch; res.append(ch); i += 1
            elif ch == "#":
                j = text.find("\n", i)
                j = n if j == -1 else j
                res.append(" " * (j - i))
                i = j
            else:
                res.append(ch); i += 1
        out[path] = "".join(res)
    return out


def check_regex(eco, name, files, pattern, expect=True, detail_ok="", flags=0,
                every=True):
    """PASS if `pattern` is found (expect=True) or absent (expect=False).

    The specification is universal -- "the learning rate is 1e-4 in every stack"
    -- so the default is universal too: with expect=True and every=True the check
    passes only when *every* file in the glob carries the pattern. An existential
    check ("some file in this glob says 1e-4") is not a conformance test: it
    cannot see a divergent sibling, and it will cite as evidence a file the
    campaign never executes. That is not hypothetical -- it is how a tree of
    unexecuted Java classes carrying lrAdam = 1e-3 sat under a passing check.

    Pass every=False only where the pattern legitimately lives in one file of a
    multi-file glob, and say why at the call site.
    """
    hits, missing = [], []
    for path, text in files.items():
        found_here = False
        for m in re.finditer(pattern, text, flags):
            line = text[: m.start()].count("\n") + 1
            hits.append(f"{path}:{line}")
            found_here = True
        if not found_here:
            missing.append(path)
    if not files:
        return Result(eco, name, SKIP, "no source files matched")

    if not expect:                      # universal by construction: no file may match
        if hits:
            return Result(eco, name, FAIL, "found at " + ", ".join(hits[:3]))
        return Result(eco, name, PASS, detail_ok or f"absent from {len(files)} file(s)")

    if every:
        if missing:
            return Result(eco, name, FAIL,
                          f"missing from {len(missing)} of {len(files)}: "
                          + ", ".join(sorted(missing)[:3]))
        return Result(eco, name, PASS,
                      detail_ok or f"in all {len(files)} file(s), e.g. {hits[0]}")

    if hits:
        return Result(eco, name, PASS, detail_ok or hits[0])
    return Result(eco, name, FAIL, "pattern not found")


def bench_src() -> str:
    return read("tools/deepgreen_bench.py")


def _check_rust_links_cuda() -> list[Result]:
    """Every released Rust binary must carry a NEEDED entry for torch_cuda."""
    import subprocess

    out: list[Result] = []
    binaries = sorted(p for p in (REPO / "rust" / "target" / "release").glob("*")
                      if p.is_file() and p.stat().st_mode & 0o111
                      and not p.suffix and p.name not in ("build",))
    if not binaries:
        return [Result("Rust/tch", "S4 binaries link CUDA", SKIP,
                       "no release build; run scripts/build_rust_cuda.sh")]
    missing = []
    for b in binaries:
        try:
            elf = subprocess.run(["readelf", "-d", str(b)], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception as exc:
            return [Result("Rust/tch", "S4 binaries link CUDA", FAIL,
                           f"{type(exc).__name__}: {exc}")]
        if "cuda" not in elf.lower():
            missing.append(b.name)
    if missing:
        out.append(Result("Rust/tch", "S4 binaries link CUDA", FAIL,
                          f"{len(missing)} of {len(binaries)} link no CUDA library: "
                          + ", ".join(missing[:4])
                          + " -- rebuild with scripts/build_rust_cuda.sh"))
    else:
        out.append(Result("Rust/tch", "S4 binaries link CUDA", PASS,
                          f"all {len(binaries)} carry a torch_cuda NEEDED entry"))
    return out


def _check_run_dir_contract() -> list[Result]:
    """Resolve both output paths with DEEPGREEN_RUN_DIR set, and see where they go.

    The campaign has two code paths that choose an output directory: the Python
    stacks resolve `RunContext.out_dir` in process, the other four go through
    `deepgreen_tracker.run_dir()`. They once disagreed -- the tracker honoured
    the run contract and the bench hardcoded the campaign directory -- and a
    calibration re-execution overwrote three of the campaign's own runs.

    So this asks the question that incident poses: with the variable set, does
    each path resolve *inside* it, and does neither fall back to the campaign
    directory? Anything short of executing them is a proxy for that question.
    """
    import importlib
    import os
    import tempfile

    out: list[Result] = []
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "run"
        env_before = os.environ.get("DEEPGREEN_RUN_DIR")
        sys.path.insert(0, str(REPO / "tools"))
        os.environ["DEEPGREEN_RUN_DIR"] = str(target)
        try:
            for name, resolve in (
                ("deepgreen_bench", lambda m: m.RunContext(
                    ecosystem="Python/PyTorch", model="resnet18",
                    dataset="fashionmnist", repetition=0).out_dir),
                ("deepgreen_tracker", lambda m: m.run_dir()),
            ):
                try:
                    mod = importlib.import_module(name)
                    importlib.reload(mod)
                    got = Path(resolve(mod)).resolve()
                    inside = got == target.resolve() or target.resolve() in got.parents
                    campaign = (REPO / "results" / "campaign_v2").resolve()
                    escaped = got == campaign or campaign in got.parents
                    if inside and not escaped:
                        out.append(Result("all", f"S5 {name} honours DEEPGREEN_RUN_DIR",
                                          PASS, "resolves inside the run contract"))
                    else:
                        out.append(Result("all", f"S5 {name} honours DEEPGREEN_RUN_DIR",
                                          FAIL, f"resolved to {got}"))
                except Exception as exc:
                    out.append(Result("all", f"S5 {name} honours DEEPGREEN_RUN_DIR",
                                      FAIL, f"{type(exc).__name__}: {exc}"))
        finally:
            if env_before is None:
                os.environ.pop("DEEPGREEN_RUN_DIR", None)
            else:
                os.environ["DEEPGREEN_RUN_DIR"] = env_before
            sys.path.remove(str(REPO / "tools"))

    # And the fallback is still the campaign directory when nothing is set,
    # so honouring the contract has not quietly changed the default.
    bench = read("tools/deepgreen_bench.py")
    out.append(Result("all", "S5 campaign_v2 is the fallback, not a second path",
                      PASS if bench.count('"campaign_v2"') == 1 else FAIL,
                      f'"campaign_v2" appears {bench.count(chr(34) + "campaign_v2" + chr(34))}x '
                      f"in deepgreen_bench.py"))
    return out


def run() -> list[Result]:
    r: list[Result] = []

    # ---------------- S2: learning rate ------------------------------------
    r.append(check_regex("Python/PyTorch", "S2 lr=1e-4",
                         glob_read("python/pytorch/models/*.py"), r"optim\.Adam\(.*lr=1e-4"))
    r.append(check_regex("Python/TensorFlow", "S2 lr=1e-4",
                         glob_read("python/tensorflow/models/*.py"), r"lr=1e-4"))
    r.append(check_regex("Python/JAX", "S2 lr=1e-4",
                         glob_read("python/jax/models/*.py"), r"learning_rate=1e-4"))
    r.append(check_regex("C++/LibTorch", "S2 lr=1e-4",
                         glob_read("cpp/src/train/**/train_model.h"), r"AdamOptions\(1e-4\)"))
    r.append(check_regex("Rust/tch", "S2 lr=1e-4",
                         glob_read("rust/src/bin/*.rs"), r"build\(&vs, 1e-4\)"))
    r.append(check_regex("Rust/tch", "S2 no stray 1e-3",
                         glob_read("rust/src/bin/*.rs"), r"build\(&vs, 1e-3\)", expect=False))
    r.append(check_regex("R/torch", "S2 lr=1e-4",
                         glob_read("R/models/*.r"), r"optim_adam\([^)]*lr = 1e-4"))
    r.append(check_regex("Java/DL4J", "S2 lr=1e-4",
                         glob_read("Java/**/expt/resnet18/*.java", "Java/**/expt/vgg16/*.java"), r"lrAdam\s*=\s*1e-4"))

    # ---------------- S3: loader parallelism -------------------------------
    r.append(check_regex("Python/PyTorch", "S3 loader workers = 2",
                         glob_read("python/pytorch/models/*.py"), r"num_workers=2"))
    r.append(check_regex("R/torch", "S3 loader workers = 2",
                         glob_read("R/models/*.r"), r"num_workers = 2"))
    r.append(check_regex("R/torch", "S3 no single-threaded loader",
                         glob_read("R/models/*.r"), r"num_workers = 0", expect=False))
    r.append(check_regex("C++/LibTorch", "S3 loader workers = 2",
                         glob_read("cpp/src/train/**/train_model.h"), r"\.workers\(2\)"))
    r.append(check_regex("Java/DL4J", "S3 loader workers = 2",
                         glob_read("Java/**/dataloader/PNGDataloader.java"),
                         r"AsyncDataSetIterator\([^,]+,\s*2\)"))
    r.append(check_regex("Rust/tch", "S3 loader pool bounded",
                         glob_read("rust/src/datasets/*.rs", exclude=("mod.rs",)), r"init_loader_pool\(\)"))

    # The input transform lives in the loader and nowhere else.
    #
    # This check exists because we broke a configuration ourselves. vgg_fashion.rs
    # kept a private copy of the transform, which was harmless while the loader
    # returned raw uint8. Fixing the loader to produce float [0,1] and switching
    # evaluation to batched inference were both correct changes; together they
    # left the private copy on the training path only, resizing an already-float
    # tensor with a function that returns uint8. Every value truncated to zero,
    # so the network trained on black images and was evaluated on real ones.
    # Nothing in the energy data showed it.
    r.append(check_regex("Rust/tch", "S3 no per-binary image transform",
                         strip_comments(glob_read("rust/src/bin/*.rs")),
                         r"image::resize|vision::image::resize|fn preprocess",
                         expect=False,
                         detail_ok="the loader owns the pipeline"))

    # S5 requires every stack to persist per-epoch quality, and "every" has to
    # be checked per binary. Two of the six Rust binaries printed accuracy to
    # stdout and never called log_metric, so their runs carried valid energy and
    # no quality at all -- and the analysis, which keys on metrics.csv, dropped
    # them from the *energy* tables too. Two of forty-two configurations went
    # missing from the results without anything failing.
    for _bin in ("resnet_cifar100", "resnet_fashion", "resnet_tiny",
                 "vgg_cifar100", "vgg_fashion", "vgg_tiny"):
        r.append(check_regex("Rust/tch", f"S5 {_bin} persists quality metrics",
                             glob_read(f"rust/src/bin/{_bin}.rs"),
                             r"log_metric\("))

    # ---------------- S3: input scaling ------------------------------------
    r.append(check_regex("Rust/tch", "S3 normalisation gated off by default",
                         glob_read("rust/src/datasets/*.rs", exclude=("mod.rs",)), r"crate::normalize_inputs\(\)"))
    r.append(check_regex("Rust/tch", "S3 default is raw [0,1]",
                         glob_read("rust/src/lib.rs"),
                         r'DEEPGREEN_NORMALIZE[\s\S]{0,120}?unwrap_or\(false\)'))

    # ---------------- S3: evaluation batching and input shape --------------
    rust_bins_nc = {k: re.sub(r"^\s*//.*$", "", v, flags=re.M)
                    for k, v in glob_read("rust/src/bin/*.rs").items()}
    r.append(check_regex("Rust/tch", "S3 eval batched (not one image at a time)",
                         rust_bins_nc,
                         r"test_data\.iter_batches\(1\)", expect=False,
                         detail_ok="batched"))
    r.append(check_regex("Rust/tch", "S3 Fashion-MNIST resized to 32x32x3 in the loader",
                         glob_read("rust/src/datasets/fashion.rs"),
                         r"resize\(&img, 32, 32\)"))
    r.append(check_regex("Rust/tch", "S3 Tiny ImageNet resized to 32",
                         glob_read("rust/src/bin/resnet_tiny.rs"), r"Some\(32\)"))

    # ---------------- S1: shared TorchScript module ------------------------
    cmake = read("cpp/CMakeLists.txt")
    export_on = bool(re.search(r"^\s*export_model_for_dataset\(", cmake, re.M))
    r.append(Result("C++/LibTorch", "S1 TorchScript export enabled",
                    PASS if export_on else FAIL,
                    "GenerateModel block active" if export_on
                    else "export_model_for_dataset is commented out; C++ falls back to a hand-written port"))
    imported_built = bool(re.search(r"^\s+resnet18_cifar100_imported", cmake, re.M))
    r.append(Result("C++/LibTorch", "S1 imported targets in TARGETS",
                    PASS if imported_built else FAIL,
                    "present" if imported_built else "imported targets not built"))
    ifh = (REPO / "cpp/src/dataset/ImageFolder.h").exists()
    r.append(Result("C++/LibTorch", "S1 single dataset loader entry point",
                    PASS if ifh else FAIL,
                    "cpp/src/dataset/ImageFolder.h" if ifh else "header missing; imported targets cannot compile"))
    hit = any(re.search(r"CModule::load|TrainableCModule", t)
              for t in glob_read("rust/src/**/*.rs").values())
    r.append(Result("Rust/tch", "S1 loads the shared TorchScript module",
                    PASS if hit else FAIL,
                    "yes" if hit else "still builds its own model"))

    # R/torch is a documented exception, not an oversight: in torch 0.17.0 a
    # script_module cannot be switched between train and eval, so loading the
    # shared module would force evaluation in training mode. The check verifies
    # the exception is still documented where a reader will meet it.
    r_documented = any("cannot use the shared TorchScript" in t
                       for t in glob_read("R/models/*.r").values())
    r.append(Result("R/torch", "S1 exception documented in the source",
                    PASS if r_documented else FAIL,
                    "builds its own model; parity with PyTorch does not hold here"
                    if r_documented else "undocumented deviation from S1"))

    # ---------------- S2: batch size and epochs ----------------------------
    for eco, files, pat in [
        ("Python/PyTorch", glob_read("python/pytorch/models/*.py"), r"batch_size=128"),
        ("Python/TensorFlow", glob_read("python/tensorflow/models/*.py"), r"batch_size=128"),
        ("Python/JAX", glob_read("python/jax/models/*.py"), r"batch_size=128"),
        ("Rust/tch", glob_read("rust/src/bin/*.rs"), r"let batch_size = 128"),
        ("R/torch", glob_read("R/models/*.r"), r"batch_size = 128"),
        ("Java/DL4J", glob_read("Java/**/expt/resnet18/*.java", "Java/**/expt/vgg16/*.java"), r"batchSize\s*=\s*128"),
        ("C++/LibTorch", glob_read("cpp/src/train/**/train_*.cpp"), r"kTrainBatchSize = 128"),
    ]:
        r.append(check_regex(eco, "S2 batch size = 128", files, pat))
    r.append(check_regex("C++/LibTorch", "S2 eval batch = train batch",
                         glob_read("cpp/src/train/**/train_*.cpp"), r"kTestBatchSize = 128"))

    # ---------------- S4: the Rust binaries actually link CUDA -------------
    #
    # `cargo build` against a CUDA LibTorch produces a binary that silently runs
    # on the CPU: torch-sys asks the linker for torch_cuda, nothing in the Rust
    # code references a symbol from it, and --as-needed drops the dependency.
    # tch::Device::cuda_if_available() then reports Cpu and the stack trains on
    # the host while looking entirely normal. scripts/build_rust_cuda.sh exists
    # to prevent exactly this and says so in its header -- and a rebuild with
    # plain cargo walked into it anyway, costing a day's measurements: 29x
    # slower on CIFAR-100, with the accelerator drawing 23 W against a 24 W
    # idle floor. Nothing checked the built artefact, so nothing noticed.
    r.extend(_check_rust_links_cuda())

    # ---------------- S4: one precision policy, all seven stacks -----------
    #
    # The defect this replaces was invisible to every check that existed. The
    # TF32 pin lived in tools/deepgreen_bench.py, which only the three Python
    # stacks import, so Python/PyTorch ran true-fp32 convolutions and the other
    # six took cuDNN's Ampere default -- worth 4.81x the GPU energy on
    # ResNet-18, against a measured PyTorch-vs-C++ gap of 3.24-3.79x. The
    # checker had a `precision` clause in the spec and no check behind it.
    env_sh = read("scripts/campaign_env.sh")
    r.append(Result("all", "S4 precision policy is campaign-wide",
                    PASS if "NVIDIA_TF32_OVERRIDE" in env_sh else FAIL,
                    "scripts/campaign_env.sh exports it to every stack"))
    r.append(check_regex("all", "S4 harness applies the precision policy",
                         glob_read("tools/deepgreen_bench.py"),
                         r"def set_precision_policy", every=False,
                         detail_ok="tools/deepgreen_bench.py"))
    # No stack may set TF32 privately: one policy, one place. The harness's own
    # definition is excluded because it *is* the one place.
    r.append(check_regex("all", "S4 no stack pins TF32 privately",
                         glob_read("python/**/*.py", "rust/src/**/*.rs",
                                   "cpp/src/**/*.h", "cpp/src/**/*.cpp",
                                   "R/**/*.r", "Java/**/*.java"),
                         r"allow_tf32|setAllowTF32|enable_tensor_float_32",
                         expect=False))
    # cudnn.deterministic was the other half of the asymmetry: set for the
    # Python stacks alone, worth a further 1.35x, and unsettable in DL4J and R.
    r.append(check_regex("all", "S4 no stack pins cuDNN determinism",
                         glob_read("python/**/*.py", "tools/*.py",
                                   "rust/src/**/*.rs", "R/**/*.r"),
                         r"cudnn\.deterministic\s*=\s*True", expect=False))

    # ---------------- S4: one LibTorch behind the control group ------------
    #: tch crate version -> the LibTorch it links against
    TCH_TO_LIBTORCH = {
        "0.14": "2.1.0", "0.15": "2.2.0", "0.16": "2.3.0", "0.17": "2.4.0",
        "0.18": "2.5.0", "0.19": "2.6.0", "0.20": "2.7.0",
    }
    versions: dict[str, str] = {}
    try:
        import json as _json
        man = _json.loads(read("models/MANIFEST.json") or "{}")
        if man.get("torch_version"):
            versions["exported TorchScript"] = str(man["torch_version"]).split("+")[0]
    except Exception:
        pass
    m = re.search(r'^tch\s*=\s*"([0-9]+\.[0-9]+)', read("rust/Cargo.toml"), re.M)
    if m:
        versions["Rust/tch"] = TCH_TO_LIBTORCH.get(m.group(1), f"unknown (tch {m.group(1)})")
    m = re.search(r"libtorch-cxx11-abi-shared-with-deps-([0-9.]+)%2Bcu", read("cpp/CMakeLists.txt"))
    if m:
        versions["C++/LibTorch"] = m.group(1)

    distinct = {v.split("+")[0] for v in versions.values()}
    detail = ", ".join(f"{k} {v}" for k, v in sorted(versions.items()))
    if len(versions) < 2:
        r.append(Result("all", "S4 one LibTorch across the control group", SKIP,
                        "not enough version information"))
    elif len(distinct) == 1:
        r.append(Result("all", "S4 one LibTorch across the control group", PASS, detail))
    else:
        r.append(Result("all", "S4 one LibTorch across the control group", FAIL,
                        detail + "  -- a module exported by a newer torch will not "
                        "load into an older LibTorch at runtime"))

    rsmoke = (REPO / "R/scripts/load_shared_module_test.r").exists()
    r.append(Result("R/torch", "S1 shared-module smoke test present",
                    PASS if rsmoke else WARN,
                    "Rscript R/scripts/load_shared_module_test.r" if rsmoke else "no smoke test"))
    smoke = (REPO / "rust/examples/load_shared_module.rs").exists()
    r.append(Result("Rust/tch", "S1 shared-module smoke test present",
                    PASS if smoke else WARN,
                    "cargo run --example load_shared_module" if smoke else "no smoke test"))

    # ---------------- S4: Java carries its own CUDA runtime ----------------
    pom = read("Java/deepgreen-dl4j/pom.xml")
    r.append(Result("Java/DL4J", "S4 bundles the CUDA runtime it links against",
                    PASS if "cuda-platform-redist" in pom else FAIL,
                    "org.bytedeco:cuda-platform-redist" if "cuda-platform-redist" in pom
                    else "DL4J 1.0.0-M2.1 needs CUDA 11.6; without the redistributables "
                         "the backend fails to load on a CUDA 12 host"))

    # ---------------- device placement -------------------------------------
    r.append(Result("all", "S5 device placement asserted at startup",
                    PASS if "_assert_accelerator" in bench_src() else FAIL,
                    "tools/deepgreen_bench.py"))
    r.append(check_regex("Java/DL4J", "S5 GPU smoke test present",
                         glob_read("Java/**/DeepGreenSmokeTest.java"), r"GPU_VISIBLE"))
    r.append(check_regex("R/torch", "S5 refuses to run on the CPU",
                         glob_read("R/models/*.r"), r"does not see a GPU"))

    # ---------------- S5: measurement --------------------------------------
    bridge = read("tools/deepgreen_tracker.py")
    r.append(Result("all", "S5 one measurement bridge for the non-Python stacks",
                    PASS if "def start(" in bridge and "def metric(" in bridge else FAIL,
                    "tools/deepgreen_tracker.py"))
    for eco, files, pat in [
        ("Rust/tch", glob_read("rust/src/emissions.rs"), r"deepgreen_tracker"),
        ("C++/LibTorch", glob_read("cpp/src/python/PythonTracker.cpp"), r"deepgreen_tracker"),
        ("R/torch", glob_read("R/models/*.r"), r"deepgreen_tracking"),
    ]:
        r.append(check_regex(eco, "S5 uses the shared bridge", files, pat))
    r.append(check_regex("Rust/tch", "S5 tracker START is synchronous",
                         glob_read("rust/src/emissions.rs"), r"read_line\(&mut ack\)"))

    cc = read("tools/codecarbon_config.json")
    for key, want in [("tracking_mode", "machine"), ("measure_power_secs", "1")]:
        ok = f'"{key}": "{want}"' in cc or f'"{key}": {want}' in cc
        r.append(Result("all", f"S5 {key} = {want}", PASS if ok else FAIL, "tools/codecarbon_config.json"))
    ok = '"required_codecarbon_version": ">=3.0"' in cc
    r.append(Result("all", "S5 CodeCarbon >= 3.0 required", PASS if ok else FAIL, ""))
    bench = read("tools/deepgreen_bench.py")
    r.append(Result("all", "S5 quality metrics persisted",
                    PASS if "def log_metrics" in bench else FAIL,
                    "tools/deepgreen_bench.py"))
    for eco, files in [("Python/PyTorch", glob_read("python/pytorch/models/*.py")),
                       ("Python/TensorFlow", glob_read("python/tensorflow/models/*.py")),
                       ("Python/JAX", glob_read("python/jax/models/*.py"))]:
        r.append(check_regex(eco, "S5 uses the shared harness", files, r"from tools\.deepgreen_bench import"))
        r.append(check_regex(eco, "S5 logs accuracy", files, r"log_metrics\("))

    # ---------------- portability ------------------------------------------
    src = {}
    for pat in ("rust/src/**/*.rs", "dataloader/*.py", "cpp/CMakeLists.txt",
                "R/models/*.r", "Java/**/resources/**/*.py", "matlab/emissions/*.py",
                "cpp/emissions/*.py"):
        src.update(glob_read(pat))
    src = {k: v for k, v in src.items() if "/target/" not in k}
    # allow the strings inside explanatory comments
    stripped = {k: re.sub(r"(^|\s)(#|//|///).*$", "", v, flags=re.M) for k, v in src.items()}
    r.append(check_regex("all", "P1 no hard-coded author paths", stripped,
                         r"/home/(pampaj|marcopaglio)/", expect=False,
                         detail_ok="none outside comments"))

    # ---------------- S5: the metrics are persisted, and populated -----------
    # The checker verified that every binary calls log_metric. It did not
    # verify that the values arrive: Java wrote NaN for test_loss in all 900 of
    # its epoch rows, which left half the collapse/pipeline discriminator blind
    # for the stack contributing five of the twelve collapses.
    records = REPO / "results" / "replication" / "metrics.csv.gz"
    if records.exists():
        try:
            import pandas as _pd
            met = _pd.read_csv(records)
            for column in ("test_acc", "test_loss", "train_loss"):
                if column not in met:
                    r.append(Result("all", f"S5 {column} present in metrics",
                                    FAIL, "column missing"))
                    continue
                missing = met.groupby("ecosystem")[column].apply(
                    lambda x: float(x.isna().mean()))
                worst = missing.idxmax()
                pct = 100 * missing.max()
                r.append(Result(
                    "all", f"S5 {column} populated by every stack",
                    PASS if pct == 0 else FAIL,
                    "all stacks" if pct == 0 else f"{worst} missing {pct:.0f}%"))
        except Exception as exc:
            r.append(Result("all", "S5 metrics populated", SKIP, str(exc)[:60]))

    # ---------------- S6: replication ---------------------------------------
    # The specification has six parts and the checker covered five of them, so
    # the one guarantee no static check can be assumed to hold -- that the runs
    # are independent, differently seeded and interleaved -- was the one taken
    # on trust. It is checkable: the driver builds the schedule in this process.
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        import run_campaign  # noqa: E402  -- path set immediately above

        # The campaign's own dimensions, not a toy plan: on a handful of
        # configurations a shuffle collides at a repetition boundary by chance,
        # and the check would be measuring the toy rather than the design.
        ecos = sorted(set(run_campaign.PYTHON_ECOSYSTEMS) |
                      set(run_campaign.EXTERNAL_ECOSYSTEMS))
        models, datasets, reps = ["resnet18", "vgg16"], \
            ["fashionmnist", "cifar100", "tinyimagenet"], 5
        plan = run_campaign.build_plan(ecos, models, datasets, repetitions=reps)

        n_expected = len(ecos) * len(models) * len(datasets) * reps
        r.append(Result("all", "S6 one job per configuration per repetition",
                        PASS if len(plan) == n_expected else FAIL,
                        f"{len(plan)} jobs"))

        # Interleaved: repetitions of one configuration must be well separated.
        order = [(j.ecosystem, j.model, j.dataset) for j in plan]
        pos: dict[tuple, list[int]] = {}
        for i, c in enumerate(order):
            pos.setdefault(c, []).append(i)
        min_gap = min(min(b - a for a, b in zip(v, v[1:])) for v in pos.values())
        r.append(Result("all", "S6 repetitions interleaved, not consecutive",
                        PASS if min_gap > 1 else FAIL,
                        f"minimum {min_gap} jobs between repetitions"))

        # Distinct seeds within a configuration, and reproducible across calls.
        seeds: dict[tuple, set] = {}
        for j in plan:
            seeds.setdefault((j.ecosystem, j.model, j.dataset), set()).add(j.seed)
        fewest = min(len(v) for v in seeds.values())
        r.append(Result("all", "S6 distinct seed per repetition",
                        PASS if fewest == reps else FAIL,
                        f"{fewest} of {reps} distinct"))

        again = run_campaign.build_plan(ecos, models, datasets, repetitions=reps)
        same = [(j.ecosystem, j.model, j.dataset, j.repetition, j.seed) for j in plan] == \
               [(j.ecosystem, j.model, j.dataset, j.repetition, j.seed) for j in again]
        r.append(Result("all", "S6 schedule is reproducible",
                        PASS if same else FAIL, "fixed shuffle seed"))
    except Exception as exc:  # a checker that cannot check must say so
        r.append(Result("all", "S6 replication schedule", FAIL, f"{type(exc).__name__}: {exc}"))

    # ---------------- S5: one output contract, both paths -------------------
    # The driver sets DEEPGREEN_RUN_DIR; deepgreen_tracker.py honoured it and
    # deepgreen_bench.py hardcoded the campaign directory. Redirecting the
    # driver moved the lock and left the Python stacks writing over the
    # campaign. Both paths must read the same variable.
    # Tested by execution, not by grep. The grep form of this check was
    # tautological -- it asked whether the variable was mentioned in a file
    # whose mention of it the previous check already asserted -- so it could
    # never fire on its own, and a second output path added beside the first
    # would have left it printing [ok]. Resolving both paths against a
    # temporary directory is what actually catches that.
    r.extend(_check_run_dir_contract())

    # ---------------- S1: one network, asserted at startup ------------------
    #
    # The check that did not exist while the specification claimed it did.
    # VGG-16 ran as four different networks across the seven stacks, spanning
    # 9.1x in parameters, and ResNet-18 as three. Every stack now asserts its
    # own count against models/MANIFEST.json, carried into the run as
    # DEEPGREEN_EXPECTED_PARAMS so no stack needs a JSON parser of its own.
    manifest_path = REPO / "models" / "MANIFEST.json"
    if manifest_path.exists():
        import json as _json
        man = _json.loads(manifest_path.read_text())
        modules = man.get("modules", {})
        r.append(Result("all", "S1 model manifest covers every block",
                        PASS if len(modules) == 6 else FAIL,
                        f"{len(modules)} of 6 modules, VGG-16 head "
                        f"'{man.get('vgg16_head')}', seed {man.get('export_seed')}"))
    else:
        r.append(Result("all", "S1 model manifest covers every block", FAIL,
                        "models/MANIFEST.json missing; run "
                        "scripts/export_torchscript_models.py"))
    r.append(Result("all", "S1 driver carries the expected parameter count",
                    PASS if "DEEPGREEN_EXPECTED_PARAMS" in read("scripts/run_campaign.py")
                    else FAIL, "scripts/run_campaign.py"))
    for eco, files in (
        ("Python/PyTorch", glob_read("python/pytorch/models/*.py")),
        ("Python/TensorFlow", glob_read("python/tensorflow/models/*.py")),
        ("Python/JAX", glob_read("python/jax/models/*.py")),
        ("R/torch", glob_read("R/models/*.r")),
    ):
        pattern = (r"dg_assert_params|load_shared_module" if eco == "R/torch"
                   else r"assert_parameter_count|load_shared_module")
        r.append(check_regex(eco, "S1 asserts its parameter count", files, pattern))
    r.append(check_regex("Java/DL4J", "S1 asserts its parameter count",
                         glob_read("Java/**/expt/resnet18/*.java",
                                   "Java/**/expt/vgg16/*.java"),
                         r"assertParameters\("))

    # ---------------- S2: thirty epochs, and the seed reaches the data ------
    r.append(check_regex("all", "S2 epochs default to 30",
                         glob_read("tools/deepgreen_bench.py"),
                         r"DEFAULT_EPOCHS = 30", every=False,
                         detail_ok="tools/deepgreen_bench.py"))
    # Java hardcoded its shuffle seed, so five repetitions shared one data
    # order while the S6 check reported five distinct seeds -- it had asked the
    # campaign planner, which never sees whether a stack uses what it is given.
    r.append(check_regex("Java/DL4J", "S6 no hardcoded data seed",
                         strip_comments(glob_read("Java/**/dataloader/*.java")),
                         r"RNG_SEED\s*=\s*\d+", expect=False))
    # Two of six Rust binaries rebuilt the generator inside the epoch loop, so
    # they replayed one permutation thirty times.
    r.append(check_regex("Rust/tch", "S6 seeded once, outside the epoch loop",
                         strip_comments(glob_read("rust/src/bin/*.rs")),
                         r"for epoch in 1\.\.=epochs \{[^}]*?StdRng::seed_from_u64",
                         expect=False, flags=re.S))

    # ---------------- S3: the whole split, in every stack -------------------
    # TensorFlow and JAX dropped the final partial batch of both splits, so
    # they trained on fewer gradient steps and evaluated on 9,984 of 10,000
    # test images -- always the same 16, the test loader being unshuffled over
    # a sorted file list -- while accuracy is the denominator of this study's
    # energy-per-quality metric.
    r.append(check_regex("all", "S3 loader keeps the final partial batch",
                         strip_hash_comments(glob_read("tools/deepgreen_loader.py")),
                         r"drop_remainder=True", expect=False))
    # Anchored on the assignment: the ceiling idiom -(-x.samples // batch_size)
    # contains the floor form as a substring, so an unanchored pattern flags the
    # corrected code.
    r.append(check_regex("all", "S3 steps per epoch round up",
                         strip_hash_comments(glob_read("python/tensorflow/models/*.py",
                                                       "python/jax/models/*.py")),
                         r"=\s*\w+\.samples\s*//\s*batch_size", expect=False))

    # ---------------- S5: JAX closes its window on finished work ------------
    # Dispatch is asynchronous and nothing forced it, so the tracked block shut
    # while work was still on the device: 35.9% of an eval loop finished after
    # the tracker closed, and the block was charged 0.70x its true energy.
    r.append(check_regex("Python/JAX", "S5 synchronises before the block closes",
                         glob_read("python/jax/models/*.py"),
                         r"jax\.block_until_ready\("))

    # ---------------- scope -------------------------------------------------
    rc = read("scripts/run_campaign.py")
    r.append(Result("all", "V2 scope excludes MATLAB",
                    PASS if "MATLAB" not in rc.split("EXTERNAL_ECOSYSTEMS")[-1].split("}")[0] else FAIL,
                    "scripts/run_campaign.py"))
    return r


#: How many checks this script defines. Asserted at the end of every run, so a
#: check that silently stops running -- a glob that matches nothing, a guarded
#: import that turns into a SKIP -- fails the gate instead of shrinking the
#: total. Raise it deliberately when you add a check.
EXPECTED_CHECKS = 91


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        print("check_consistency.py needs pandas: three checks read the campaign's\n"
              "own records, and without them this script reports a clean run on a\n"
              "repository it has not finished checking. Use the campaign "
              "interpreter:\n\n    .venv-deepgreen/bin/python scripts/check_consistency.py\n",
              file=sys.stderr)
        return 2

    results = run()
    width = max(len(x.ecosystem) for x in results)
    order = {FAIL: 0, WARN: 1, SKIP: 2, PASS: 3}
    print("=" * 100)
    print("EXPERIMENT CONSISTENCY CHECK  (results/analysis/experiment_spec.md)")
    print("=" * 100)
    for x in sorted(results, key=lambda x: (order[x.status], x.ecosystem, x.check)):
        mark = {PASS: "ok  ", FAIL: "FAIL", WARN: "warn", SKIP: "skip"}[x.status]
        print(f"  [{mark}] {x.ecosystem:<{width}}  {x.check:<42} {x.detail}")

    counts = {s: sum(1 for x in results if x.status == s) for s in (PASS, FAIL, WARN, SKIP)}
    print("\n" + "-" * 100)
    print(f"  {counts[PASS]} pass, {counts[FAIL]} fail, {counts[WARN]} warn, {counts[SKIP]} skip")
    miscount = len(results) != EXPECTED_CHECKS
    if miscount:
        print(f"\n  {len(results)} checks ran, {EXPECTED_CHECKS} expected. A check that "
              f"stops running is\n  indistinguishable from one that passes; treat this "
              f"as a failure.")
    if counts[FAIL] or counts[WARN] or counts[SKIP]:
        print("\n  Anything not PASS is an inconsistency between ecosystems and invalidates")
        print("  a cross-ecosystem comparison until it is resolved or explicitly justified.")
    print("-" * 100)
    # SKIP counts against --strict too: a check that did not run has not passed.
    return 1 if (args.strict and (counts[FAIL] or counts[SKIP] or miscount)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
