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


def glob_read(pattern: str) -> dict[str, str]:
    return {
        str(p.relative_to(REPO)): p.read_text(errors="replace")
        for p in sorted(REPO.glob(pattern))
        if p.is_file()
    }


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


def check_regex(eco, name, files, pattern, expect=True, detail_ok="", flags=0):
    """PASS if `pattern` is found (expect=True) or absent (expect=False)."""
    hits = []
    for path, text in files.items():
        for m in re.finditer(pattern, text, flags):
            line = text[: m.start()].count("\n") + 1
            hits.append(f"{path}:{line}")
    found = bool(hits)
    ok = found == expect
    if not files:
        return Result(eco, name, SKIP, "no source files matched")
    if ok:
        return Result(eco, name, PASS, detail_ok or (hits[0] if hits else "absent"))
    if expect:
        return Result(eco, name, FAIL, "pattern not found")
    return Result(eco, name, FAIL, "found at " + ", ".join(hits[:3]))


def bench_src() -> str:
    return read("tools/deepgreen_bench.py")


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
                         glob_read("Java/**/expt/**/*.java"), r"lrAdam\s*=\s*1e-4"))

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
                         glob_read("rust/src/datasets/*.rs"), r"init_loader_pool\(\)"))

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
                         glob_read("rust/src/datasets/*.rs"), r"crate::normalize_inputs\(\)"))
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
        ("Java/DL4J", glob_read("Java/**/expt/**/*.java"), r"batchSize\s*=\s*128"),
        ("C++/LibTorch", glob_read("cpp/src/train/**/train_*.cpp"), r"kTrainBatchSize = 128"),
    ]:
        r.append(check_regex(eco, "S2 batch size = 128", files, pat))
    r.append(check_regex("C++/LibTorch", "S2 eval batch = train batch",
                         glob_read("cpp/src/train/**/train_*.cpp"), r"kTestBatchSize = 128"))

    # ---------------- S4: one LibTorch behind the control group ------------
    #: tch crate version -> the LibTorch it links against
    TCH_TO_LIBTORCH = {
        "0.14": "2.1.0", "0.15": "2.2.0", "0.16": "2.3.0", "0.17": "2.4.0",
        "0.18": "2.5.0", "0.19": "2.6.0", "0.20": "2.7.0",
    }
    versions: dict[str, str] = {}
    man = read("models/MANIFEST.txt")
    m = re.search(r"torch_version=([0-9.]+)", man)
    if m:
        versions["exported TorchScript"] = m.group(1)
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
    for name in ("tools/deepgreen_bench.py", "tools/deepgreen_tracker.py"):
        src = read(name)
        r.append(Result("all", f"S5 {Path(name).stem} honours DEEPGREEN_RUN_DIR",
                        PASS if "DEEPGREEN_RUN_DIR" in src else FAIL, name))
    bench = read("tools/deepgreen_bench.py")
    hardcoded = re.search(r'"results"\s*/\s*"campaign_v2"', bench) is not None
    unconditional = hardcoded and "DEEPGREEN_RUN_DIR" not in bench
    r.append(Result("all", "S5 output path is not hardcoded past the contract",
                    FAIL if unconditional else PASS,
                    "campaign_v2 appears only as the fallback"))

    # ---------------- scope -------------------------------------------------
    rc = read("scripts/run_campaign.py")
    r.append(Result("all", "V2 scope excludes MATLAB",
                    PASS if "MATLAB" not in rc.split("EXTERNAL_ECOSYSTEMS")[-1].split("}")[0] else FAIL,
                    "scripts/run_campaign.py"))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

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
    if counts[FAIL] or counts[WARN]:
        print("\n  Anything not PASS is an inconsistency between ecosystems and invalidates")
        print("  a cross-ecosystem comparison until it is resolved or explicitly justified.")
    print("-" * 100)
    return 1 if (args.strict and counts[FAIL]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
