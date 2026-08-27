#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Campaign driver: independent run-level repetitions.

The first campaign executed each (ecosystem, model, dataset) configuration once
and treated the 30 epochs of that run as repeated measurements. Epochs within a
run share the initialisation, the JIT outcome, the allocator state and the
thermal trajectory, so the effective sample size per configuration was one.

This driver executes each configuration ``--repetitions`` times as separate
processes with distinct seeds, interleaving repetitions rather than running them
back to back, so that drift in machine state (thermal, background load, driver
clock behaviour) is spread across conditions instead of aliasing onto one
ecosystem.

Only the Python-hosted ecosystems can be launched directly from here. The
C++, Java, R, MATLAB and Rust stacks are launched through their own build
systems; ``--print-plan`` emits the exact command list, including repetition
index and seed, so the same schedule can be driven externally.

    python3 scripts/run_campaign.py --repetitions 5 --print-plan
    python3 scripts/run_campaign.py --repetitions 5 --ecosystems Python/PyTorch
"""

from __future__ import annotations

import argparse
import atexit
import errno
import fcntl
import itertools
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.deepgreen_bench import RunContext  # noqa: E402

MODELS = ["resnet18", "vgg16"]
DATASETS = {
    "fashionmnist": "data/fashion_mnist_png",
    "cifar100": "data/cifar100_png",
    "tinyimagenet": "data/tiny_imagenet_png",
}

#: Python-hosted ecosystems this driver can launch itself.
PYTHON_ECOSYSTEMS = {
    "Python/PyTorch": "python.pytorch.models.{model}",
    "Python/TensorFlow": "python.tensorflow.models.{model}",
    "Python/JAX": "python.jax.models.{model}",
}

#: One virtualenv per Python ecosystem.
#
# They cannot share one: torch's cu128 wheels and TensorFlow's bundled CUDA
# libraries resolve to different versions of the same shared objects, and
# whichever loses ends up on CPU silently -- TensorFlow reported
# "Cannot dlopen some GPU libraries" and ran unaccelerated in a shared venv,
# which would have been measured as an ecosystem property rather than a
# packaging accident. Override any of them with DEEPGREEN_PYTHON_<STACK>.
VENV_FOR_ECOSYSTEM = {
    "Python/PyTorch": ".venv-deepgreen",
    "Python/TensorFlow": ".venv-tensorflow",
    "Python/JAX": ".venv-jax",
}


def interpreter_for(ecosystem: str) -> str:
    """The interpreter that runs one ecosystem, with a clear failure if absent."""
    override = os.environ.get(
        "DEEPGREEN_PYTHON_" + ecosystem.split("/")[-1].upper()
    )
    if override:
        return override
    venv = REPO_ROOT / VENV_FOR_ECOSYSTEM[ecosystem] / "bin" / "python"
    if not venv.exists():
        raise SystemExit(
            f"{ecosystem}: no interpreter at {venv}.\n"
            "Run scripts/setup_environment.sh, or set "
            f"DEEPGREEN_PYTHON_{ecosystem.split('/')[-1].upper()}."
        )
    return str(venv)

# MATLAB/DLT is deliberately absent: it needs a proprietary toolbox with no
# license on the replication machine, and it cannot be pinned to the shared
# LibTorch build the other stacks are aligned on. The replicated campaign covers
# seven ecosystems. See common.py::EXCLUDED_FROM_V2.

#: Everything else. The command is a template the external harness must honour;
#: the repetition index and seed must be threaded through to CodeCarbon's output
#: directory exactly as the Python stacks do.
# These are driven through the shared environment contract (see
# tools/deepgreen_tracker.py) rather than command-line flags: adding argument
# parsing to a C++ binary, a Maven exec target, an Rscript and a Rust binary is
# four different pieces of plumbing and four places for the stacks to drift.
EXTERNAL_ECOSYSTEMS = {
    "Rust/tch": "rust/target/release/{rust_bin}",
    "C++/LibTorch": "cpp/build-cuda/{model}_{cpp_dataset}_imported",
    "Java/DL4J": "mvn -q -f Java/deepgreen-dl4j/pom.xml exec:java "
                 "-Dexec.mainClass=io.github.stlabunifi.deepgreen.dl4j.expt.{model}."
                 "{java_class}",
    "R/torch": "Rscript R/train/{r_model}/train_{r_dataset}.r",
}

#: Per-language naming, kept in one place so the driver does not encode it inline.
#: Short names used by the Rust binaries and the R script tree; the campaign
#: speaks in full names (resnet18, tinyimagenet) and these map onto the
#: per-language conventions. The preflight check catches a mismatch before
#: a multi-day run rather than at job 40.
SHORT_MODEL = {"resnet18": "resnet", "vgg16": "vgg"}
RUST_BIN = SHORT_MODEL
CPP_DATASET = {"fashionmnist": "fashion", "cifar100": "cifar100", "tinyimagenet": "tiny"}
#: R uses the same short dataset names as the C++ targets.
R_DATASET = CPP_DATASET
JAVA_CLASS = {
    ("resnet18", "fashionmnist"): "ResNet18TrainFashionExpt",
    ("resnet18", "cifar100"): "ResNet18TrainCifar100Expt",
    ("resnet18", "tinyimagenet"): "ResNet18TrainTinyExpt",
    ("vgg16", "fashionmnist"): "Vgg16TrainFashionExpt",
    ("vgg16", "cifar100"): "Vgg16TrainCifar100Expt",
    ("vgg16", "tinyimagenet"): "Vgg16TrainTinyExpt",
}


def run_environment(job: "Job") -> dict[str, str]:
    """The shared run contract every ecosystem reads."""
    ctx = RunContext(ecosystem=job.ecosystem, model=job.model,
                     dataset=job.dataset, repetition=job.repetition)
    return {
        "DEEPGREEN_RUN_DIR": str(REPO_ROOT / "results" / "campaign_v2" / ctx.slug),
        "DEEPGREEN_ECOSYSTEM": job.ecosystem,
        "DEEPGREEN_MODEL": job.model,
        "DEEPGREEN_DATASET": job.dataset,
        "DEEPGREEN_REP": str(job.repetition),
        "DEEPGREEN_SEED": str(job.seed),
        "DEEPGREEN_EPOCHS": os.environ.get("DEEPGREEN_EPOCHS", "30"),
        "DEEPGREEN_DATA": os.environ.get("DEEPGREEN_DATA", str(REPO_ROOT / "data")),
        "DEEPGREEN_MODELS": os.environ.get("DEEPGREEN_MODELS", str(REPO_ROOT / "models")),
        "DEEPGREEN_PYTHON": os.environ.get(
            "DEEPGREEN_PYTHON", str(REPO_ROOT / ".venv-deepgreen" / "bin" / "python")),
        "DEEPGREEN_LOADER_THREADS": os.environ.get("DEEPGREEN_LOADER_THREADS", "2"),
    }

COOLDOWN_S = 60  # let the machine return to a comparable thermal state


@dataclass(frozen=True)
class Job:
    ecosystem: str
    model: str
    dataset: str
    repetition: int

    @property
    def seed(self) -> int:
        return RunContext(
            ecosystem=self.ecosystem, model=self.model,
            dataset=self.dataset, repetition=self.repetition,
        ).seed


def build_plan(ecosystems: list[str], models: list[str], datasets: list[str],
               repetitions: int, shuffle_seed: int = 7) -> list[Job]:
    """One job per (ecosystem, model, dataset, repetition).

    Repetitions are interleaved: the schedule iterates repetition-major, and
    within a repetition the configuration order is shuffled with a fixed seed.
    """
    plan: list[Job] = []
    rng = random.Random(shuffle_seed)
    for rep in range(repetitions):
        block = [
            Job(e, m, d, rep)
            for e, m, d in itertools.product(ecosystems, models, datasets)
        ]
        rng.shuffle(block)
        plan.extend(block)
    return plan


def python_command(job: Job) -> list[str]:
    module = PYTHON_ECOSYSTEMS[job.ecosystem].format(model=job.model)
    return [
        interpreter_for(job.ecosystem),
        "-c",
        (
            f"import {module} as M; "
            f"M.run_experiment(dataset_path={DATASETS[job.dataset]!r}, "
            + (
                f"output_file_train='{job.model}_{job.dataset}_train', "
                f"output_file_eval='{job.model}_{job.dataset}_eval', "
                if job.ecosystem == "Python/TensorFlow"
                else f"output_file_base='{job.model}_{job.dataset}', "
                if job.ecosystem == "Python/JAX"
                else f"output_file='{job.model}_{job.dataset}', "
            )
            + f"checkpoint_path='checkpoints/{job.ecosystem.replace('/', '_')}_{job.model}_"
              f"{job.dataset}_rep{job.repetition}.ckpt', "
            + f"repetition={job.repetition}, seed={job.seed}, "
              f"dataset_name={job.dataset!r})"
        ),
    ]


def stack_environment(ecosystem: str) -> dict[str, str]:
    """Launch environment for one ecosystem, from tools/stack_environments.json.

    Each non-Python stack needs library paths no other stack needs, and getting
    one wrong is not a loud failure: the Rust binary falls back to the CPU, the R
    package reports "Lantern is not loaded", the C++ binary cannot find
    codecarbon. Keeping them in one file makes them reviewable.
    """
    path = REPO_ROOT / "tools" / "stack_environments.json"
    if not path.exists():
        return {}
    spec = json.loads(path.read_text()).get(ecosystem, {})
    out: dict[str, str] = {}
    for key, value in spec.items():
        if key.startswith("_"):
            continue
        expanded = value.replace("$REPO", str(REPO_ROOT))
        expanded = os.path.expandvars(expanded)
        if "${" in expanded:
            raise SystemExit(
                f"{ecosystem}: unresolved variable in {key}={value!r}. Set it in the "
                "environment, or edit tools/stack_environments.json for this host."
            )
        out[key] = expanded
    return out


def external_command(job: Job) -> str:
    return EXTERNAL_ECOSYSTEMS[job.ecosystem].format(
        model=job.model,
        dataset=job.dataset,
        rust_bin=f"{SHORT_MODEL.get(job.model, job.model)}_{CPP_DATASET.get(job.dataset, job.dataset)}",
        cpp_dataset=CPP_DATASET.get(job.dataset, job.dataset),
        r_dataset=R_DATASET.get(job.dataset, job.dataset),
        r_model=SHORT_MODEL.get(job.model, job.model)
        if (REPO_ROOT / 'R' / 'train' / SHORT_MODEL.get(job.model, job.model)).is_dir()
        else job.model,
        java_class=JAVA_CLASS.get((job.model, job.dataset), "?"),
    )


def _acquire_exclusive_lock() -> None:
    """Refuse to start while another campaign is running.

    Nothing prevented two drivers from executing at once, and when it happened
    they wrote to the same run directories: counters.csv gained a second run's
    epochs appended to the first, so a "30-epoch" run held 60. Worse, the
    directories that ended up with a plausible 30 were measured while a second
    training job shared the accelerator, which contaminates the energy without
    leaving any trace in the file.

    Machine-mode energy measurement assumes the machine is doing one thing.
    That assumption now has a lock behind it rather than a convention.
    """
    lock_path = REPO_ROOT / "results" / "campaign_v2" / ".campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EAGAIN):
            raise
        try:
            holder = lock_path.read_text().strip()
        except OSError:
            holder = "unknown"
        print(
            f"error: another campaign holds {lock_path} (pid {holder}).\n"
            "Two drivers writing the same run directories corrupts both, and\n"
            "sharing the accelerator invalidates the energy of whatever else is\n"
            "measuring. Stop it first, or pass --dry-run to inspect the plan.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    # Held for the process lifetime; released when it exits, however it exits.
    atexit.register(handle.close)
    globals()["_CAMPAIGN_LOCK"] = handle


def _assert_accelerator_idle() -> None:
    """Refuse to start while anything else is on the accelerator.

    The exclusive lock stops a second *driver*, which is not the same thing as
    stopping a second *workload*: killing a driver leaves its training binary
    running, and that orphan keeps a CUDA context and a share of the GPU. A
    campaign started next to one measures two jobs and attributes both to one,
    with nothing in the output to show for it.

    Machine-mode measurement is a claim about the whole machine, so the check
    has to be about the whole machine too.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        print("warning: could not query the accelerator; proceeding unchecked.",
              file=sys.stderr)
        return
    busy = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if busy:
        print(
            "error: the accelerator is already in use:\n  "
            + "\n  ".join(busy)
            + "\nWhole-machine energy measurement attributes every watt to the run\n"
              "being tracked, so a second workload silently inflates it. Stop those\n"
              "processes first -- note that killing a campaign driver does not kill\n"
              "the training binary it launched.",
            file=sys.stderr,
        )
        raise SystemExit(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repetitions", type=int, default=5,
                    help="independent run-level repetitions per configuration (default 5)")
    ap.add_argument("--ecosystems", nargs="*",
                    default=list(PYTHON_ECOSYSTEMS) + list(EXTERNAL_ECOSYSTEMS))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--print-plan", action="store_true",
                    help="write the schedule and exit without executing anything")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_S)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.print_plan or args.dry_run):
        _acquire_exclusive_lock()
        _assert_accelerator_idle()

    if args.repetitions < 3:
        print(
            f"warning: {args.repetitions} repetitions gives a weak estimate of between-run "
            "variability; 5 to 10 is the usual recommendation.",
            file=sys.stderr,
        )

    plan = build_plan(args.ecosystems, args.models, args.datasets, args.repetitions)

    if args.print_plan:
        out = REPO_ROOT / "results" / "campaign_v2" / "plan.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "index": i,
                "ecosystem": j.ecosystem,
                "model": j.model,
                "dataset": j.dataset,
                "repetition": j.repetition,
                "seed": j.seed,
                "command": (
                    " ".join(python_command(j)) if j.ecosystem in PYTHON_ECOSYSTEMS
                    else external_command(j)
                ),
                "env": run_environment(j) | stack_environment(j.ecosystem),
            }
            for i, j in enumerate(plan)
        ]
        out.write_text(json.dumps(payload, indent=2))
        print(f"{len(plan)} jobs written to {out.relative_to(REPO_ROOT)}")
        for row in payload[:5]:
            print(f"  [{row['index']}] {row['ecosystem']} {row['model']}/{row['dataset']} "
                  f"rep{row['repetition']} seed{row['seed']}")
        print("  ...")
        return 0

    failures = []
    for i, job in enumerate(plan, 1):
        tag = f"[{i}/{len(plan)}] {job.ecosystem} {job.model}/{job.dataset} rep{job.repetition}"
        env = os.environ | run_environment(job) | stack_environment(job.ecosystem)
        if job.ecosystem in PYTHON_ECOSYSTEMS:
            cmd: list[str] | str = python_command(job)
            shell = False
            shown = " ".join(cmd[:2])
        else:
            cmd = external_command(job)
            shell = True
            shown = cmd
        print(f"{tag}: {shown}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, cwd=REPO_ROOT, env=env, shell=shell)
        if rc != 0:
            failures.append((tag, rc))
            print(f"{tag}: FAILED with exit code {rc}", file=sys.stderr)
        if args.cooldown and i < len(plan):
            time.sleep(args.cooldown)

    if failures:
        print(f"\n{len(failures)} job(s) failed:", file=sys.stderr)
        for tag, rc in failures:
            print(f"  {tag} (exit {rc})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
