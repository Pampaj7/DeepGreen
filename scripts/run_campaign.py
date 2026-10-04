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
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.deepgreen_bench import RunContext, expected_parameters  # noqa: E402

MODELS = ["resnet18", "vgg16"]
#: Pre-resized to the training resolution by
#: scripts/normalise_dataset_resolution.py, so that no stack resizes anything.
#: The seven ecosystems used four different resamplers and did not agree: over
#: Tiny ImageNet's whole test split the pixel standard deviation was 0.2639 in
#: C++, Java and R, 0.2584 in Rust and 0.2561 in PyTorch and TensorFlow, with
#: the means agreeing to 0.1% -- the signature of a filter, not of content.
#: Every image is 32x32 on disk now, so each stack's resize is a no-op and they
#: decode identical pixels, which is the position CIFAR-100 was already in.
#: The directory names are unchanged deliberately: the resolution is hardcoded
#: at some thirty sites across Rust, R, Java and the Python wrappers, and thirty
#: edits is how a constant drifts. The originals are kept beside them with an
#: _original suffix.
DATASETS = {
    "fashionmnist": "data/fashion_mnist_png",
    "cifar100": "data/cifar100_png",
    "tinyimagenet": "data/tiny_imagenet_png",
}

#: The accelerator-saturation cell (spec S7), deliberately outside the grid above.
#
# The campaign's three datasets are 32x32 at batch 128, and at that shape the
# accelerator is idle for most of the wall clock -- mean GPU utilisation runs
# from 4.7% (R/torch, ResNet-18) to 79.9% (Java/DL4J, VGG-16), with most stacks
# between 20% and 50%. The reasonable objection is that such a campaign measures
# host-side overhead rather than deep-learning energy. This cell answers it with
# a measurement instead of an argument: the same two networks on ten ImageNet
# classes at 224x224, where the GPU is the bottleneck, run through this same
# driver under the same contract.
#
# It is NOT in the default grid and never joins it. `results/campaign_v2` holds
# 210 runs and every table in the manuscript counts them; a 71st run in that
# directory would change the paper's numbers without changing its text. So
# imagenette is selectable only by naming it, it may not be mixed with the 32x32
# datasets in one invocation, and it refuses to run unless the operator has
# pointed DEEPGREEN_CAMPAIGN_DIR somewhere that is not campaign_v2.
SATURATION_DATASETS = {"imagenette": "data/imagenette_png"}

#: Every dataset this driver can address, whichever grid it belongs to.
ALL_DATASETS = DATASETS | SATURATION_DATASETS

#: Per-dataset shape. One place, because seven stacks read it out of the
#: environment and a second copy is how the batch size drifts.
DEFAULT_BATCH_SIZE = 128
DEFAULT_IMG_SIZE = 32
#: VGG-16 at 224x224 does not fit 24 GB at batch 128, so the saturation cell
#: trains and evaluates at 32. Stated here, carried to every stack as
#: DEEPGREEN_BATCH_SIZE, and grepped for in each stack's sources by
#: scripts/check_consistency.py.
BATCH_SIZE = {"imagenette": 32}
IMG_SIZE = {"imagenette": 224}


def batch_size_for(dataset: str) -> int:
    return BATCH_SIZE.get(dataset, DEFAULT_BATCH_SIZE)


def img_size_for(dataset: str) -> int:
    return IMG_SIZE.get(dataset, DEFAULT_IMG_SIZE)

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
CPP_DATASET = {"fashionmnist": "fashion", "cifar100": "cifar100", "tinyimagenet": "tiny",
               #: The saturation cell keeps its full name in every language:
               #: rust/target/release/{resnet,vgg}_imagenette,
               #: cpp/build-cuda/{resnet18,vgg16}_imagenette_imported,
               #: R/train/{resnet18,vgg}/train_imagenette.r.
               "imagenette": "imagenette"}
#: R uses the same short dataset names as the C++ targets.
R_DATASET = CPP_DATASET
JAVA_CLASS = {
    ("resnet18", "fashionmnist"): "ResNet18TrainFashionExpt",
    ("resnet18", "cifar100"): "ResNet18TrainCifar100Expt",
    ("resnet18", "tinyimagenet"): "ResNet18TrainTinyExpt",
    ("resnet18", "imagenette"): "ResNet18TrainImagenetteExpt",
    ("vgg16", "fashionmnist"): "Vgg16TrainFashionExpt",
    ("vgg16", "cifar100"): "Vgg16TrainCifar100Expt",
    ("vgg16", "tinyimagenet"): "Vgg16TrainTinyExpt",
    ("vgg16", "imagenette"): "Vgg16TrainImagenetteExpt",
}


def campaign_dir() -> Path:
    """Where runs are written.

    Configurable so that a calibration re-execution -- re-running an already
    completed configuration to measure drift between two time windows -- cannot
    overwrite the campaign it is calibrating against. The default is unchanged.

    Resolved against the repository root, once, and absolute from here on.
    A relative value used to be resolved twice against two different working
    directories: the saturation guard resolved it against the caller's cwd
    while the children were launched with cwd=REPO_ROOT and given the raw
    string, so `DEEPGREEN_CAMPAIGN_DIR=results/campaign_v2` from any directory
    but the repository root passed the guard that exists to refuse it and then
    wrote into the frozen campaign. The children now receive an absolute
    DEEPGREEN_RUN_DIR, and the guard and the run-directory backstop ask about
    the same path the runs are written to.
    """
    configured = os.environ.get("DEEPGREEN_CAMPAIGN_DIR")
    if not configured:
        return REPO_ROOT / "results" / "campaign_v2"
    return (REPO_ROOT / configured).resolve()


def run_environment(job: "Job") -> dict[str, str]:
    """The shared run contract every ecosystem reads."""
    ctx = RunContext(ecosystem=job.ecosystem, model=job.model,
                     dataset=job.dataset, repetition=job.repetition)
    return {
        "DEEPGREEN_RUN_DIR": str(campaign_dir() / ctx.slug),
        "DEEPGREEN_ECOSYSTEM": job.ecosystem,
        "DEEPGREEN_MODEL": job.model,
        "DEEPGREEN_DATASET": job.dataset,
        "DEEPGREEN_REP": str(job.repetition),
        "DEEPGREEN_SEED": str(job.seed),
        "DEEPGREEN_EPOCHS": os.environ.get("DEEPGREEN_EPOCHS", "30"),
        # Shape of the workload, per dataset. The three 32x32 datasets keep the
        # values every stack already hardcodes (128 and 32), so setting these
        # changes nothing for them; the saturation cell is the reason they are
        # in the contract at all rather than in seven source trees.
        "DEEPGREEN_BATCH_SIZE": str(batch_size_for(job.dataset)),
        "DEEPGREEN_IMG_SIZE": str(img_size_for(job.dataset)),
        "DEEPGREEN_DATA": os.environ.get("DEEPGREEN_DATA", str(REPO_ROOT / "data")),
        "DEEPGREEN_MODELS": os.environ.get("DEEPGREEN_MODELS", str(REPO_ROOT / "models")),
        "DEEPGREEN_PYTHON": os.environ.get(
            "DEEPGREEN_PYTHON", str(REPO_ROOT / ".venv-deepgreen" / "bin" / "python")),
        "DEEPGREEN_LOADER_THREADS": os.environ.get("DEEPGREEN_LOADER_THREADS", "2"),
        # The network this job must train, as a number every stack can check
        # without parsing JSON in four languages. VGG-16 ran as four different
        # networks across the seven stacks -- 134,670,244 parameters in the
        # LibTorch lineage against 14,765,988 in JAX -- under a specification
        # that claimed parameter counts were checked against the exported
        # module. Nothing checked them. Empty when the manifest is missing, and
        # each stack decides whether that is fatal.
        "DEEPGREEN_EXPECTED_PARAMS": str(
            expected_parameters(job.model, job.dataset) or ""),
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
    n = img_size_for(job.dataset)
    return [
        interpreter_for(job.ecosystem),
        "-c",
        (
            f"import {module} as M; "
            f"M.run_experiment(dataset_path={ALL_DATASETS[job.dataset]!r}, "
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
            # epochs comes from the run contract, like everything else. It did
            # not: the C++, Rust, R and Java stacks read DEEPGREEN_EPOCHS while
            # these three took the signature default, so setting the variable
            # gave four stacks one epoch count and three another -- a contract
            # that four of seven honoured.
            + f"repetition={job.repetition}, seed={job.seed}, "
              f"epochs={int(os.environ.get('DEEPGREEN_EPOCHS', 30))}, "
            # The shape travels as arguments rather than as a signature default,
            # for the reason above the epochs line: a default is a value four of
            # seven stacks honour. Explicit for every dataset, so the 32x32 cells
            # pass the numbers they were already using.
            + f"batch_size={batch_size_for(job.dataset)}, "
              f"img_size=({n}, {n}), "
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
    lock_path = campaign_dir() / ".campaign.lock"
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


def _validate_datasets(datasets: list[str]) -> None:
    """Gate the saturation cell. Refuses loudly rather than writing somewhere wrong.

    Three refusals, each for a way this cell could quietly contaminate the
    campaign it is a contrast to:

      * an unknown dataset name -- a typo silently produced a plan with a job
        the driver could not launch;
      * imagenette without a campaign directory of its own, or with one that
        resolves inside results/campaign_v2. The 210-run campaign is frozen and
        every table in the manuscript counts it; the saturation cell is 70
        further runs at a different resolution and batch size, and one of them
        landing in that directory would change published numbers with nothing
        in the output to show for it;
      * imagenette mixed with the 32x32 datasets in one invocation. They differ
        in resolution and batch size, so a mixed plan is two experiments sharing
        one cooldown schedule and one run directory, and the analysis would have
        to separate them afterwards from the dataset column alone.
    """
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        raise SystemExit(
            f"error: unknown dataset(s) {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(sorted(ALL_DATASETS))}.")

    saturation = [d for d in datasets if d in SATURATION_DATASETS]
    if not saturation:
        return

    standard = [d for d in datasets if d in DATASETS]
    if standard:
        raise SystemExit(
            "error: the saturation cell cannot share an invocation with the "
            f"32x32 datasets ({', '.join(sorted(standard))}).\n"
            "They differ in input resolution (224 vs 32) and batch size (32 vs "
            "128), so one plan\nover both is two experiments in one run "
            "directory. Run them separately.")

    configured = os.environ.get("DEEPGREEN_CAMPAIGN_DIR", "").strip()
    if not configured:
        raise SystemExit(
            "error: --datasets imagenette needs DEEPGREEN_CAMPAIGN_DIR set.\n"
            "The saturation cell is a separate campaign: results/campaign_v2 "
            "holds the frozen\n210 runs the manuscript's tables count, and this "
            "cell adds 70 runs at a different\nresolution and batch size. "
            "Point it somewhere of its own, e.g.\n\n"
            "    DEEPGREEN_CAMPAIGN_DIR=results/campaign_saturation \\\n"
            "        python3 scripts/run_campaign.py --datasets imagenette "
            "--repetitions 5\n")
    frozen = (REPO_ROOT / "results" / "campaign_v2").resolve()
    # campaign_dir(), not a second resolution of the same string: this must be
    # the directory the runs are actually written to, resolved the same way and
    # against the same root, or the guard answers a question about a path
    # nothing uses.
    target = campaign_dir().resolve()
    if target == frozen or frozen in target.parents:
        raise SystemExit(
            f"error: DEEPGREEN_CAMPAIGN_DIR={configured} resolves inside "
            f"{frozen}.\nThat directory holds the frozen 210-run campaign and "
            "must not gain a run.\nUse results/campaign_saturation, or another "
            "directory outside it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repetitions", type=int, default=5,
                    help="independent run-level repetitions per configuration (default 5)")
    ap.add_argument("--ecosystems", nargs="*",
                    default=list(PYTHON_ECOSYSTEMS) + list(EXTERNAL_ECOSYSTEMS))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    help="default: the three 32x32 datasets. "
                         f"{', '.join(SATURATION_DATASETS)} is the "
                         "accelerator-saturation cell: it is not in the default "
                         "grid, cannot be mixed with them, and needs "
                         "DEEPGREEN_CAMPAIGN_DIR set outside results/campaign_v2")
    ap.add_argument("--print-plan", action="store_true",
                    help="write the schedule and exit without executing anything")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_S)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="replace a run directory that already holds data")
    args = ap.parse_args()

    # Before the lock and before the accelerator check: a plan that must not be
    # built should not first take a lock on the directory it must not write to.
    _validate_datasets(args.datasets)

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
        out = campaign_dir() / "plan.json"
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
        # relative_to raises for a campaign directory outside the repository,
        # which is exactly where a smoke test points DEEPGREEN_CAMPAIGN_DIR, so
        # --print-plan died after writing the plan it was asked for.
        try:
            shown_out: Path | str = out.relative_to(REPO_ROOT)
        except ValueError:
            shown_out = out
        print(f"{len(plan)} jobs written to {shown_out}")
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
        # Refuse to run into a directory that already holds a run. Both output
        # paths open counters.csv in append mode, so re-running a job over its
        # own output silently doubles the rows -- thirty epochs recorded as
        # sixty, with the duplicates interleaved and no marker distinguishing
        # them. Found by re-running one smoke test into the same directory and
        # noticing Java had four blocks where every other stack had two.
        # Absolute, from campaign_dir(): the same resolved path the child is
        # given and the same one the saturation guard was asked about. It used
        # to be whatever string the environment held, evaluated in the parent's
        # cwd, so a relative campaign directory made this backstop look at a
        # different place from the one the run was written to.
        run_dir = Path(env["DEEPGREEN_RUN_DIR"])
        existing = run_dir / "counters.csv"
        if existing.exists() and existing.stat().st_size > 0:
            if not args.force:
                print(f"{tag}: refusing, {existing} already has data "
                      f"(use --force to replace)", flush=True)
                failures.append((tag, "run directory not empty"))
                continue
            shutil.rmtree(run_dir)
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
