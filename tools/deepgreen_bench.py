#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared measurement harness for every Python-hosted ecosystem in DeepGreen.

Motivation
----------
The first campaign configured CodeCarbon differently in each ecosystem:

    ecosystem            CodeCarbon   tracking_mode   measure_power_secs
    Python/PyTorch       2.8.4        machine         15 (default)
    Python/TensorFlow    2.8.4        machine         15 (default)
    Python/JAX           2.8.4        machine          1
    Rust/tch             2.8.4        machine         15 (default)
    R/torch              2.8.4        PROCESS         15 (default)
    C++/LibTorch         3.0.4        machine         15 (default)
    Java/DL4J            3.0.4        machine         15 (default)
    MATLAB/DLT           3.0.4        machine         15 (default)

CodeCarbon 2.8.x and 3.0.x use different CPU and RAM power models, and process
mode excludes most host energy, so the measured quantity was not the same
across ecosystems.  ``results/analysis/01_data_audit.py`` quantifies the effect.

This module pins one configuration for every ecosystem, records the
configuration alongside the measurements, and logs model quality per epoch so
that energy can be normalised by the useful work produced.

Usage
-----
    from tools.deepgreen_bench import RunContext, Harness

    ctx = RunContext(ecosystem="Python/PyTorch", model="resnet18",
                     dataset="cifar100", repetition=0)
    h = Harness(ctx)
    h.set_seeds()
    for epoch in range(1, ctx.epochs + 1):
        with h.track("train", epoch):
            train_loss = train(...)
        with h.track("eval", epoch):
            test_loss, test_acc = evaluate(...)
        h.log_metrics(epoch, train_loss=train_loss,
                      test_loss=test_loss, test_acc=test_acc)
    h.close()
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
import csv as _csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# The single instrument configuration used by every ecosystem.
# Any change here must be applied to the non-Python stacks too; the values are
# mirrored in tools/codecarbon_config.json for the C++, Java, R, MATLAB and
# Rust harnesses, which drive CodeCarbon through their own bridges.
# ---------------------------------------------------------------------------
CODECARBON_CONFIG: dict[str, Any] = {
    # Whole-machine boundary. Process mode was used for R in the first campaign
    # and made its energy incomparable with the rest.
    "tracking_mode": "machine",
    # 1 s sampling. The default of 15 s is longer than most inference blocks in
    # this workload (1.6-13 s), which makes the sampled *_power columns
    # unusable on short blocks.
    "measure_power_secs": 1,
    "save_to_file": True,
    "allow_multiple_runs": True,
    "log_level": "error",
}

#: Minimum CodeCarbon version. 3.x is required so that GPU energy comes from
#: the NVML energy counter rather than integrated instantaneous samples.
REQUIRED_CODECARBON = (3, 0)

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
#: Precision is pinned rather than left to each ecosystem's default. Mixed
#: precision alone can change energy by roughly a factor of two, so leaving it
#: to the default made "ecosystem" and "precision policy" inseparable.
DEFAULT_PRECISION = "fp32"

#: Whether TF32 tensor cores may serve fp32 convolutions and matmuls.
#:
#: This is the single most expensive setting in the study and it was, for one
#: campaign, the least visible. The pin lived here and here only, and this
#: module is imported by the six Python model files and by nothing else, so
#: Python/PyTorch ran true-fp32 convolutions while the other six stacks took
#: cuDNN's Ampere default. Measured on an RTX 3090, ResNet-18, batch 128:
#: turning TF32 off costs 4.81x the GPU energy and 3.46x the step time. The
#: campaign's PyTorch-vs-C++ gap was 3.24-3.79x. One flag, not a binding.
#:
#: So it is a campaign-wide policy now, read from one variable by every stack,
#: with the value recorded per run. "1" allows TF32 -- what every framework
#: does by default on Ampere, and what a practitioner gets without touching
#: anything. "0" pins true fp32, which is what S4 asked for and what no stack
#: except one was doing; it costs roughly 138 h of machine time against 57 h.
#: scripts/campaign_env.sh sets this and exports NVIDIA_TF32_OVERRIDE to match,
#: which is how the C++, Rust, R and Java processes -- which never import this
#: module -- obey the same policy through the CUDA libraries.
TF32_ENV = "DEEPGREEN_TF32"


def tf32_allowed() -> bool:
    """The campaign's TF32 policy, defaulting to the frameworks' own default."""
    return os.environ.get(TF32_ENV, "1") == "1"


def set_precision_policy() -> None:
    """Apply the campaign's precision policy to whichever frameworks are here.

    Deliberately does *not* set cudnn.deterministic. Pinning it in one stack and
    not the others was the second half of the same defect: it forces cuDNN away
    from its fastest algorithms and cost a further 1.35x, and it cannot be set
    at all in DL4J or R. Algorithm selection is therefore left to every stack
    alike, and run-to-run variation includes it.
    """
    allow = tf32_allowed()
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = allow
        torch.backends.cudnn.allow_tf32 = allow
    except ImportError:
        pass
    try:
        import tensorflow as tf

        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("float32")   # fp16 is a separate axis
        tf.config.experimental.enable_tensor_float_32_execution(allow)
    except ImportError:
        pass
    try:
        import jax

        # JAX's "default" matmul precision is the fastest the device offers,
        # which on Ampere is TF32 -- so this needs saying either way.
        jax.config.update("jax_default_matmul_precision",
                          "tensorfloat32" if allow else "float32")
    except ImportError:
        pass


@dataclass
class RunContext:
    """Everything that identifies one independent run."""

    ecosystem: str
    model: str
    dataset: str
    repetition: int = 0
    seed: int | None = None
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    precision: str = DEFAULT_PRECISION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed is None:
            # Distinct but reproducible seed per repetition.
            self.seed = 1000 + self.repetition

    @property
    def slug(self) -> str:
        eco = self.ecosystem.replace("/", "-").replace("+", "p")
        return f"{eco}_{self.model}_{self.dataset}_rep{self.repetition}"

    @property
    def out_dir(self) -> Path:
        """Where this run writes.

        DEEPGREEN_RUN_DIR is the run contract: the campaign driver sets it, and
        tools/deepgreen_tracker.py -- the path the non-Python stacks take --
        has always honoured it. This path, taken by the Python stacks in
        process, hardcoded the campaign directory and ignored it. The two
        disagreed silently, so redirecting the driver's output moved the lock
        and the plan and left the Python runs writing over the campaign they
        were supposed to be measured against. They did exactly that, once.

        One variable, read the same way on both paths, with the campaign
        directory as the fallback so the default is unchanged.
        """
        override = os.environ.get("DEEPGREEN_RUN_DIR")
        if override:
            return Path(override)
        root = os.environ.get("DEEPGREEN_CAMPAIGN_DIR")
        base = Path(root) if root else REPO_ROOT / "results" / "campaign_v2"
        return base / self.slug


class Harness:
    """Pinned CodeCarbon tracking plus per-epoch quality logging."""

    def __init__(self, ctx: RunContext, out_dir: Path | None = None):
        self.ctx = ctx
        self.out_dir = Path(out_dir) if out_dir else ctx.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.out_dir / "metrics.csv"
        self._metrics_fh = self._metrics_path.open("w", newline="")
        self._metrics_writer: csv.DictWriter | None = None
        self._counters = None
        try:
            from tools.hardware_counters import HardwareCounters

            counters = HardwareCounters()
            if any(counters.available.values()):
                self._counters = counters
        except Exception as e:  # never let the second instrument break a run
            print(f"[deepgreen] hardware counters unavailable: {e}", file=sys.stderr)
        self._write_manifest()
        self._assert_accelerator()

    # -- environment -----------------------------------------------------
    def _write_manifest(self) -> None:
        manifest = {
            "run": asdict(self.ctx),
            "codecarbon_config": CODECARBON_CONFIG,
            "python": sys.version,
            "platform": platform.platform(),
            "env": {
                k: os.environ.get(k)
                for k in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_HOME",
                    "LD_LIBRARY_PATH",
                    "OMP_NUM_THREADS",
                    "TF_ENABLE_ONEDNN_OPTS",
                    "XLA_FLAGS",
                )
            },
            "nvidia_smi": _safe_cmd(
                ["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"]
            ),
            "codecarbon_version": _pkg_version("codecarbon"),
            "framework_versions": _framework_versions(),
            "accelerator": detect_accelerator(self.ctx.ecosystem),
            "hardware_counters": (self._counters.describe()
                                  if self._counters is not None else None),
            "machine_state": machine_state(),
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    def _assert_accelerator(self) -> None:
        """Refuse to measure a stack that silently fell back to the CPU.

        Set DEEPGREEN_ALLOW_CPU=1 for a deliberate CPU run.
        """
        acc = detect_accelerator(self.ctx.ecosystem)
        if acc.get("gpu_visible"):
            print(f"[deepgreen] {acc['framework']} on {acc['devices']}")
            return
        msg = (
            f"{self.ctx.ecosystem}: {acc.get('framework') or 'the framework'} does not see a GPU "
            f"(devices: {acc.get('devices')}). Measuring it now would attribute a CPU fallback "
            "to the ecosystem."
        )
        if acc.get("build_compute_capabilities"):
            msg += f" Build supports {acc['build_compute_capabilities']}; check it covers this card."
        if os.environ.get("DEEPGREEN_ALLOW_CPU") == "1":
            print("[deepgreen] WARNING " + msg)
            return
        raise RuntimeError(msg + " Set DEEPGREEN_ALLOW_CPU=1 to override.")

    # -- reproducibility -------------------------------------------------
    def set_seeds(self) -> None:
        seed = int(self.ctx.seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass
        set_precision_policy()

    # -- measurement -----------------------------------------------------
    @contextmanager
    def track(self, phase: str, epoch: int):
        """Track one epoch of one phase with both instruments.

        CodeCarbon and the hardware counters run over the same window, so the
        two can be compared per block rather than assumed to agree. CodeCarbon's
        CPU and RAM terms are models; RAPL and NVML are counters.
        """
        from codecarbon import EmissionsTracker

        _check_codecarbon_version()
        hw = self._counters
        tracker = EmissionsTracker(
            output_dir=str(self.out_dir),
            output_file=f"emissions_{phase}_epoch{epoch}.csv",
            project_name=self.ctx.slug,
            **CODECARBON_CONFIG,
        )
        tracker.start()
        # Snapshot INSIDE the tracker window. Taken outside, the hardware window
        # also covers CodeCarbon's own start-up and shutdown, which on a
        # four-second inference block is most of the measurement: comparing the
        # two then reports a 70% discrepancy that is entirely an artefact of the
        # comparison. Bracketed properly, the two instruments agree to within 1%
        # from 2 s to 40 s (scripts/validate_instrument.py).
        hw_before = hw.snapshot() if hw is not None else None
        try:
            yield tracker
        finally:
            hw_after = hw.snapshot() if hw is not None else None
            tracker.stop()
            if hw is not None and hw_before is not None:
                self._write_counters(phase, epoch, hw.delta(hw_before, hw_after))

    # -- quality ---------------------------------------------------------
    def log_metrics(self, epoch: int, **metrics: float) -> None:
        """Persist per-epoch model quality.

        The first campaign computed test accuracy in most ecosystems but only
        printed it, so energy could never be normalised by the useful work
        produced. Every ecosystem must now write it here.
        """
        row = {
            "ecosystem": self.ctx.ecosystem,
            "model": self.ctx.model,
            "dataset": self.ctx.dataset,
            "repetition": self.ctx.repetition,
            "seed": self.ctx.seed,
            "precision": self.ctx.precision,
            "epoch": epoch,
            **{k: (None if v is None else float(v)) for k, v in metrics.items()},
        }
        if self._metrics_writer is None:
            self._metrics_writer = csv.DictWriter(self._metrics_fh, fieldnames=list(row))
            self._metrics_writer.writeheader()
        self._metrics_writer.writerow(row)
        self._metrics_fh.flush()

    def _write_counters(self, phase: str, epoch: int, delta: dict) -> None:
        """Append one row of hardware-counter energy for this block."""
        path = self.out_dir / "counters.csv"
        row = {"phase": phase, "epoch": epoch} | {k: round(v, 6) for k, v in delta.items()}
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def close(self) -> None:
        self._metrics_fh.close()

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def machine_state() -> dict[str, object]:
    """The mutable machine state a replicator would need and never had.

    Every manifest in the first replicated campaign recorded the software and
    none of the hardware's settings -- no power limit, no clock policy, no
    persistence mode, no ECC state, no governor, no driver version, not even a
    timestamp, so run order had to be recovered from file mtimes. That mattered
    more than it sounds: one configuration's evaluation blocks turned out to be
    bimodal in accelerator power, 161 W against 200 W at identical duration,
    and with persistence mode unrecorded there is no way to tell from the
    artefact whether the card was dropping out of its performance state.

    Written by both output paths, so the four stacks that never import this
    module are described as fully as the three that do.
    """
    gpu = _safe_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,power.limit,persistence_mode,ecc.mode.current,"
        "clocks.max.sm,clocks.sm,temperature.gpu,pstate",
        "--format=csv,noheader",
    ])
    return {
        "utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "gpu": gpu,
        "cpu_governor": _safe_cmd(
            ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]),
        "cpu_boost": _safe_cmd(["cat", "/sys/devices/system/cpu/cpufreq/boost"]),
        "precision_policy": {
            # The policy as asked for, and as the driver and framework report it.
            # These can disagree, and which one is true is not obvious: with
            # NVIDIA_TF32_OVERRIDE=0 torch still reports cudnn.allow_tf32 True
            # while executing fp32 kernels. Record all three and let the reader
            # see the disagreement rather than inferring one from another.
            TF32_ENV: os.environ.get(TF32_ENV),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "torch_cudnn_allow_tf32": _torch_tf32_flag(),
        },
    }


def _torch_tf32_flag() -> bool | None:
    try:
        import torch

        return bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        return None


def _safe_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def detect_accelerator(ecosystem: str | None = None) -> dict[str, object]:
    """What device will this process actually compute on?

    A framework that cannot see the GPU falls back to the CPU and reports
    nothing but a warning. Its energy is then attributed to the ecosystem when it
    is really a packaging or hardware-support accident. TensorFlow 2.21, for
    instance, ships compute capabilities sm_60/70/80/89 and compute_90: on an
    RTX 3090 (sm_86) it runs on the CPU while printing only "Cannot dlopen some
    GPU libraries". In the first campaign the TensorFlow stack had the lowest
    mean GPU power of all eight (113 W training against a 350 W card), which is
    exactly what a partial CPU fallback looks like.
    """
    info: dict[str, object] = {"framework": None, "gpu_visible": False, "devices": []}

    # Check the framework this ecosystem actually computes with, not whichever
    # one imports first: the JAX venv also carries tensorflow-cpu for the shared
    # tf.data loader, and probing that would report "no GPU" for a stack whose
    # compute device is fine.
    wanted = {
        "Python/PyTorch": "torch",
        "Python/TensorFlow": "tensorflow",
        "Python/JAX": "jax",
    }.get(ecosystem or "")

    if wanted == "jax":
        return _jax_accelerator(info)
    if wanted == "tensorflow":
        return _tensorflow_accelerator(info)
    if wanted == "torch":
        return _torch_accelerator(info)

    try:
        import torch

        info["framework"] = "torch"
        info["gpu_visible"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["devices"] = [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())]
            info["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
        return info
    except ImportError:
        pass
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        info["framework"] = "tensorflow"
        info["gpu_visible"] = bool(gpus)
        info["devices"] = [g.name for g in gpus]
        from tensorflow.python.platform import build_info as bi

        info["build_compute_capabilities"] = bi.build_info.get("cuda_compute_capabilities")
        return info
    except ImportError:
        pass
    try:
        import jax

        devs = jax.devices()
        info["framework"] = "jax"
        info["gpu_visible"] = any(d.platform in ("gpu", "cuda") for d in devs)
        info["devices"] = [str(d) for d in devs]
    except ImportError:
        pass
    return info


def _torch_accelerator(info: dict[str, object]) -> dict[str, object]:
    import torch

    info["framework"] = "torch"
    info["gpu_visible"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
        info["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
    return info


def _tensorflow_accelerator(info: dict[str, object]) -> dict[str, object]:
    import tensorflow as tf
    from tensorflow.python.platform import build_info as bi

    gpus = tf.config.list_physical_devices("GPU")
    info["framework"] = "tensorflow"
    info["gpu_visible"] = bool(gpus)
    info["devices"] = [g.name for g in gpus]
    info["build_compute_capabilities"] = bi.build_info.get("cuda_compute_capabilities")
    return info


def _jax_accelerator(info: dict[str, object]) -> dict[str, object]:
    import jax

    devs = jax.devices()
    info["framework"] = "jax"
    info["gpu_visible"] = any(d.platform in ("gpu", "cuda") for d in devs)
    info["devices"] = [str(d) for d in devs]
    return info


def _framework_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for mod in ("torch", "torchvision", "tensorflow", "jax", "jaxlib", "flax", "numpy"):
        out[mod] = _pkg_version(mod)
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["cudnn"] = str(torch.backends.cudnn.version())
    except Exception:
        pass
    return out


def _check_codecarbon_version() -> None:
    v = _pkg_version("codecarbon")
    if v is None:
        return
    parts = tuple(int(p) for p in v.split(".")[:2] if p.isdigit())
    if parts and parts < REQUIRED_CODECARBON:
        raise RuntimeError(
            f"codecarbon {v} is installed but >= {'.'.join(map(str, REQUIRED_CODECARBON))} is "
            "required: 2.x falls back to a hardcoded constant CPU power (85 W x 0.5) when RAPL is "
            "unavailable and models RAM power from installed memory, which makes ~2/3 of the reported energy a "
            "deterministic function of wall-clock time. See results/analysis/01_data_audit.py."
        )


if __name__ == "__main__":
    ctx = RunContext(ecosystem="Python/PyTorch", model="resnet18", dataset="cifar100")
    print(json.dumps(asdict(ctx), indent=2))
    print("slug:", ctx.slug)
    print("codecarbon config:", json.dumps(CODECARBON_CONFIG, indent=2))
