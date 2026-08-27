#!/usr/bin/env bash
# Prepare a machine to run the replicated DeepGreen campaign.
#
#   ./scripts/setup_environment.sh [--datasets-only] [--stack pytorch|tensorflow|jax|all]
#
# Creates .venv-deepgreen, installs the pinned instrument plus one or more DL
# stacks, and materialises the three datasets as PNG folders under $DEEPGREEN_DATA
# (default ./data).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
export DEEPGREEN_DATA="${DEEPGREEN_DATA:-$REPO/data}"

STACK=all
DATASETS_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets-only) DATASETS_ONLY=1; shift ;;
    --stack) STACK="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- 0. hard preconditions ---------------------------------------------------
if ! nvidia-smi >/dev/null 2>&1; then
  cat >&2 <<'MSG'
ERROR: nvidia-smi cannot talk to the driver.

If the GPU is present in lspci but /dev/nvidia* is missing, the kernel module is
not loaded. That needs root:

    sudo modprobe nvidia nvidia_uvm nvidia_modeset     # or simply reboot

Energy measurement depends on NVML, so the campaign cannot start without it.
MSG
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,power.limit,memory.total --format=csv

# --- 1. environment ----------------------------------------------------------
if [[ ! -d .venv-deepgreen ]]; then
  python3 -m venv .venv-deepgreen
fi
PY=".venv-deepgreen/bin/python"
$PY -m pip install -q --upgrade pip

# CodeCarbon >= 3 is mandatory: 2.x substitutes a constant for CPU power and
# derives RAM power from installed memory, which makes ~2/3 of the reported
# energy a function of wall-clock time. See results/analysis/01_data_audit.py.
$PY -m pip install -q "codecarbon>=3.0" pandas numpy scipy matplotlib seaborn tabulate tqdm

case "$STACK" in
  # Spec S4: one LibTorch behind the four LibTorch-based ecosystems.
  # 2.7.0 is what cpp/CMakeLists.txt fetches and what tch 0.20 links against.
  pytorch|all)    $PY -m pip install -q "torch==2.7.0" "torchvision==0.22.0" \
                      --index-url https://download.pytorch.org/whl/cu128 ;;&
  tensorflow|all) $PY -m pip install -q "tensorflow[and-cuda]" tf-keras ;;&
  jax|all)        $PY -m pip install -q "jax[cuda12]" flax optax ;;&
esac

$PY - <<'PYCHECK'
import importlib
for m in ("torch", "codecarbon"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  {m:12s} NOT INSTALLED")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
except Exception as e:
    print("  torch check failed:", e)
PYCHECK

# --- 2. datasets -------------------------------------------------------------
mkdir -p "$DEEPGREEN_DATA"
echo "materialising datasets under $DEEPGREEN_DATA"
[[ -d "$DEEPGREEN_DATA/cifar100_png"      ]] || $PY dataloader/download_convert_cifar100.py
[[ -d "$DEEPGREEN_DATA/fashion_mnist_png" ]] || $PY dataloader/download_convert_fashion.py
[[ -d "$DEEPGREEN_DATA/tiny_imagenet_png" ]] || $PY dataloader/download_convert_tinyimage.py
du -sh "$DEEPGREEN_DATA"/*_png 2>/dev/null || true

# --- 3. shared TorchScript modules ------------------------------------------
# The C++, Rust and R stacks load these instead of building their own models.
$PY scripts/export_torchscript_models.py

[[ "$DATASETS_ONLY" == 1 ]] && exit 0

cat <<MSG

Ready. Next:

    PYTHONPATH=. .venv-deepgreen/bin/python scripts/run_campaign.py --repetitions 5 --print-plan
    PYTHONPATH=. .venv-deepgreen/bin/python scripts/run_campaign.py --repetitions 5 \\
        --ecosystems Python/PyTorch Python/TensorFlow Python/JAX

Then:

    .venv-deepgreen/bin/python results/analysis/09_campaign_v2.py

Note: this machine is not the machine the first campaign ran on. Absolute
energies are not comparable with results/data/combined_data.csv; treat a run
here as an independent replication on different hardware, which is a useful
external-validity check in its own right.
MSG
