# Host-specific roots for the replicated campaign. Source before run_campaign.py:
#
#     source scripts/campaign_env.sh && python3 scripts/run_campaign.py --repetitions 5
#
# These are the four paths that differ between machines; everything else is
# derived in tools/stack_environments.json.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DEEPGREEN_DATA="$REPO/data"
export DEEPGREEN_MODELS="$REPO/models"
export DEEPGREEN_PYTHON="$REPO/.venv-deepgreen/bin/python"

# CUDA toolkit whose runtime the C++ and R stacks load
export DEEPGREEN_CUDA="${DEEPGREEN_CUDA:-$HOME/miniforge3/envs/dg-cuda128}"
# conda environment providing OpenCV, the JDK and R
export DEEPGREEN_CONDA="${DEEPGREEN_CONDA:-$HOME/miniforge3/envs/deepgreen}"
# CUDA LibTorch the Rust crate links against.
#
# Defaults to the one the C++ build fetches into the repository, which is the
# same 2.7.0+cu128 the modules are exported from. It used to be a required
# variable, and in the campaign that ran it pointed at a scratch directory under
# /tmp: the manifests recorded that path, it did not survive, and the Rust
# binaries built against it now fail to resolve libtorch_cuda.so. A replication
# instruction that names a temporary directory is not one.
export DEEPGREEN_LIBTORCH="${DEEPGREEN_LIBTORCH:-$REPO/cpp/tools/libtorch/libtorch_linux_cuda-src}"
if [ ! -f "$DEEPGREEN_LIBTORCH/lib/libtorch_cuda.so" ]; then
  echo "campaign_env.sh: no CUDA LibTorch at $DEEPGREEN_LIBTORCH." >&2
  echo "  Configure the C++ build once (it fetches 2.7.0+cu128 into the repo)," >&2
  echo "  or set DEEPGREEN_LIBTORCH to a CUDA build of the version in" >&2
  echo "  models/MANIFEST.json." >&2
fi
# Deliberately NOT put on the global LD_LIBRARY_PATH. Rust and C++ get it from
# tools/stack_environments.json, per stack, which is what that file is for.
# Exporting it here instead makes LibTorch's bundled cuDNN 9 shadow the one in
# JAX's virtualenv, and JAX dies with a segfault and "Unable to load any of
# {libcudnn_engines_runtime_compiled.so...}" -- found by running one epoch of
# each stack rather than by reading, which is the only way this kind of thing
# is ever found.
# R library tree holding the torch package and its bundled LibTorch
export DEEPGREEN_R_LIBS="${DEEPGREEN_R_LIBS:-$HOME/R/deepgreen}"

export PATH="$DEEPGREEN_CONDA/bin:$PATH"

# --- precision policy, campaign-wide -----------------------------------------
#
# The one setting that has to be the same in all seven stacks, because it is
# worth more than any of them: on an RTX 3090, ResNet-18, batch 128, denying
# fp32 convolutions the tensor cores costs 4.81x the GPU energy. For one
# campaign this was pinned in tools/deepgreen_bench.py alone -- a module only
# the three Python stacks import -- so Python/PyTorch ran true fp32 and the
# other six took cuDNN's Ampere default, and the difference was read as binding
# overhead.
#
# DEEPGREEN_TF32 is the policy; the Python stacks apply it through their
# frameworks. NVIDIA_TF32_OVERRIDE is the same policy stated to the CUDA
# libraries, which is the only lever the C++, Rust, R and Java processes have --
# they never import the harness. Measured equivalent: setting the variable alone
# reproduces the flag-based figures to within 0.5%.
#
# Note that the two disagree out loud when the driver overrides: with
# NVIDIA_TF32_OVERRIDE=0 torch still reports cudnn.allow_tf32 == True while
# executing fp32 kernels. The framework flag is not the source of truth, which
# is why the conformance check reads this variable and not the flag.
#
#   1  allow TF32 -- every framework's own default on Ampere, and what a
#      practitioner gets without touching anything. ~57 h for 210 runs.
#   0  pin true fp32, as S4 originally asked. ~138 h for the same 210 runs.
export DEEPGREEN_TF32="${DEEPGREEN_TF32:-1}"
export NVIDIA_TF32_OVERRIDE="$DEEPGREEN_TF32"
