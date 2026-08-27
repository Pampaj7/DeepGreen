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
# CUDA LibTorch the Rust crate was linked against
export DEEPGREEN_LIBTORCH="${DEEPGREEN_LIBTORCH:?set DEEPGREEN_LIBTORCH to the CUDA LibTorch used by scripts/build_rust_cuda.sh}"
# R library tree holding the torch package and its bundled LibTorch
export DEEPGREEN_R_LIBS="${DEEPGREEN_R_LIBS:-$HOME/R/deepgreen}"

export PATH="$DEEPGREEN_CONDA/bin:$PATH"
