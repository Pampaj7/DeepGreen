#!/usr/bin/env bash
# Post-campaign sequence: rebuild Rust, re-run the invalidated configuration,
# then regenerate the analysis and the manuscript.
#
# The five Rust vgg16/fashionmnist runs are invalid: vgg_fashion.rs applied a
# private input transform on the training path only, which resized an
# already-float tensor with a function returning uint8 and so trained the
# network on black images. The source is fixed; the binary was deliberately NOT
# rebuilt while the campaign was measuring, both to avoid contaminating live
# measurements with a compile and so that all five re-runs share one binary.
#
#   ./scripts/finish_campaign.sh            # everything
#   ./scripts/finish_campaign.sh --analysis-only
set -euo pipefail
cd "$(dirname "$0")/.."

SCRATCH="${DEEPGREEN_SCRATCH:-/tmp/claude-1000/-home-pampaj-Desktop/d86a4516-a091-47a0-8271-eade8d18981b/scratchpad}"
export PATH="$HOME/.cargo/bin:$PATH"
export DEEPGREEN_LIBTORCH="${DEEPGREEN_LIBTORCH:-$SCRATCH/libtorch27cu/libtorch}"
PY="$PWD/.venv-deepgreen/bin/python"

if [[ "${1:-}" != "--analysis-only" ]]; then
  # Match the interpreter running the driver, not any shell whose command line
  # merely mentions it -- a waiter built around `pgrep -f run_campaign.py`
  # matched itself and would have waited forever.
  if pgrep -af "python.* scripts/run_campaign\.py" | grep -qv "finish_campaign"; then
    echo "A campaign is still running. Refusing to rebuild underneath it." >&2
    exit 1
  fi

  echo "=== conformance ==="
  "$PY" scripts/check_consistency.py | tail -n 3

  echo "=== rebuilding the Rust binaries ==="
  LIBTORCH="$DEEPGREEN_LIBTORCH" ./scripts/build_rust_cuda.sh | tail -n 4

  echo "=== discarding the invalidated runs ==="
  rm -rfv results/campaign_v2/Rust-tch_vgg16_fashionmnist_rep* | sed 's/^/  /'

  echo "=== re-running them ==="
  source scripts/campaign_env.sh
  "$PY" -u scripts/run_campaign.py --repetitions 5 \
      --ecosystems Rust/tch --models vgg16 --datasets fashionmnist \
      2>&1 | tee -a "$SCRATCH/campaign9_vggfashion.log" | grep -E "^\[|FAILED|failed"
fi

echo "=== analysis, numbers and figures ==="
PYTHON="$PY" ./results/analysis/run_all.sh | grep -E "^###|excluded|!!|wrote paper"

echo "=== replication package ==="
# The raw tree is 13,000 files and 54 MB and is not distributed; these four
# gzipped tables carry the identical records at 2 MB and are.
"$PY" scripts/consolidate_raw.py

echo "=== manuscript ==="
./paper/build.sh --no-data
