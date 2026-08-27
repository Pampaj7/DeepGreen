#!/usr/bin/env bash
# Build the Rust ecosystem against a CUDA LibTorch.
#
#   LIBTORCH=/path/to/libtorch-2.7.0+cu128 ./scripts/build_rust_cuda.sh
#
# Why this script exists: `cargo build` against a CUDA LibTorch produces a
# BINARY THAT SILENTLY RUNS ON THE CPU. torch-sys asks the linker for
# torch_cuda, but nothing in the Rust code references a symbol from it, so the
# default --as-needed behaviour drops the dependency; tch::Device::cuda_if_available()
# then reports Cpu and the whole campaign measures CPU training while looking
# entirely normal. torch-sys's own build.rs documents the problem in a comment
# and does not solve it.
#
# Verify with scripts/../rust/examples/load_shared_module.rs: it must print
# "device: Cuda(0)".
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LIBTORCH:?set LIBTORCH to a CUDA-enabled LibTorch (must match models/MANIFEST.txt)}"

if [[ ! -f "$LIBTORCH/lib/libtorch_cuda.so" ]]; then
  echo "ERROR: $LIBTORCH is a CPU-only LibTorch (no libtorch_cuda.so)." >&2
  exit 1
fi
echo "LibTorch: $(cat "$LIBTORCH/build-version" 2>/dev/null || echo unknown)"

export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:-cu128}"
export DEEPGREEN_MODELS="${DEEPGREEN_MODELS:-$PWD/models}"
# The linker must keep torch_cuda even though no Rust symbol pulls it in.
export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed -C link-arg=-L$LIBTORCH/lib -C link-arg=-ltorch_cuda -C link-arg=-lc10_cuda -C link-arg=-ltorch"

cargo build --release --manifest-path rust/Cargo.toml --bins --examples

echo
echo "linked CUDA libraries: $(readelf -d rust/target/release/examples/load_shared_module | grep -c 'NEEDED.*cuda')"
cargo run --release --manifest-path rust/Cargo.toml --example load_shared_module
