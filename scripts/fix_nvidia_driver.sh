#!/usr/bin/env bash
# Diagnose and repair the NVIDIA driver on this machine.
#
#   sudo ./scripts/fix_nvidia_driver.sh            # diagnose + fix
#   ./scripts/fix_nvidia_driver.sh --diagnose      # diagnose only, no root needed
#
# Findings on 2026-08-18:
#   running kernel      7.0.0-29-generic
#   nvidia modules for  7.0.0-22, -27, -28   (prebuilt Ubuntu packages)
#   nvidia modules for  7.0.0-29             MISSING
#   meta package linux-modules-nvidia-595-open-generic-hwe-26.04 pinned at 7.0.0-28.28
#
# The machine booted a kernel newer than the NVIDIA module package that was
# installed for it. Ubuntu ships prebuilt per-ABI modules here (not DKMS), so
# `modprobe nvidia` cannot work: there is no nvidia.ko under the running kernel.
set -euo pipefail

KERNEL="$(uname -r)"
DRIVER_BRANCH="595-open"
MODPKG="linux-modules-nvidia-${DRIVER_BRANCH}-${KERNEL}"

diagnose() {
  echo "kernel in esecuzione : $KERNEL"
  echo -n "GPU su PCI           : "; lspci 2>/dev/null | grep -i "vga.*nvidia" || echo "nessuna"
  echo -n "moduli nvidia per il kernel corrente: "
  local n; n=$(find "/lib/modules/$KERNEL" -name "nvidia.ko*" 2>/dev/null | wc -l)
  [[ "$n" -gt 0 ]] && echo "presenti" || echo "ASSENTI  <-- causa"
  echo "kernel con moduli nvidia disponibili:"
  for k in /lib/modules/*/; do
    k="$(basename "$k")"
    [[ -n "$(find "/lib/modules/$k" -name 'nvidia.ko*' 2>/dev/null)" ]] && echo "  $k"
  done
  echo -n "modulo caricato      : "; lsmod | grep -q '^nvidia' && echo "si" || echo "no"
  echo -n "nvidia-smi           : "; nvidia-smi >/dev/null 2>&1 && echo "ok" || echo "fallisce"
  echo -n "Secure Boot          : "; mokutil --sb-state 2>/dev/null | head -1 || echo "sconosciuto"
}

diagnose
[[ "${1:-}" == "--diagnose" ]] && exit 0

if [[ "$(id -u)" -ne 0 ]]; then
  echo
  echo "Serve root per la riparazione. Rilancia con: sudo $0" >&2
  exit 1
fi

echo
echo "installo $MODPKG ..."
apt-get update -qq
apt-get install -y "$MODPKG"
# Keep it self-maintaining across future kernel upgrades.
apt-get install -y "linux-modules-nvidia-${DRIVER_BRANCH}-generic-hwe-26.04" || true

echo "carico i moduli ..."
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_modeset

echo
nvidia-smi --query-gpu=name,driver_version,power.limit,memory.total --format=csv
echo
echo "OK. Ora: ./scripts/setup_environment.sh"
