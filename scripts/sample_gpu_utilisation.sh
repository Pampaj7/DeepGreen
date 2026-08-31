#!/usr/bin/env bash
# Record GPU utilisation at 1 Hz alongside the campaign.
#
# counters.csv records energy and duration per block, not how busy the device
# was. R draws 128 W where five stacks draw 260-291 W, and the reason turned out
# to be 5.7% utilisation rather than a CPU fallback -- a distinction the recorded
# columns cannot make. This samples independently so the finished campaign can
# report utilisation beside energy, joining on time: manifest.json's mtime marks
# a run's start, counters.csv's its end.
#
# 1 Hz against the tracker's own 10 Hz NVML polling; the added load is noise.
out="${1:-results/gpu_utilisation.csv}"
mkdir -p "$(dirname "$out")"
[ -s "$out" ] || echo "unix_s,utilisation_pct,power_w,sm_clock_mhz,memory_used_mib" > "$out"
while :; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used \
           --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || true
  [ -n "$line" ] && echo "$(date +%s),$line" >> "$out"
  sleep 1
done
