import os
from codecarbon import EmissionsTracker

tracker = None
_last_path = None

def start_tracker(output_dir, output_file):
    global tracker, _last_path
    os.makedirs(output_dir, exist_ok=True)
    _last_path = os.path.join(output_dir, output_file)
    tracker = EmissionsTracker(
        output_dir=output_dir,
        output_file=output_file,
        measure_power_secs=1,
        save_to_file=True,
        log_level="error",
        tracking_mode="process",
        allow_multiple_runs=True,   # <<<< evita /tmp/.codecarbon.lock
    )
    tracker.start()
    print(f"[CodeCarbon] START -> {os.path.abspath(_last_path)}", flush=True)

def stop_tracker():
    global tracker, _last_path
    if tracker is None:
        print("[CodeCarbon] STOP (no tracker)", flush=True)
        return 0.0
    try:
        emissions = tracker.stop()
    finally:
        tracker = None
    val = 0.0 if emissions is None else float(emissions)
    print(f"[CodeCarbon] STOP -> {val} kg | file={os.path.abspath(_last_path)}", flush=True)
    return val
