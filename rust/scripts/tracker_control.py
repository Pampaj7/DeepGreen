# scripts/tracker_control.py
from codecarbon import EmissionsTracker
import time

tracker = None

def start_tracker(output_dir, output_file, measure_power_secs=15):
    global tracker
    tracker = EmissionsTracker(
        output_dir=output_dir,
        output_file=output_file,
        measure_power_secs=measure_power_secs,  # 🔑 sampling interval
        save_to_file=True,
    )
    tracker.start()
    print(f"[CodeCarbon] Tracker started: {output_file} (sampling={measure_power_secs}s)", flush=True)

def stop_tracker():
    global tracker
    if tracker is not None:
        emissions = tracker.stop()
        print(f"[CodeCarbon] Tracker stopped. Emissions: {emissions} kg CO2", flush=True)
        tracker = None
    else:
        print("[CodeCarbon] Tracker was not running.", flush=True)

