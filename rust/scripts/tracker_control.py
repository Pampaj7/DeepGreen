# scripts/tracker_control.py
from codecarbon import EmissionsTracker

tracker = None

def start_tracker(output_dir, output_file):
    global tracker
    tracker = EmissionsTracker(output_dir=output_dir, output_file=output_file)
    tracker.start()
    print(f"[CodeCarbon] Tracker started: {output_file}", flush=True)

def stop_tracker():
    global tracker
    if tracker is not None:
        emissions = tracker.stop()
        print(f"[CodeCarbon] Tracker stopped. Emissions: {emissions} kg CO2", flush=True)
        tracker = None
    else:
        print("[CodeCarbon] Tracker was not running.", flush=True)
