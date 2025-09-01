# R/scripts/tracker_daemon.py
import sys, os, logging, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# garantisce import locali
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from tracker_control import start_tracker, stop_tracker

logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

print("[Daemon] Ready", flush=True)

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue

    if line.startswith("START"):
        try:
            _, _out_dir, out_file = line.split(" ", 2)
            # Always use DeepGreen/R/emissions
            emissions_dir = os.path.abspath(os.path.join(THIS_DIR, '..', 'emissions'))
            os.makedirs(emissions_dir, exist_ok=True)
            print(f"[Daemon] WRITING {os.path.join(emissions_dir, out_file)}", flush=True)
            start_tracker(emissions_dir, out_file)
            print("[Daemon] STARTED", flush=True)
        except Exception as e:
            print(f"[Daemon] ERROR START: {e}", flush=True)

    # R/scripts/tracker_daemon.py  (solo ramo STOP)
    elif line == "STOP":
        try:
            kg = stop_tracker()
            val = 0.0 if kg is None else float(kg)
            print(f"[Daemon] STOPPED {val:.9f}", flush=True)
        except Exception as e:
            print(f"[Daemon] ERROR STOP: {e}", flush=True)


    elif line == "EXIT":
        print("[Daemon] BYE", flush=True)
        break
