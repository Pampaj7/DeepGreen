# scripts/tracker_daemon.py
import sys
import os
import logging
from tracker_control import start_tracker, stop_tracker

# Zittisce CodeCarbon completamente
logging.getLogger("codecarbon").setLevel(logging.ERROR)

logging.getLogger().setLevel(logging.ERROR)

print("[Daemon] Ready to receive commands", flush=True)

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue

    if line.startswith("START"):
        try:
            _, out_dir, out_file = line.split(" ", 2)
            full_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", out_dir))
            os.makedirs(full_dir, exist_ok=True)
            start_tracker(full_dir, out_file)
        except Exception as e:
            print(f"[Daemon] ERROR during START: {e}", flush=True)

    elif line == "STOP":
        try:
            stop_tracker()
        except Exception as e:
            print(f"[Daemon] ERROR during STOP: {e}", flush=True)

    elif line == "EXIT":
        break
