# R/scripts/run_tracker.py
import argparse
import sys
import os
import warnings

# zittisce FutureWarning di pandas in codecarbon
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# garantisce import locali
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import tracker_control  # dopo aver fissato sys.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "stop"])
    parser.add_argument("--output_dir", default="emissions")
    parser.add_argument("--output_file", default="default.csv")
    args = parser.parse_args()

    if args.command == "start":
        os.makedirs(args.output_dir, exist_ok=True)
        tracker_control.start_tracker(args.output_dir, args.output_file)
    # R/scripts/run_tracker.py (solo la branch stop)
    elif args.command == "stop":
        val = tracker_control.stop_tracker()
        # stampa anche in forma parsabile
        print(f"EMISSIONS:{val}", flush=True)

