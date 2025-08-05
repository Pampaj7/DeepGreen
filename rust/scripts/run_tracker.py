# scripts/run_tracker.py
import argparse
import tracker_control

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "stop"])
    parser.add_argument("--output_dir", default="emissions")
    parser.add_argument("--output_file", default="default.csv")
    args = parser.parse_args()

    if args.command == "start":
        tracker_control.start_tracker(args.output_dir, args.output_file)
    elif args.command == "stop":
        tracker_control.stop_tracker()
