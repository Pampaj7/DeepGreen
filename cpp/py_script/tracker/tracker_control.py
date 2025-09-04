import os
from codecarbon import EmissionsTracker

tracker = None

class Tracker:
    @staticmethod
    def start_tracker(output_dir, output_file):
        global tracker
        os.makedirs(output_dir, exist_ok=True)
        tracker = EmissionsTracker(output_dir=output_dir, output_file=output_file)
        tracker.start()
        print(f"Tracker started.")

    @staticmethod
    def stop_tracker():
        global tracker
        if tracker is not None:
            emissions = tracker.stop()
            print(f"Tracker stopped. Emissions: {emissions} kg CO2")
            tracker = None
        else:
            print("Tracker was not running.")