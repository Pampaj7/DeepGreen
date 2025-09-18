import os
import sys
from codecarbon import EmissionsTracker
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

tracker = None
output_dir = os.path.abspath("/home/pampaj/DeepGreen/java/DL4J")  # Use absolute path for output directory


def start_tracker(output_file):
    global tracker
    if tracker is None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            #if os.path.exists(os.path.join(output_dir, output_file)):
            #    os.remove(os.path.join(output_dir, output_file))
            #    print(f"File {output_file} already existed and now is removed.")
            tracker = EmissionsTracker(
                output_dir=output_dir, output_file=output_file, log_level="error"
            )
            tracker.start()
            logging.info(
                f"Tracker started. Output will be saved to {os.path.join(output_dir, output_file)}"
            )
        except Exception as e:
            logging.error(f"Error starting tracker: {str(e)}")
    else:
        logging.info("Tracker is already running.")
    sys.stdout.flush()


def stop_tracker():
    global tracker
    if tracker is not None:
        try:
            emissions = tracker.stop()

            if emissions is None:
                logging.warning("Tracker stopped, but emissions data is None.")
            else:
                logging.info(f"Tracker stopped. Emissions: {emissions} kg CO2")
            tracker = None
        except Exception as e:
            logging.error(f"Error stopping tracker: {str(e)}")
    else:
        logging.info("Tracker was not running.")
    sys.stdout.flush()


def command_listener():
    for line in sys.stdin:
        command = line.strip()
        if command.startswith("start"):
            _, output_file, readyWord = command.split(maxsplit=2)
            start_tracker(output_file)
            print(readyWord, flush=True)
        elif command == "stop":
            stop_tracker()
        elif command == "exit":
            logging.info("Exiting...")
            sys.stdout.flush()
            break

    # Ensure any remaining output is written
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = os.path.abspath(sys.argv[1]) # change output_dir value
    
    command_listener()
