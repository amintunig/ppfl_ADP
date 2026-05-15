# utils/logger.py

import os
import json
import csv
from datetime import datetime


class ExperimentLogger:

    def __init__(self, log_dir="logs"):

        self.log_dir = log_dir # Log directory

        os.makedirs(self.log_dir, exist_ok=True) # Create log directory if it doesn't exist

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Unique timestamp for log files

        self.csv_file = os.path.join(
            self.log_dir,
            f"metrics_{timestamp}.csv"
        )

        self.json_file = os.path.join(
            self.log_dir,
            f"summary_{timestamp}.json"
        )

        self.metrics = []

        # Create CSV header
        with open(self.csv_file, mode="w", newline="") as file: # Open CSV file for writing

            writer = csv.writer(file) # Create CSV writer

            writer.writerow([
                "round",
                "loss",
                "accuracy"
            ])

    def log_round(self, round_num, loss, accuracy): # Log metrics for a training round

        self.metrics.append({
            "round": round_num,
            "loss": loss,
            "accuracy": accuracy
        })

        with open(self.csv_file, mode="a", newline="") as file: # Open CSV file for appending

            writer = csv.writer(file)

            writer.writerow([
                round_num,
                loss,
                accuracy
            ])

    def save_summary(self):

        with open(self.json_file, "w") as file:

            json.dump(self.metrics, file, indent=4)

        print(f"\nLogs saved to: {self.log_dir}")