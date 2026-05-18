# experiments/evaluate.py

import os
import json
import pandas as pd


# -----------------------------------------------------
# Paths
# -----------------------------------------------------
LOG_DIR = "logs"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------
# Find Latest CSV Log
# -----------------------------------------------------
csv_files = [

    f for f in os.listdir(LOG_DIR)
    if f.endswith(".csv")
]

if len(csv_files) == 0:

    raise FileNotFoundError(
        "No CSV log files found in logs/"
    )

latest_csv = sorted(csv_files)[-1]

csv_path = os.path.join(
    LOG_DIR,
    latest_csv
)

print(f"\nLoading: {csv_path}\n")


# -----------------------------------------------------
# Load Metrics
# -----------------------------------------------------
df = pd.read_csv(csv_path)

print(df.head())


# -----------------------------------------------------
# Compute Final Statistics
# -----------------------------------------------------
final_round = int(df["round"].iloc[-1])

final_accuracy = float(
    df["accuracy"].iloc[-1]
)

best_accuracy = float(
    df["accuracy"].max()
)

avg_accuracy = float(
    df["accuracy"].mean()
)

final_loss = float(
    df["loss"].iloc[-1]
)

best_loss = float(
    df["loss"].min()
)


# -----------------------------------------------------
# Summary Dictionary
# -----------------------------------------------------
summary = {

    "final_round": final_round,

    "final_accuracy": round(
        final_accuracy, 4
    ),

    "best_accuracy": round(
        best_accuracy, 4
    ),

    "average_accuracy": round(
        avg_accuracy, 4
    ),

    "final_loss": round(
        final_loss, 4
    ),

    "best_loss": round(
        best_loss, 4
    )
}


# -----------------------------------------------------
# Print Summary
# -----------------------------------------------------
print("\n========== Evaluation Summary ==========\n")

for k, v in summary.items():

    print(f"{k}: {v}")

print("\n========================================\n")


# -----------------------------------------------------
# Save Summary
# -----------------------------------------------------
summary_path = os.path.join(
    RESULTS_DIR,
    "evaluation_summary.json"
)

with open(summary_path, "w") as f:

    json.dump(summary, f, indent=4)

print(
    f"Saved evaluation summary to:\n"
    f"{summary_path}\n"
)