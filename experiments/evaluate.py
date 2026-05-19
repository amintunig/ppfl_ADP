# experiments/evaluate.py

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import json
import pandas as pd

from utils.config import load_config
from utils.artifact_manager import ArtifactManager


# =====================================================
# LOAD CONFIG
# =====================================================
CONFIG_PATH = "configs/fedavg_iid.yaml"

CONFIG = load_config(CONFIG_PATH)

experiment_name = CONFIG["experiment_name"]


# =====================================================
# INITIALIZE ARTIFACT MANAGER
# =====================================================
manager = ArtifactManager(
    experiment_name
)


# =====================================================
# LOG DIRECTORY
# =====================================================
LOG_DIR = "logs"


# =====================================================
# FIND LATEST CSV LOG
# =====================================================
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

print(f"\nLoading Metrics:\n{csv_path}\n")


# =====================================================
# LOAD METRICS
# =====================================================
df = pd.read_csv(csv_path)

print(df.head())


# =====================================================
# SAVE metrics.csv
# =====================================================
manager.save_csv(
    df,
    "metrics.csv"
)


# =====================================================
# COMPUTE STATISTICS
# =====================================================
final_round = int(
    df["round"].iloc[-1]
)

final_accuracy = float(
    df["accuracy"].iloc[-1]
)

best_accuracy = float(
    df["accuracy"].max()
)

average_accuracy = float(
    df["accuracy"].mean()
)

final_loss = float(
    df["loss"].iloc[-1]
)

best_loss = float(
    df["loss"].min()
)

average_loss = float(
    df["loss"].mean()
)


# =====================================================
# SUMMARY DICTIONARY
# =====================================================
summary = {

    "experiment_name": experiment_name,

    "final_round": final_round,

    "final_accuracy": round(
        final_accuracy,
        4
    ),

    "best_accuracy": round(
        best_accuracy,
        4
    ),

    "average_accuracy": round(
        average_accuracy,
        4
    ),

    "final_loss": round(
        final_loss,
        4
    ),

    "best_loss": round(
        best_loss,
        4
    ),

    "average_loss": round(
        average_loss,
        4
    ),

    "partition_type": CONFIG[
        "partition"
    ]["type"],

    "privacy_enabled": CONFIG[
        "privacy"
    ]["enabled"]
}


# =====================================================
# PRINT SUMMARY
# =====================================================
print("\n===================================")
print("EXPERIMENT EVALUATION SUMMARY")
print("===================================\n")

for key, value in summary.items():

    print(f"{key}: {value}")

print("\n===================================\n")


# =====================================================
# SAVE summary.json
# =====================================================
manager.save_json(
    summary,
    "summary.json"
)


# =====================================================
# SAVE config.yaml
# =====================================================
manager.save_config(CONFIG)


# =====================================================
# OPTIONAL: CREATE privacy_report.json
# =====================================================
if CONFIG["privacy"]["enabled"]:

    privacy_report = {

        "dp_enabled": True,

        "noise_multiplier": CONFIG[
            "privacy"
        ].get(
            "noise_multiplier",
            None
        ),

        "max_grad_norm": CONFIG[
            "privacy"
        ].get(
            "max_grad_norm",
            None
        ),

        "delta": CONFIG[
            "privacy"
        ].get(
            "delta",
            None
        )
    }

    manager.save_json(
        privacy_report,
        "privacy_report.json"
    )


# =====================================================
# FINISH
# =====================================================
print(
    f"\nEvaluation artifacts saved to:\n"
    f"results/{experiment_name}/\n"
)