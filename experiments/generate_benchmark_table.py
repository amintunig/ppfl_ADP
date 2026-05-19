# experiments/generate_benchmark_table.py

import os
import json
import pandas as pd


RESULTS_DIR = "results"

rows = []


for experiment in os.listdir(RESULTS_DIR):

    summary_path = os.path.join(

        RESULTS_DIR,
        experiment,
        "summary.json"
    )

    if not os.path.exists(summary_path):

        continue


    with open(summary_path, "r") as f:

        summary = json.load(f)


    rows.append({

        "Method": experiment,

        "Final Accuracy":
            summary.get(
                "final_accuracy",
                None
            ),

        "Best Accuracy":
            summary.get(
                "best_accuracy",
                None
            ),

        "Final Loss":
            summary.get(
                "final_loss",
                None
            ),

        "Best Loss":
            summary.get(
                "best_loss",
                None
            )
    })


df = pd.DataFrame(rows)

print("\nBenchmark Table:\n")

print(df)

df.to_csv(

    "results/benchmark_table.csv",
    index=False
)

print(
    "\nSaved benchmark table.\n"
)