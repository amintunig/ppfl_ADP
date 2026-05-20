# =====================================================
# experiments/generate_benchmark_table.py
# =====================================================

import os
import pandas as pd


# =====================================================
# LOG DIRECTORY
# =====================================================
LOG_DIR = "logs"

rows = []


# =====================================================
# PROCESS ALL METRIC FILES
# =====================================================
for file_name in os.listdir(LOG_DIR):

    if not file_name.endswith("_metrics.csv"):

        continue

    csv_path = os.path.join(
        LOG_DIR,
        file_name
    )

    # =============================================
    # EXPERIMENT NAME
    # =============================================
    experiment_name = file_name.replace(
        "_metrics.csv",
        ""
    )

    # =============================================
    # LOAD CSV
    # =============================================
    df = pd.read_csv(csv_path)

    if len(df) == 0:

        continue

    # =============================================
    # FINAL VALUES
    # =============================================
    final_accuracy = df[
        "accuracy"
    ].iloc[-1]

    best_accuracy = df[
        "accuracy"
    ].max()

    final_loss = df[
        "loss"
    ].iloc[-1]

    best_loss = df[
        "loss"
    ].min()

    # =============================================
    # APPEND
    # =============================================
    rows.append({

        "Method":
            experiment_name,

        "Final Accuracy":
            round(
                final_accuracy,
                4
            ),

        "Best Accuracy":
            round(
                best_accuracy,
                4
            ),

        "Final Loss":
            round(
                final_loss,
                4
            ),

        "Best Loss":
            round(
                best_loss,
                4
            )
    })


# =====================================================
# CREATE DATAFRAME
# =====================================================
benchmark_df = pd.DataFrame(rows)

benchmark_df = benchmark_df.sort_values(
    by="Method"
)


# =====================================================
# PRINT
# =====================================================
print("\nBenchmark Table:\n")

print(benchmark_df)


# =====================================================
# SAVE CSV
# =====================================================
output_path = os.path.join(

    "results",

    "benchmark_table.csv"
)

benchmark_df.to_csv(

    output_path,

    index=False
)

print(
    f"\nSaved benchmark table to:\n"
    f"{output_path}\n"
)