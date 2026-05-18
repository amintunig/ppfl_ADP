# experiments/compare_methods.py

import pandas as pd


methods = {

    "FedAvg": "results/fedavg/metrics.csv",

    "DP-FedAvg": "results/dp_fedavg/metrics.csv"
}


summary = []

for method, path in methods.items():

    df = pd.read_csv(path)

    final_accuracy = df["accuracy"].iloc[-1]

    summary.append({
        "method": method,
        "final_accuracy": final_accuracy
    })


summary_df = pd.DataFrame(summary)

print("\nMethod Comparison:\n")

print(summary_df)