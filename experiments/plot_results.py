# experiments/plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------
# Directories
# -----------------------------------------------------
LOG_DIR = "logs"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------
# Find Latest CSV
# -----------------------------------------------------
csv_files = [

    f for f in os.listdir(LOG_DIR)
    if f.endswith(".csv")
]

if len(csv_files) == 0:

    raise FileNotFoundError(
        "No CSV log files found."
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


# =====================================================
# 1. ACCURACY PLOT
# =====================================================
plt.figure(figsize=(8, 5))

plt.plot(
    df["round"],
    df["accuracy"],
    marker="o",
    linewidth=2,
    label="Accuracy"
)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")

plt.title("Federated Learning Accuracy")

plt.grid(True)

accuracy_plot_path = os.path.join(
    RESULTS_DIR,
    "accuracy_plot.png"
)

plt.savefig(accuracy_plot_path)

print(
    f"\nAccuracy plot saved to:\n"
    f"{accuracy_plot_path}\n"
)

plt.close()


# =====================================================
# 2. LOSS PLOT
# =====================================================
plt.figure(figsize=(8, 5))

plt.plot(
    df["round"],
    df["loss"],
    marker="o",
    linewidth=2,
    label="Loss"
)

plt.xlabel("Communication Round")
plt.ylabel("Loss")

plt.title("Federated Learning Loss")

plt.grid(True)

loss_plot_path = os.path.join(
    RESULTS_DIR,
    "loss_plot.png"
)

plt.savefig(loss_plot_path)

print(
    f"\nLoss plot saved to:\n"
    f"{loss_plot_path}\n"
)

plt.close()


# =====================================================
# 3. Combined Plot
# =====================================================
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(
    df["round"],
    df["accuracy"],
    marker="o",
    linewidth=2,
    label="Accuracy"
)

ax1.set_xlabel("Communication Round")
ax1.set_ylabel("Accuracy")

ax2 = ax1.twinx()

ax2.plot(
    df["round"],
    df["loss"],
    marker="s",
    linewidth=2,
    linestyle="--",
    label="Loss"
)

ax2.set_ylabel("Loss")

plt.title("FL Accuracy and Loss Convergence")

combined_plot_path = os.path.join(
    RESULTS_DIR,
    "combined_plot.png"
)

plt.savefig(combined_plot_path)

print(
    f"\nCombined plot saved to:\n"
    f"{combined_plot_path}\n"
)

plt.close()