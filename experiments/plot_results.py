# # experiments/plot_results.py

# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# import argparse

# from utils.config import load_config
# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--config",
#     type=str,
#     required=True
# )

# args = parser.parse_args()
# CONFIG = load_config(
#     args.config
# )
# # -----------------------------------------------------
# # Directories
# # -----------------------------------------------------
# LOG_DIR = "logs"
# RESULTS_DIR = "results"

# os.makedirs(RESULTS_DIR, exist_ok=True)


# # -----------------------------------------------------
# # Find Latest CSV
# # -----------------------------------------------------
# csv_files = [

#     f for f in os.listdir(LOG_DIR)
#     if f.endswith(".csv")
# ]

# if len(csv_files) == 0:

#     raise FileNotFoundError(
#         "No CSV log files found."
#     )

# latest_csv = sorted(csv_files)[-1]

# csv_path = os.path.join(
#     LOG_DIR,
#     latest_csv
# )

# print(f"\nLoading: {csv_path}\n")


# # -----------------------------------------------------
# # Load Metrics
# # -----------------------------------------------------
# df = pd.read_csv(csv_path)

# print(df.head())


# # =====================================================
# # 1. ACCURACY PLOT
# # =====================================================
# plt.figure(figsize=(8, 5))

# plt.plot(
#     df["round"],
#     df["accuracy"],
#     marker="o",
#     linewidth=2,
#     label="Accuracy"
# )

# plt.xlabel("Communication Round")
# plt.ylabel("Accuracy")

# plt.title("Federated Learning Accuracy")

# plt.grid(True)

# accuracy_plot_path = os.path.join(
#     RESULTS_DIR,
#     "accuracy_plot.png"
# )

# plt.savefig(accuracy_plot_path)

# print(
#     f"\nAccuracy plot saved to:\n"
#     f"{accuracy_plot_path}\n"
# )

# plt.close()


# # =====================================================
# # 2. LOSS PLOT
# # =====================================================
# plt.figure(figsize=(8, 5))

# plt.plot(
#     df["round"],
#     df["loss"],
#     marker="o",
#     linewidth=2,
#     label="Loss"
# )

# plt.xlabel("Communication Round")
# plt.ylabel("Loss")

# plt.title("Federated Learning Loss")

# plt.grid(True)

# loss_plot_path = os.path.join(
#     RESULTS_DIR,
#     "loss_plot.png"
# )

# plt.savefig(loss_plot_path)

# print(
#     f"\nLoss plot saved to:\n"
#     f"{loss_plot_path}\n"
# )

# plt.close()


# # =====================================================
# # 3. Combined Plot
# # =====================================================
# fig, ax1 = plt.subplots(figsize=(9, 5))

# ax1.plot(
#     df["round"],
#     df["accuracy"],
#     marker="o",
#     linewidth=2,
#     label="Accuracy"
# )

# ax1.set_xlabel("Communication Round")
# ax1.set_ylabel("Accuracy")

# ax2 = ax1.twinx()

# ax2.plot(
#     df["round"],
#     df["loss"],
#     marker="s",
#     linewidth=2,
#     linestyle="--",
#     label="Loss"
# )

# ax2.set_ylabel("Loss")

# plt.title("FL Accuracy and Loss Convergence")

# combined_plot_path = os.path.join(
#     RESULTS_DIR,
#     "combined_plot.png"
# )

# plt.savefig(combined_plot_path)

# print(
#     f"\nCombined plot saved to:\n"
#     f"{combined_plot_path}\n"
# )

# plt.close()

# experiments/plot_results.py

import sys
import os
import argparse

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.artifact_manager import ArtifactManager


# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True
)

args = parser.parse_args()


# =====================================================
# LOAD CONFIG
# =====================================================
CONFIG = load_config(
    args.config
)

experiment_name = CONFIG[
    "experiment_name"
]


# =====================================================
# ARTIFACT MANAGER
# =====================================================
manager = ArtifactManager(
    experiment_name
)

RESULTS_DIR = manager.base_dir


# =====================================================
# FIND LATEST CSV
# =====================================================
LOG_DIR = "logs"

csv_files = [

    f for f in os.listdir(LOG_DIR)
    if f.endswith(".csv")
]

if len(csv_files) == 0:

    raise FileNotFoundError(
        "No CSV logs found."
    )

latest_csv = sorted(csv_files)[-1]

csv_path = os.path.join(
    LOG_DIR,
    latest_csv
)

print(f"\nLoading:\n{csv_path}\n")


# =====================================================
# LOAD METRICS
# =====================================================
df = pd.read_csv(csv_path)

print(df.head())


# =====================================================
# ACCURACY PLOT
# =====================================================
plt.figure(figsize=(8, 5))

plt.plot(

    df["round"],

    df["accuracy"],

    marker="o",

    linewidth=2
)

plt.xlabel("Communication Round")

plt.ylabel("Accuracy")

plt.title("FL Accuracy Convergence")

plt.grid(True)

accuracy_path = os.path.join(

    RESULTS_DIR,

    "accuracy_plot.png"
)

plt.savefig(accuracy_path)

print(f"Saved: {accuracy_path}")

plt.close()


# =====================================================
# LOSS PLOT
# =====================================================
plt.figure(figsize=(8, 5))

plt.plot(

    df["round"],

    df["loss"],

    marker="o",

    linewidth=2
)

plt.xlabel("Communication Round")

plt.ylabel("Loss")

plt.title("FL Loss Convergence")

plt.grid(True)

loss_path = os.path.join(

    RESULTS_DIR,

    "loss_plot.png"
)

plt.savefig(loss_path)

print(f"Saved: {loss_path}")

plt.close()


# =====================================================
# COMBINED PLOT
# =====================================================
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(

    df["round"],

    df["accuracy"],

    marker="o",

    linewidth=2
)

ax1.set_xlabel(
    "Communication Round"
)

ax1.set_ylabel(
    "Accuracy"
)

ax2 = ax1.twinx()

ax2.plot(

    df["round"],

    df["loss"],

    marker="s",

    linewidth=2,

    linestyle="--"
)

ax2.set_ylabel(
    "Loss"
)

plt.title(
    "FL Accuracy and Loss"
)

combined_path = os.path.join(

    RESULTS_DIR,

    "combined_plot.png"
)

plt.savefig(combined_path)

print(f"Saved: {combined_path}")

plt.close()


# =====================================================
# FINISH
# =====================================================
print(
    f"\nPlots saved to:\n"
    f"{RESULTS_DIR}\n"
)