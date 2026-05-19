# scripts/create_result_structure.py

import os
import json
import yaml
import pandas as pd


# =====================================================
# Experiment Definitions
# =====================================================
experiments = {

    "fedavg_iid": {

        "files": [
            "metrics.csv",
            "summary.json",
            "config.yaml",
            "accuracy_plot.png",
            "loss_plot.png",
            "combined_plot.png",
            "checkpoint.pth"
        ]
    },

    "fedavg_noniid": {

        "files": [
            "metrics.csv",
            "summary.json",
            "config.yaml",
            "accuracy_plot.png",
            "loss_plot.png",
            "combined_plot.png",
            "checkpoint.pth"
        ]
    },

    "dp_fedavg_iid": {

        "files": [
            "metrics.csv",
            "epsilon.csv",
            "summary.json",
            "config.yaml",
            "privacy_report.json",
            "accuracy_plot.png",
            "loss_plot.png",
            "combined_plot.png",
            "checkpoint.pth"
        ]
    },

    "dp_fedavg_noniid": {

        "files": [
            "metrics.csv",
            "epsilon.csv",
            "summary.json",
            "config.yaml",
            "privacy_report.json",
            "accuracy_plot.png",
            "loss_plot.png",
            "combined_plot.png",
            "checkpoint.pth"
        ]
    },

    "adaptive_dp": {

        "files": [
            "metrics.csv",
            "adaptive_epsilon.csv",
            "clipping_history.csv",
            "privacy_report.json",
            "summary.json"
        ],

        "subdirs": [
            "plots"
        ]
    }
}


# =====================================================
# Create Structure
# =====================================================
BASE_DIR = "results"

os.makedirs(BASE_DIR, exist_ok=True)


# =====================================================
# Create Experiments
# =====================================================
for exp_name, exp_info in experiments.items():

    exp_dir = os.path.join(
        BASE_DIR,
        exp_name
    )

    os.makedirs(exp_dir, exist_ok=True)

    print(f"\nCreated Experiment Folder: {exp_dir}")


    # -------------------------------------------------
    # Create Files
    # -------------------------------------------------
    for filename in exp_info["files"]:

        file_path = os.path.join(
            exp_dir,
            filename
        )

        extension = os.path.splitext(filename)[1]


        # =============================================
        # CSV
        # =============================================
        if extension == ".csv":

            df = pd.DataFrame()

            df.to_csv(
                file_path,
                index=False
            )


        # =============================================
        # JSON
        # =============================================
        elif extension == ".json":

            with open(file_path, "w") as f:

                json.dump({}, f, indent=4)


        # =============================================
        # YAML
        # =============================================
        elif extension in [".yaml", ".yml"]:

            with open(file_path, "w") as f:

                yaml.dump({}, f)


        # =============================================
        # Binary/Other
        # =============================================
        else:

            open(file_path, "a").close()


        print(f"  ├── {filename}")


    # -------------------------------------------------
    # Create Subdirectories
    # -------------------------------------------------
    if "subdirs" in exp_info:

        for subdir in exp_info["subdirs"]:

            subdir_path = os.path.join(
                exp_dir,
                subdir
            )

            os.makedirs(
                subdir_path,
                exist_ok=True
            )

            print(f"  └── {subdir}/")


print("\nResult structure successfully created.\n")