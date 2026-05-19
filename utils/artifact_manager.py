# utils/artifact_manager.py

import os
import json
import yaml
import shutil
import pandas as pd


class ArtifactManager:

    def __init__(self, experiment_name):

        self.experiment_name = experiment_name

        self.base_dir = os.path.join(
            "results",
            experiment_name
        )

        os.makedirs(
            self.base_dir,
            exist_ok=True
        )

    # -------------------------------------------------
    # Get File Path
    # -------------------------------------------------
    def get_path(self, filename):

        return os.path.join(
            self.base_dir,
            filename
        )

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------
    def save_csv(self, dataframe, filename):

        path = self.get_path(filename)

        dataframe.to_csv(
            path,
            index=False
        )

        print(f"Saved CSV: {path}")

    # -------------------------------------------------
    # Save JSON
    # -------------------------------------------------
    def save_json(self, data, filename):

        path = self.get_path(filename)

        with open(path, "w") as f:

            json.dump(
                data,
                f,
                indent=4
            )

        print(f"Saved JSON: {path}")

    # -------------------------------------------------
    # Save YAML Config
    # -------------------------------------------------
    def save_config(self, config):

        path = self.get_path(
            "config.yaml"
        )

        with open(path, "w") as f:

            yaml.dump(config, f)

        print(f"Saved Config: {path}")

    # -------------------------------------------------
    # Copy Artifact
    # -------------------------------------------------
    def copy_artifact(self, source_path):

        if not os.path.exists(source_path):

            return

        filename = os.path.basename(
            source_path
        )

        destination = self.get_path(
            filename
        )

        shutil.copy(
            source_path,
            destination
        )

        print(f"Copied: {destination}")