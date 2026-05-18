# utils/result_manager.py

import os
import shutil
from datetime import datetime


class ResultManager:

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
    # Get Experiment Directory
    # -------------------------------------------------
    def get_result_dir(self):

        return self.base_dir

    # -------------------------------------------------
    # Save File
    # -------------------------------------------------
    def save_file(self, source_path):

        if not os.path.exists(source_path):

            return

        filename = os.path.basename(source_path)

        destination = os.path.join(
            self.base_dir,
            filename
        )

        shutil.copy(
            source_path,
            destination
        )

    # -------------------------------------------------
    # Save Multiple Files
    # -------------------------------------------------
    def save_files(self, file_list):

        for file_path in file_list:

            self.save_file(file_path)