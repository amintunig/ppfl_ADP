# experiments/ablation_dp.py

import subprocess


configs = [

    "configs/dp_noise_05.yaml",
    "configs/dp_noise_10.yaml",
    "configs/dp_noise_20.yaml"
]


for config in configs:

    print(f"\nRunning DP Ablation: {config}\n")

    subprocess.run([
        "python",
        "experiments/run_experiment.py",
        "--config",
        config
    ])