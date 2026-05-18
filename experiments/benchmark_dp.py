# experiments/benchmark_dp.py

import subprocess


configs = [

    "configs/fedavg_iid.yaml",
    "configs/fedavg_noniid.yaml",
    "configs/dp_fedavg_iid.yaml",
    "configs/dp_fedavg_noniid.yaml"
]


for config in configs:

    print(f"\nRunning: {config}\n")

    subprocess.run([
        "python",
        "experiments/run_experiment.py",
        "--config",
        config
    ])