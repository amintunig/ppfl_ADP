# experiments/benchmark_noniid.py

import subprocess


configs = [

    "configs/noniid_alpha_1.yaml",
    "configs/noniid_alpha_05.yaml",
    "configs/noniid_alpha_01.yaml"
]


for config in configs:

    print(f"\nRunning Non-IID Benchmark: {config}\n")

    subprocess.run([
        "python",
        "experiments/run_experiment.py",
        "--config",
        config
    ])