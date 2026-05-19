# experiments/run_all_benchmarks.py

import subprocess


configs = [

    "configs/fedavg_iid.yaml",
    "configs/fedavg_noniid.yaml",

    "configs/fedprox_iid.yaml",
    "configs/fedprox_noniid.yaml",

    "configs/dp_fedavg_iid.yaml",
    "configs/dp_fedavg_noniid.yaml"
]


for config in configs:

    print("\n===================================")
    print(f"RUNNING: {config}")
    print("===================================\n")


    # ---------------------------------------------
    # TRAIN
    # ---------------------------------------------
    subprocess.run([

        "python",
        "experiments/train_fedavg.py"
    ])


    # ---------------------------------------------
    # EVALUATE
    # ---------------------------------------------
    subprocess.run([

        "python",
        "experiments/evaluate.py",
        "--config",
        config
    ])


    # ---------------------------------------------
    # PLOT
    # ---------------------------------------------
    subprocess.run([

        "python",
        "experiments/plot_results.py",
        "--config",
        config
    ])