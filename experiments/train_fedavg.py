# experiments/train_fedavg.py

import subprocess
import time


# -----------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------
CONFIG_PATH = "configs/fedavg_noniid.yaml"

NUM_CLIENTS = 2


# -----------------------------------------------------
# Start Server
# -----------------------------------------------------
print("\nStarting FedAvg Server...\n")

server_process = subprocess.Popen(
    ["python", "server.py"]
)

time.sleep(5)


# -----------------------------------------------------
# Start Clients
# -----------------------------------------------------
client_processes = []

for cid in range(NUM_CLIENTS):

    print(f"Starting Client {cid}...")

    process = subprocess.Popen([
        "python",
        "client.py",
        str(cid)
    ])

    client_processes.append(process)


# -----------------------------------------------------
# Wait for Completion
# -----------------------------------------------------
server_process.wait()

for process in client_processes:

    process.wait()


# -----------------------------------------------------
# Finish
# -----------------------------------------------------
print("\nFedAvg Training Completed.\n")