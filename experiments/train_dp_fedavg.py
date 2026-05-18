# experiments/train_dp_fedavg.py

import subprocess
import time


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
CONFIG_PATH = "configs/dp_fedavg_noniid.yaml"

NUM_CLIENTS = 2


# -----------------------------------------------------
# Start Server
# -----------------------------------------------------
print("\nStarting DP-FedAvg Server...\n")

server_process = subprocess.Popen(
    ["python", "server.py"]
)

time.sleep(5)


# -----------------------------------------------------
# Start Clients
# -----------------------------------------------------
client_processes = []

for cid in range(NUM_CLIENTS):

    print(f"Starting DP Client {cid}...")

    process = subprocess.Popen([
        "python",
        "client.py",
        str(cid)
    ])

    client_processes.append(process)


# -----------------------------------------------------
# Wait
# -----------------------------------------------------
server_process.wait()

for process in client_processes:

    process.wait()


# -----------------------------------------------------
# Done
# -----------------------------------------------------
print("\nDP-FedAvg Training Completed.\n")