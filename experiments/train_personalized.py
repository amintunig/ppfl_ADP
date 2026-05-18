# experiments/train_personalized.py

import subprocess
import time


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
CONFIG_PATH = "configs/personalized_fl.yaml"

NUM_CLIENTS = 2


# -----------------------------------------------------
# Start Server
# -----------------------------------------------------
print("\nStarting Personalized FL Server...\n")

server_process = subprocess.Popen(
    ["python", "server.py"]
)

time.sleep(5)


# -----------------------------------------------------
# Start Clients
# -----------------------------------------------------
client_processes = []

for cid in range(NUM_CLIENTS):

    print(f"Starting Personalized Client {cid}...")

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
print("\nPersonalized FL Training Completed.\n")