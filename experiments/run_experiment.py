# experiments/run_experiment.py

import argparse
import subprocess


def run_server():

    return subprocess.Popen(
        ["python", "server.py"]
    )


def run_client(client_id):

    return subprocess.Popen(
        ["python", "client.py", str(client_id)]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    print(f"\nRunning Experiment: {args.config}\n")

    # Start server
    server_process = run_server()

    # Start clients
    clients = []

    for cid in range(2):

        clients.append(
            run_client(cid)
        )

    # Wait for completion
    server_process.wait()

    for c in clients:
        c.wait()

    print("\nExperiment Completed.\n")