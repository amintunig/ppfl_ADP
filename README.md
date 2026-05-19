<!-- ## Creating the first experiemntal setup for adaptive differential privacy
**This project structure in different directories
1. datasets/mnint.py
2. logs/resulst
3. strategies/fedavg_strategy.py
4. utils/metrics.py and logger.py
5. client.py
6. server.py

   ## To run this project
   create venv using: python3 -m venv venv and activate by source venv/bin/activate
   pip install -r requirements.txt
   open there terminal
   1. terminal 1: python server.py
   2. terminal 2: python client.py 0
   3. terminal 3: python client.py 1

  ** please use the UV for managing the project package easily -->

## PPFL-ADP: Adaptive Differential Privacy in Federated Learning
#### This project implements an experimental setup for Adaptive Differential Privacy (ADP) within a Privacy-Preserving Federated Learning (PPFL) framework.

🚀 Getting Started
This project uses  for extremely fast Python package and project management.uv

Prerequisites
Ensure you have  installed:uv

Bash
curl -LsSf https://astral-sh/uv/install.sh | sh
Installation & Setup
Clone the repository:

Bash
git clone https://github.com/amintunig/ppfl_ADP.git
cd ppfl_ADP
Create and sync the environment:
Using , you can create the virtual environment and install all dependencies in one command:uv

Bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
📂 Project Structure

## ppfl_ADP/
## ├── datasets/
#### │   └── mnist.py             # Data loading and preprocessing
#### ├── logs/
#### │   └── results/             # Experiment output and logs
#### ├── strategies/
#### │   └── fedavg_strategy.py   # Federated Averaging implementation
#### ├── utils/
#### │   ├── metrics.py           # Evaluation functions
#### │   └── logger.py            # Logging utilities
#### ├── client.py                # Federated Learning client logic
#### ├── server.py                # Federated Learning server/aggregator
#### └── requirements.txt         # Project dependencies
### 🛠 Running the Experiment
#### To simulate the federated environment, you need to open multiple terminal windows (or use a multiplexer like ). Make sure your virtual environment is active in all terminals.tmux

## Terminal 1: Start the Server

### Bash
#### python server.py
#### Terminal 2: Start Client 0

### Bash
#### python client.py 0
#### Terminal 3: Start Client 1
## To run the experiment, execute the following commands in separate terminals:
    python experiments/run_experiment.py
    python experiments/evaluate.py
    python experiments/plot_results.py
    python experiments/run_all_benchmarks.py
    python experiments/generate_benchmark_table.py
### Bash
#### python client.py 1
## 📝 Features
#### Adaptive DP: Dynamically adjusts noise levels during training.

#### Federated Strategy: Custom  implementation for secure aggregation.FedAvg

#### Logging: Comprehensive tracking of metrics and privacy budget consumption.

## 🤝 Contributing
#### Feel free to open issues or submit pull requests to improve the adaptive mechanisms or add support for new datasets.