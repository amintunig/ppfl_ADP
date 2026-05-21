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

## PPFL-ADP: Privacy-Preserving Federated Learning with Adaptive Differential Privacy
#### This project implements an experimental setup for Adaptive Differential Privacy (ADP) within a Privacy-Preserving Federated Learning (PPFL) framework.

### Overview

#### PPFL-ADP is a research-oriented federated learning framework designed for privacy-preserving and robust machine learning in healthcare environments. The framework focuses on:

    Federated Learning (FL)
    Privacy-Preserving FL (PPFL)
    Differential Privacy (DP)
    Adaptive Differential Privacy (ADP)
    Non-IID data handling
    Robust aggregation
    Benchmark experimentation
    Research reproducibility

    The project is aligned with PhD research objectives in:

    Privacy-preserving federated learning
    Adaptive differential privacy
    Non-IID medical data learning
    Robust and secure healthcare AI systems
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


### Current Implemented Features
### Federated Learning Infrastructure
        Flower-based FL orchestration
        PyTorch CNN implementation
        Multi-client simulation
        GPU support
        Modular experiment management
        Config-driven experimentation

### Aggregation Methods
### FedAvg

Standard Federated Averaging baseline.

###    FedProx

    Federated optimization with proximal regularization for handling Non-IID data.

    FedProx objective:
L(w)+2μ​∥w−wt​∥2

### where:
    L(w) is the local objective,
    wt is the global model,
    μ is the proximal regularization coefficient.

Privacy Infrastructure

### Implemented:

    Differential Privacy configuration system
    Privacy-aware experiment structure
    Adaptive DP configuration placeholders

### Planned:

    Opacus integration
    DP-SGD
    Dynamic noise scaling
    Adaptive clipping
    Personalized privacy budgets

### Dataset Support

#### Currently implemented:

        MNIST

#### Partitioning:

    IID partitioning
    Dirichlet Non-IID partitioning

Bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
📂 Project Structure

    ppfl_ADP/
    │
    ├── client.py
    ├── server.py
    │
    ├── configs/
    │   ├── fedavg_iid.yaml
    │   ├── fedavg_noniid.yaml
    │   ├── fedprox_iid.yaml
    │   ├── fedprox_noniid.yaml
    │   ├── dp_fedavg_iid.yaml
    │   ├── dp_fedavg_noniid.yaml
    │   └── adaptive_dp.yaml
    │
    ├── datasets/
    │   └── mnist.py
    │
    ├── models/
    │   └── cnn.py
    │
    ├── privacy/
    │   └── dp_engine.py
    │
    ├── strategies/
    │   └── fedavg_strategy.py
    │
    ├── utils/
    │   ├── logger.py
    │   ├── metrics.py
    │   ├── config.py
    │   ├── checkpoint.py
    │   ├── artifact_manager.py
    │   └── fedprox.py
    │
    ├── experiments/
    │   ├── train_fedavg.py
    │   ├── train_fedprox.py
    │   ├── train_dp_fedavg.py
    │   ├── train_adaptive_dp.py
    │   ├── evaluate.py
    │   ├── plot_results.py
    │   ├── generate_benchmark_table.py
    │   └── run_all_benchmarks.py
    │
    ├── logs/
    │
    └── results/


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

### Installation
    Create Virtual Environment
     python -m venv venv
    Activate Virtual Environment
     source venv/bin/activate
    Install Dependencies
     pip install -r requirements.txt
#### Required Libraries

#### Main dependencies:
        torch
        torchvision
        flower
        opacus
        matplotlib
        numpy
        pandas
        scikit-learn
### Bash
#### Running Experiments

    Run All Benchmarks
        python experiments/run_all_benchmarks.py
    Generate Benchmark Table
        python experiments/generate_benchmark_table.py
### This automatically executes:

        FedAvg IID
        FedAvg Non-IID
        FedProx IID
        FedProx Non-IID
        DP-FedAvg IID
        DP-FedAvg Non-IID    
### Benchmark Results

#### Generated benchmark logs:
        logs/fedavg_iid.log
        logs/fedavg_noniid.log
        logs/fedprox_iid.log
        logs/fedprox_noniid.log
        logs/dp_fedavg_iid.log
        logs/dp_fedavg_noniid.log
        logs/adaptive_dp.log 
## 📝 Features
#### Adaptive DP: Dynamically adjusts noise levels during training.

#### Federated Strategy: Custom  implementation for secure aggregation.FedAvg

#### Logging: Comprehensive tracking of metrics and privacy budget consumption.

#### To see all results tables and plots, run the following command:
    python experiments/generate_benchmark_table.py
    python experiments/plot_results.py
    cat results/benchmark_results.csv
    python experiments/train_dp_fedavg.py --config configs/dp_fedavg_iid.yaml
    python experiments/train_dp_fedavg.py --config configs/dp_fedavg_noniid.yaml    
    



## 🤝 Contributing
#### Feel free to open issues or submit pull requests to improve the adaptive mechanisms or add support for new datasets.