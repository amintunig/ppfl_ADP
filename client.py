# # client.py

# import torch
# import flwr as fl

# from models.cnn import SimpleCNN
# from datasets.mnist import load_mnist


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class FlowerClient(fl.client.NumPyClient):

#     def __init__(self, client_id, num_clients):

#         self.client_id = client_id

#         self.model = SimpleCNN().to(DEVICE)

#         self.train_loader, self.test_loader = load_mnist(
#             client_id=client_id,
#             num_clients=num_clients
#         )

#     # -------------------------------------------------
#     # Get model parameters
#     # -------------------------------------------------
#     def get_parameters(self, config):

#         return [
#             val.cpu().numpy()
#             for _, val in self.model.state_dict().items()
#         ]

#     # -------------------------------------------------
#     # Set model parameters
#     # -------------------------------------------------
#     def set_parameters(self, parameters):

#         params_dict = zip(
#             self.model.state_dict().keys(),
#             parameters
#         )

#         state_dict = {
#             k: torch.tensor(v)
#             for k, v in params_dict
#         }

#         self.model.load_state_dict(state_dict, strict=True)

#     # -------------------------------------------------
#     # Local training
#     # -------------------------------------------------
#     def train_model(self):

#         criterion = torch.nn.CrossEntropyLoss()

#         optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=0.001
#         )

#         self.model.train()

#         for epoch in range(1):

#             for images, labels in self.train_loader:

#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 optimizer.zero_grad()

#                 outputs = self.model(images)

#                 loss = criterion(outputs, labels)

#                 loss.backward()

#                 optimizer.step()

#     # -------------------------------------------------
#     # Internal evaluation function
#     # -------------------------------------------------
#     def evaluate_model(self):

#         criterion = torch.nn.CrossEntropyLoss()

#         self.model.eval()

#         correct = 0
#         total = 0
#         loss_total = 0.0

#         with torch.no_grad():

#             for images, labels in self.test_loader:

#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 outputs = self.model(images)

#                 loss = criterion(outputs, labels)

#                 loss_total += loss.item()

#                 _, predicted = torch.max(outputs, 1)

#                 total += labels.size(0)

#                 correct += (predicted == labels).sum().item()

#         accuracy = correct / total

#         avg_loss = loss_total / len(self.test_loader)

#         return avg_loss, accuracy

#     # -------------------------------------------------
#     # Flower fit()
#     # -------------------------------------------------
#     def fit(self, parameters, config):

#         self.set_parameters(parameters)

#         self.train_model()

#         return (
#             self.get_parameters(config),
#             len(self.train_loader.dataset),
#             {}
#         )

#     # -------------------------------------------------
#     # Flower evaluate()
#     # -------------------------------------------------
#     def evaluate(self, parameters, config):

#         self.set_parameters(parameters)

#         loss, accuracy = self.evaluate_model()

#         return (
#             float(loss),
#             len(self.test_loader.dataset),
#             {"accuracy": float(accuracy)}
#         )


# # -----------------------------------------------------
# # Client function
# # -----------------------------------------------------
# def client_fn(cid: str):

#     return FlowerClient(
#         client_id=int(cid),
#         num_clients=2
#     )


# # -----------------------------------------------------
# # Start client
# # -----------------------------------------------------
# if __name__ == "__main__":

#     fl.client.start_numpy_client(
#         server_address="127.0.0.1:8080",
#         client=client_fn("0")
#     )

# client.py

# import os
# from numpy.__config__ import CONFIG
# import sys
# import torch
# import flwr as fl
# from experiments.evaluate import CONFIG

# from models.cnn import SimpleCNN
# from datasets.mnist import load_mnist
# from utils.metrics import compute_metrics

# from utils.checkpoint import save_checkpoint
# from utils.checkpoint import save_checkpoint    
# checkpoint_path = os.path.join(

#     "results",
#     CONFIG["experiment_name"],
#     "checkpoint.pth"
# )

# save_checkpoint(
#     self.model,
#     checkpoint_path
# )
# # -----------------------------------------------------
# # Device Configuration
# # -----------------------------------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # -----------------------------------------------------
# # Flower Client
# # -----------------------------------------------------
# class FlowerClient(fl.client.NumPyClient):

#     def __init__(self, client_id, num_clients):

#         self.client_id = client_id

#         # Load model
#         self.model = SimpleCNN().to(DEVICE)

#         # Load dataset
#         self.train_loader, self.test_loader = load_mnist(
#             client_id=client_id,
#             num_clients=num_clients,
#             batch_size=32,

#             # Choose:
#             # "iid"
#             # "label_skew"
#             # "dirichlet"
#             partition_type="dirichlet",

#             alpha=0.5
#         )

#     # -------------------------------------------------
#     # Get Parameters
#     # -------------------------------------------------
#     def get_parameters(self, config):

#         return [
#             val.cpu().numpy()
#             for _, val in self.model.state_dict().items()
#         ]

#     # -------------------------------------------------
#     # Set Parameters
#     # -------------------------------------------------
#     def set_parameters(self, parameters):

#         params_dict = zip(
#             self.model.state_dict().keys(),
#             parameters
#         )

#         state_dict = {
#             k: torch.tensor(v)
#             for k, v in params_dict
#         }

#         self.model.load_state_dict(state_dict, strict=True)

#     # -------------------------------------------------
#     # Local Training
#     # -------------------------------------------------
#     def train_model(self):

#         criterion = torch.nn.CrossEntropyLoss()

#         optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=0.001
#         )

#         self.model.train()

#         for epoch in range(1):

#             running_loss = 0.0

#             for images, labels in self.train_loader:

#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 optimizer.zero_grad()

#                 outputs = self.model(images)

#                 loss = criterion(outputs, labels)

#                 loss.backward()

#                 optimizer.step()

#                 running_loss += loss.item()

#         avg_loss = running_loss / len(self.train_loader)

#         print(
#             f"[Client {self.client_id}] "
#             f"Training Loss: {avg_loss:.4f}"
#         )

#     # -------------------------------------------------
#     # Evaluation
#     # -------------------------------------------------
#     def evaluate_model(self):

#         criterion = torch.nn.CrossEntropyLoss()

#         self.model.eval()

#         total_loss = 0.0

#         y_true = []
#         y_pred = []

#         with torch.no_grad():

#             for images, labels in self.test_loader:

#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 outputs = self.model(images)

#                 loss = criterion(outputs, labels)

#                 total_loss += loss.item()

#                 _, predicted = torch.max(outputs, 1)

#                 y_true.extend(labels.cpu().numpy())
#                 y_pred.extend(predicted.cpu().numpy())

#         avg_loss = total_loss / len(self.test_loader)

#         metrics = compute_metrics(y_true, y_pred)

#         return avg_loss, metrics

#     # -------------------------------------------------
#     # Flower Fit
#     # -------------------------------------------------
#     def fit(self, parameters, config):

#         self.set_parameters(parameters)

#         self.train_model()

#         return (
#             self.get_parameters(config),
#             len(self.train_loader.dataset),
#             {}
#         )

#     # -------------------------------------------------
#     # Flower Evaluate
#     # -------------------------------------------------
#     def evaluate(self, parameters, config):

#         self.set_parameters(parameters)

#         loss, metrics = self.evaluate_model()

#         print(
#             f"[Client {self.client_id}] "
#             f"Accuracy: {metrics['accuracy']:.4f} | "
#             f"F1: {metrics['f1_score']:.4f}"
#         )

#         return (
#             float(loss),
#             len(self.test_loader.dataset),
#             {
#                 "accuracy": float(metrics["accuracy"]),
#                 "precision": float(metrics["precision"]),
#                 "recall": float(metrics["recall"]),
#                 "f1_score": float(metrics["f1_score"])
#             }
#         )


# # -----------------------------------------------------
# # Client Function
# # -----------------------------------------------------
# def client_fn(cid: str):

#     return FlowerClient(
#         client_id=int(cid),
#         num_clients=2
#     )


# # -----------------------------------------------------
# # Start Client
# # -----------------------------------------------------
# if __name__ == "__main__":

#     # Read client ID from terminal
#     cid = sys.argv[1]

#     print(f"\nStarting Client {cid}...\n")

#     fl.client.start_numpy_client(
#         server_address="127.0.0.1:8080",
#         client=client_fn(cid)
#     )

# client.py
# =====================================================
# client.py
# =====================================================
# =====================================================
# client.py
# =====================================================

# import os
# import sys
# import argparse

# from copy import deepcopy

# import torch
# import flwr as fl

# from torch import nn

# sys.path.append(
#     os.path.abspath(
#         os.path.dirname(__file__)
#     )
# )

# # =====================================================
# # ARGUMENTS
# # =====================================================
# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--config",
#     type=str,
#     required=True
# )

# parser.add_argument(
#     "--cid",
#     type=str,
#     required=True
# )

# args = parser.parse_args()


# # =====================================================
# # PROJECT IMPORTS
# # =====================================================
# from models.cnn import SimpleCNN

# from datasets.mnist import load_mnist

# from utils.config import load_config

# from utils.checkpoint import save_checkpoint

# from strategies.fedprox import fedprox_regularizer

# from privacy.dp_engine import DPEngine
# import numpy as np
# import pandas as pd
# # =====================================================
# # LOAD CONFIG
# # =====================================================
# CONFIG = load_config(
#     args.config
# )


# # =====================================================
# # DEVICE
# # =====================================================
# DEVICE = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )

# print(f"\nUsing Device: {DEVICE}\n")


# # =====================================================
# # TRAIN FUNCTION
# # =====================================================
# def train_model(
#     model,
#     trainloader,
#     epochs,
#     learning_rate
# ):

#     criterion = nn.CrossEntropyLoss()
    
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=learning_rate
#     )
    

#     global_model = deepcopy(model)

#     model.train()

#     for epoch in range(epochs):

#         running_loss = 0.0

#         for images, labels in trainloader:

#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             optimizer.zero_grad()

#             outputs = model(images)

#             loss = criterion(
#                 outputs,
#                 labels
#             )

#             # =========================================
#             # FEDPROX
#             # =========================================
#             if CONFIG[
#                 "aggregation"
#             ]["strategy"] == "fedprox":

#                 mu = CONFIG[
#                     "fedprox"
#                 ]["mu"]

#                 prox_loss = fedprox_regularizer(
#                     model,
#                     global_model,
#                     mu
#                 )

#                 loss += prox_loss

#             loss.backward()

#             optimizer.step()

#             running_loss += loss.item()

#         avg_loss = (
#             running_loss /
#             len(trainloader)
#         )

#         print(
#             f"Epoch [{epoch+1}/{epochs}] "
#             f"Loss: {avg_loss:.4f}"
#         )


# # =====================================================
# # TEST FUNCTION
# # =====================================================
# def test_model(
#     model,
#     testloader
# ):

#     criterion = nn.CrossEntropyLoss()

#     model.eval()

#     correct = 0
#     total = 0
#     loss_total = 0.0

#     with torch.no_grad():

#         for images, labels in testloader:

#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             outputs = model(images)

#             loss = criterion(
#                 outputs,
#                 labels
#             )

#             loss_total += loss.item()

#             _, predicted = torch.max(
#                 outputs.data,
#                 1
#             )

#             total += labels.size(0)

#             correct += (
#                 predicted == labels
#             ).sum().item()

#     accuracy = correct / total

#     avg_loss = (
#         loss_total /
#         len(testloader)
#     )

#     return avg_loss, accuracy


# # =====================================================
# # FLOWER CLIENT
# # =====================================================
# class FlowerClient(fl.client.NumPyClient):

#     def __init__(self, cid):

#         self.cid = cid

#         self.model = SimpleCNN().to(DEVICE)

#         self.trainloader, self.testloader = (

#             load_mnist(

#                 client_id=int(cid),

#                 num_clients=CONFIG[
#                     "clients"
#                 ]["num_clients"],

#                 partition_type=CONFIG[
#                     "partition"
#                 ]["type"],

#                 alpha=CONFIG[
#                     "partition"
#                 ].get(
#                     "alpha",
#                     0.5
#                 ),

#                 batch_size=CONFIG[
#                     "training"
#                 ]["batch_size"]
#             )
#         )

#     # -------------------------------------------------
#     # GET PARAMETERS
#     # -------------------------------------------------
#     def get_parameters(
#         self,
#         config
#     ):

#         return [

#             val.cpu().numpy()

#             for _, val in
#             self.model.state_dict().items()
#         ]

#     # -------------------------------------------------
#     # SET PARAMETERS
#     # -------------------------------------------------
#     def set_parameters(
#         self,
#         parameters
#     ):

#         params_dict = zip(
#             self.model.state_dict().keys(),
#             parameters
#         )

#         state_dict = {

#             k: torch.tensor(v)

#             for k, v in params_dict
#         }

#         self.model.load_state_dict(
#             state_dict,
#             strict=True
#         )

#     # -------------------------------------------------
#     # FIT
#     # -------------------------------------------------
#     def fit(
#         self,
#         parameters,
#         config
#     ):

#         self.set_parameters(parameters)

#         train_model(

#             self.model,

#             self.trainloader,

#             epochs=CONFIG[
#                 "training"
#             ]["local_epochs"],

#             learning_rate=CONFIG[
#                 "training"
#             ]["learning_rate"]
#         )

#         checkpoint_path = os.path.join(

#             "results",

#             CONFIG["experiment_name"],

#             "checkpoint.pth"
#         )

#         save_checkpoint(
#             self.model,
#             checkpoint_path
#         )

#         return (

#             self.get_parameters(config={}),

#             len(self.trainloader.dataset),

#             {}
#         )

#     # -------------------------------------------------
#     # EVALUATE
#     # -------------------------------------------------
#     def evaluate(
#         self,
#         parameters,
#         config
#     ):

#         self.set_parameters(parameters)

#         loss, accuracy = test_model(
#             self.model,
#             self.testloader
#         )

#         return (

#             float(loss),

#             len(self.testloader.dataset),

#             {
#                 "accuracy": float(accuracy)
#             }
#         )


# # =====================================================
# # CLIENT FUNCTION
# # =====================================================
# def client_fn(cid):

#     return FlowerClient(cid)


# # =====================================================
# # START CLIENT
# # =====================================================
# if __name__ == "__main__":

#     fl.client.start_numpy_client(

#         server_address="127.0.0.1:8080",

#         client=client_fn(args.cid)
#     )
# =====================================================
# client.py
# =====================================================

import os
import sys
import argparse

from copy import deepcopy

import torch
import flwr as fl
import pandas as pd

from torch import nn

# =====================================================
# PROJECT ROOT
# =====================================================
sys.path.append(
    os.path.abspath(
        os.path.dirname(__file__)
    )
)

# =====================================================
# ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True
)

parser.add_argument(
    "--cid",
    type=str,
    required=True
)

args = parser.parse_args()

# =====================================================
# PROJECT IMPORTS
# =====================================================
from models.cnn import SimpleCNN

from datasets.mnist import load_mnist

from utils.config import load_config

from utils.checkpoint import save_checkpoint

from strategies.fedprox import fedprox_regularizer

from privacy.dp_engine import DPEngine

# =====================================================
# LOAD CONFIG
# =====================================================
CONFIG = load_config(
    args.config
)

# =====================================================
# DEVICE
# =====================================================
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}\n")

# =====================================================
# TRAIN FUNCTION
# =====================================================
def train_model(
    model,
    trainloader,
    optimizer,
    epochs
):

    criterion = nn.CrossEntropyLoss()

    # ================================================
    # IMPORTANT FOR TRAINING
    # ================================================
    model.train()

    # ================================================
    # FEDPROX GLOBAL MODEL
    # ================================================
    global_model = deepcopy(model)

    # ================================================
    # TRAINING LOOP
    # ================================================
    for epoch in range(epochs):

        running_loss = 0.0

        for images, labels in trainloader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            # ========================================
            # BASE LOSS
            # ========================================
            loss = criterion(
                outputs,
                labels
            )

            # ========================================
            # FEDPROX REGULARIZATION
            # ========================================
            if CONFIG[
                "aggregation"
            ]["strategy"] == "fedprox":

                mu = CONFIG[
                    "fedprox"
                ]["mu"]

                prox_loss = fedprox_regularizer(

                    model,

                    global_model,

                    mu
                )

                loss += prox_loss

            # ========================================
            # BACKPROP
            # ========================================
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # ============================================
        # EPOCH LOSS
        # ============================================
        avg_loss = (
            running_loss /
            len(trainloader)
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

# =====================================================
# TEST FUNCTION
# =====================================================
def test_model(
    model,
    testloader
):

    criterion = nn.CrossEntropyLoss()

    model.eval()

    correct = 0

    total = 0

    loss_total = 0.0

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss_total += loss.item()

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = correct / total

    avg_loss = (
        loss_total /
        len(testloader)
    )

    return avg_loss, accuracy

# =====================================================
# FLOWER CLIENT
# =====================================================
class FlowerClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid
    ):

        self.cid = cid

        self.model = SimpleCNN().to(DEVICE)

        # ============================================
        # DIFFERENTIAL PRIVACY
        # ============================================
        self.dp_engine = None

        self.epsilon_history = []

        # ============================================
        # DATALOADERS
        # ============================================
        self.trainloader, self.testloader = (

            load_mnist(

                client_id=int(cid),

                num_clients=CONFIG[
                    "clients"
                ]["num_clients"],

                partition_type=CONFIG[
                    "partition"
                ]["type"],

                alpha=CONFIG[
                    "partition"
                ].get(
                    "alpha",
                    0.5
                ),

                batch_size=CONFIG[
                    "training"
                ]["batch_size"]
            )
        )

        # ============================================
        # OPTIMIZER
        # ============================================
        self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=CONFIG[
                "training"
            ]["learning_rate"]
        )

        # ============================================
        # ENABLE DIFFERENTIAL PRIVACY
        # ============================================
        if CONFIG["privacy"]["enabled"]:

            print(
                "\n[INFO] Initializing "
                "Differential Privacy Engine\n"
            )

            self.dp_engine = DPEngine(

                noise_multiplier=CONFIG[
                    "privacy"
                ]["noise_multiplier"],

                max_grad_norm=CONFIG[
                    "privacy"
                ]["max_grad_norm"],

                delta=float(
                    CONFIG[
                        "privacy"
                    ]["delta"]
                )
            )

            # ========================================
            # REQUIRED FOR OPACUS
            # ========================================
            self.model.train()

            # ========================================
            # MAKE PRIVATE ONLY ONCE
            # ========================================
            self.model, self.optimizer, self.trainloader = (

                self.dp_engine.make_private(

                    self.model,

                    self.optimizer,

                    self.trainloader
                )
            )

    # =================================================
    # GET PARAMETERS
    # =================================================
    def get_parameters(
        self,
        config
    ):

        return [

            val.cpu().numpy()

            for _, val in
            self.model.state_dict().items()
        ]

    # =================================================
    # SET PARAMETERS
    # =================================================
    def set_parameters(
        self,
        parameters
    ):

        params_dict = zip(

            self.model.state_dict().keys(),

            parameters
        )

        state_dict = {

            k: torch.tensor(v)

            for k, v in params_dict
        }

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

    # =================================================
    # FIT
    # =================================================
    def fit(
        self,
        parameters,
        config
    ):

        self.set_parameters(parameters)

        train_model(

            self.model,

            self.trainloader,

            self.optimizer,

            epochs=CONFIG[
                "training"
            ]["local_epochs"]
        )

        # ============================================
        # EPSILON TRACKING
        # ============================================
        if self.dp_engine is not None:

            epsilon = self.dp_engine.get_epsilon()

            self.epsilon_history.append({

                "round":
                    len(self.epsilon_history) + 1,

                "epsilon":
                    epsilon
            })

            print(
                f"[DP] ε = {epsilon:.4f}"
            )

            epsilon_df = pd.DataFrame(
                self.epsilon_history
            )

            epsilon_path = os.path.join(

                "results",

                CONFIG[
                    "experiment_name"
                ],

                "epsilon.csv"
            )

            epsilon_df.to_csv(

                epsilon_path,

                index=False
            )

        # ============================================
        # SAVE CHECKPOINT
        # ============================================
        checkpoint_path = os.path.join(

            "results",

            CONFIG[
                "experiment_name"
            ],

            "checkpoint.pth"
        )

        save_checkpoint(

            self.model,

            checkpoint_path
        )

        return (

            self.get_parameters(config={}),

            len(self.trainloader.dataset),

            {}
        )

    # =================================================
    # EVALUATE
    # =================================================
    def evaluate(
        self,
        parameters,
        config
    ):

        self.set_parameters(parameters)

        loss, accuracy = test_model(

            self.model,

            self.testloader
        )

        return (

            float(loss),

            len(self.testloader.dataset),

            {
                "accuracy": float(accuracy)
            }
        )

# =====================================================
# CLIENT FUNCTION
# =====================================================
def client_fn(cid):

    return FlowerClient(cid)

# =====================================================
# START CLIENT
# =====================================================
if __name__ == "__main__":

    fl.client.start_numpy_client(

        server_address="127.0.0.1:8080",

        client=client_fn(args.cid)
    )