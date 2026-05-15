# client.py

import torch
import flwr as fl

from models.cnn import SimpleCNN
from datasets.mnist import load_mnist


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id, num_clients):

        self.client_id = client_id

        self.model = SimpleCNN().to(DEVICE)

        self.train_loader, self.test_loader = load_mnist(
            client_id=client_id,
            num_clients=num_clients
        )

    # -------------------------------------------------
    # Get model parameters
    # -------------------------------------------------
    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    # -------------------------------------------------
    # Set model parameters
    # -------------------------------------------------
    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters
        )

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        self.model.load_state_dict(state_dict, strict=True)

    # -------------------------------------------------
    # Local training
    # -------------------------------------------------
    def train_model(self):

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.model.train()

        for epoch in range(1):

            for images, labels in self.train_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

    # -------------------------------------------------
    # Internal evaluation function
    # -------------------------------------------------
    def evaluate_model(self):

        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()

        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                loss_total += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        avg_loss = loss_total / len(self.test_loader)

        return avg_loss, accuracy

    # -------------------------------------------------
    # Flower fit()
    # -------------------------------------------------
    def fit(self, parameters, config):

        self.set_parameters(parameters)

        self.train_model()

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {}
        )

    # -------------------------------------------------
    # Flower evaluate()
    # -------------------------------------------------
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        loss, accuracy = self.evaluate_model()

        return (
            float(loss),
            len(self.test_loader.dataset),
            {"accuracy": float(accuracy)}
        )


# -----------------------------------------------------
# Client function
# -----------------------------------------------------
def client_fn(cid: str):

    return FlowerClient(
        client_id=int(cid),
        num_clients=2
    )


# -----------------------------------------------------
# Start client
# -----------------------------------------------------
if __name__ == "__main__":

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client_fn("0")
    )