# # datasets/mnist.py

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split


# def load_mnist(client_id, num_clients, batch_size=32):

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     train_dataset = datasets.MNIST(
#         root="./datasets",
#         train=True,
#         download=True,
#         transform=transform
#     )

#     test_dataset = datasets.MNIST(
#         root="./datasets",
#         train=False,
#         download=True,
#         transform=transform
#     )

#     # IID partition
#     partition_size = len(train_dataset) // num_clients
#     lengths = [partition_size] * num_clients

#     subsets = random_split(train_dataset, lengths)

#     client_dataset = subsets[client_id]

#     train_loader = DataLoader(
#         client_dataset,
#         batch_size=batch_size,
#         shuffle=True
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )

#     return train_loader, test_loader


# datasets/mnist.py

# import numpy as np

# from torchvision import datasets, transforms

# from torch.utils.data import DataLoader, Subset


# def iid_partition(dataset, num_clients):

#     num_items = len(dataset) // num_clients

#     all_indices = np.random.permutation(len(dataset))

#     client_indices = []

#     for i in range(num_clients):

#         start = i * num_items
#         end = start + num_items

#         client_indices.append(all_indices[start:end])

#     return client_indices


# def label_skew_partition(dataset, num_clients, shards_per_client=2):

#     labels = np.array(dataset.targets)

#     indices = np.arange(len(dataset))

#     sorted_indices = indices[np.argsort(labels)]

#     num_shards = num_clients * shards_per_client

#     shard_size = len(dataset) // num_shards

#     shards = []

#     for i in range(num_shards):

#         start = i * shard_size
#         end = start + shard_size

#         shards.append(sorted_indices[start:end])

#     np.random.shuffle(shards)

#     client_indices = []

#     for i in range(num_clients):

#         client_shards = shards[
#             i * shards_per_client:
#             (i + 1) * shards_per_client
#         ]

#         client_indices.append(
#             np.concatenate(client_shards)
#         )

#     return client_indices


# def dirichlet_partition(dataset, num_clients, alpha=0.5):

#     labels = np.array(dataset.targets)

#     num_classes = len(np.unique(labels))

#     client_indices = [[] for _ in range(num_clients)]

#     for c in range(num_classes):

#         class_indices = np.where(labels == c)[0]

#         np.random.shuffle(class_indices)

#         proportions = np.random.dirichlet(
#             alpha=np.repeat(alpha, num_clients)
#         )

#         proportions = (
#             np.cumsum(proportions) * len(class_indices)
#         ).astype(int)[:-1]

#         split_indices = np.split(
#             class_indices,
#             proportions
#         )

#         for client_id, idx in enumerate(split_indices):

#             client_indices[client_id].extend(idx)

#     return client_indices


# def load_mnist(
#     client_id,
#     num_clients,
#     batch_size=32,
#     partition_type="iid",
#     alpha=0.5
# ):

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     train_dataset = datasets.MNIST(
#         root="./datasets",
#         train=True,
#         download=True,
#         transform=transform
#     )

#     test_dataset = datasets.MNIST(
#         root="./datasets",
#         train=False,
#         download=True,
#         transform=transform
#     )

#     # -----------------------------
#     # Partition Selection
#     # -----------------------------
#     if partition_type == "iid":

#         client_indices = iid_partition(
#             train_dataset,
#             num_clients
#         )

#     elif partition_type == "label_skew":

#         client_indices = label_skew_partition(
#             train_dataset,
#             num_clients
#         )

#     elif partition_type == "dirichlet":

#         client_indices = dirichlet_partition(
#             train_dataset,
#             num_clients,
#             alpha=alpha
#         )

#     else:
#         raise ValueError("Invalid partition type")

#     train_subset = Subset(
#         train_dataset,
#         client_indices[client_id]
#     )

#     train_loader = DataLoader(
#         train_subset,
#         batch_size=batch_size,
#         shuffle=True
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )

#     return train_loader, test_loader

# datasets/mnist.py

import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# -----------------------------------------------------
# IID Partition
# -----------------------------------------------------
def iid_partition(dataset, num_clients):

    num_items = len(dataset) // num_clients

    all_indices = np.random.permutation(len(dataset))

    client_indices = []

    for i in range(num_clients):

        start = i * num_items
        end = start + num_items

        client_indices.append(
            all_indices[start:end]
        )

    return client_indices


# -----------------------------------------------------
# Label Skew Partition
# -----------------------------------------------------
def label_skew_partition(
    dataset,
    num_clients,
    shards_per_client=2
):

    labels = np.array(dataset.targets)

    indices = np.arange(len(dataset))

    sorted_indices = indices[np.argsort(labels)]

    num_shards = num_clients * shards_per_client

    shard_size = len(dataset) // num_shards

    shards = []

    for i in range(num_shards):

        start = i * shard_size
        end = start + shard_size

        shards.append(
            sorted_indices[start:end]
        )

    np.random.shuffle(shards)

    client_indices = []

    for i in range(num_clients):

        client_shards = shards[
            i * shards_per_client:
            (i + 1) * shards_per_client
        ]

        client_indices.append(
            np.concatenate(client_shards)
        )

    return client_indices


# -----------------------------------------------------
# Dirichlet Partition
# -----------------------------------------------------
def dirichlet_partition(
    dataset,
    num_clients,
    alpha=0.5
):

    labels = np.array(dataset.targets)

    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):

        class_indices = np.where(labels == c)[0]

        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(
            alpha=np.repeat(alpha, num_clients)
        )

        proportions = (
            np.cumsum(proportions) * len(class_indices)
        ).astype(int)[:-1]

        split_indices = np.split(
            class_indices,
            proportions
        )

        for client_id, idx in enumerate(split_indices):

            client_indices[client_id].extend(idx)

    return client_indices


# -----------------------------------------------------
# Load MNIST
# -----------------------------------------------------
def load_mnist(
    client_id,
    num_clients,
    batch_size=32,
    partition_type="iid",
    alpha=0.5
):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./datasets",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./datasets",
        train=False,
        download=True,
        transform=transform
    )

    # -------------------------------------------------
    # Select Partition Strategy
    # -------------------------------------------------
    if partition_type == "iid":

        client_indices = iid_partition(
            train_dataset,
            num_clients
        )

    elif partition_type == "label_skew":

        client_indices = label_skew_partition(
            train_dataset,
            num_clients
        )

    elif partition_type == "dirichlet":

        client_indices = dirichlet_partition(
            train_dataset,
            num_clients,
            alpha
        )

    else:
        raise ValueError(
            "Invalid partition_type"
        )

    # -------------------------------------------------
    # Client Subset
    # -------------------------------------------------
    train_subset = Subset(
        train_dataset,
        client_indices[client_id]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader