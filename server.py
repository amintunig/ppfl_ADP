# # server.py

# import flwr as fl


# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,
#     fraction_evaluate=1.0,
#     min_fit_clients=2,
#     min_evaluate_clients=2,
#     min_available_clients=2,
# )


# if __name__ == "__main__":

#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         config=fl.server.ServerConfig(num_rounds=5),
#         strategy=strategy,
#     )

# server.py

# server.py

import flwr as fl

from strategies.fedavg_strategy import CustomFedAvg


# -----------------------------------------------------
# Metric Aggregation
# -----------------------------------------------------
def weighted_average(metrics):

    accuracies = [
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ]

    examples = [
        num_examples
        for num_examples, _ in metrics
    ]

    return {
        "accuracy": sum(accuracies) / sum(examples)
    }


# -----------------------------------------------------
# Strategy
# -----------------------------------------------------
strategy = CustomFedAvg(

    fraction_fit=1.0,
    fraction_evaluate=1.0,

    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,

    evaluate_metrics_aggregation_fn=weighted_average
)


# -----------------------------------------------------
# Start Server
# -----------------------------------------------------
if __name__ == "__main__":

    print("\nStarting Flower Server...\n")

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=5
        ),

        strategy=strategy,
    )

    # Save logs
    strategy.finalize_logs()

    print("\nTraining Complete.\n")