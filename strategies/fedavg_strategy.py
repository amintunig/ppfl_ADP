# strategies/fedavg_strategy.py

import flwr as fl

from utils.logger import ExperimentLogger


# -----------------------------------------------------
# Initialize Logger
# -----------------------------------------------------
logger = ExperimentLogger(log_dir="logs")


# -----------------------------------------------------
# Custom FedAvg Strategy
# -----------------------------------------------------
class CustomFedAvg(fl.server.strategy.FedAvg):

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures
    ):

        aggregated_result = super().aggregate_evaluate(
            server_round,
            results,
            failures
        )

        if aggregated_result is not None:

            loss, metrics = aggregated_result

            accuracy = metrics.get("accuracy", 0.0)

            print(
                f"\n[ROUND {server_round}] "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

            # Save metrics
            logger.log_round(
                round_num=server_round,
                loss=loss,
                accuracy=accuracy
            )

        return aggregated_result

    # -------------------------------------------------
    # Save Summary at End
    # -------------------------------------------------
    def finalize_logs(self):

        logger.save_summary()