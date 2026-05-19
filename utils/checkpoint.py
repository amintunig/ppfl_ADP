# utils/checkpoint.py

import torch


def save_checkpoint(
    model,
    path
):

    torch.save(
        model.state_dict(),
        path
    )

    print(f"Checkpoint saved: {path}")