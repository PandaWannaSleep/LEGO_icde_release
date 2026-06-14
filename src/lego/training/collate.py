from __future__ import annotations

import numpy as np
import torch

from .dataset import OperatorTrainingExample


def collate_operator_examples(batch: list[OperatorTrainingExample]) -> dict[str, object]:
    node_values = np.stack([example.node_values for example in batch], axis=0)
    initial_adjacency = np.stack([example.initial_adjacency for example in batch], axis=0)
    return {
        "node_values": torch.from_numpy(node_values).to(torch.float32),
        "initial_adjacency": torch.from_numpy(initial_adjacency).to(torch.float32),
        "targets": torch.tensor([example.target for example in batch], dtype=torch.float32),
        "contexts": [example.context for example in batch],
    }
