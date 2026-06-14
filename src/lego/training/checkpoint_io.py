from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_operator_checkpoint(output_dir: str | Path, checkpoint: dict[str, Any], metadata: dict[str, Any]) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target_dir / "model.pt")
    with (target_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_operator_checkpoint(output_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target_dir = Path(output_dir)
    checkpoint = torch.load(target_dir / "model.pt", map_location="cpu")
    with (target_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return checkpoint, metadata
