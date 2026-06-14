"""Build a per-task checkpoint manifest with encoder metadata.

The manifest is a JSON file that downstream LEGO consumers (cost
estimation, knob tuning, workload characterization, index recommendation)
read to find the right encoder for each operator type. Each per-operator entry
is a dict that exposes the trainer-level fields needed to identify the
encoder's pre-training objective without loading the checkpoint pickle:

* ``pretraining_objective`` — one of the four objective modes
  (``no_pretrain`` / ``contrastive_only`` / ``cost_only`` / ``full``);
  legacy ``OperatorTrainer`` checkpoints surface as ``"cost_only"``.
* ``alpha`` / ``beta`` — outer InfoNCE / cost loss weights.
* ``gamma_conn`` / ``gamma_sp`` — graph-reg sub-weights.
* ``tau_c`` — InfoNCE temperature.
* ``T_max`` — training-time iterative-refinement budget.
* ``metric_type`` — which graph-learner branch was trained
  (for example, ``learned_weight`` or ``weighted_cosine``).

All of these are optional in the on-disk manifest: legacy checkpoints
without a ``trainer_config`` block surface them as ``null``. The
:func:`build_manifest` function is a thin wrapper around
:func:`extract_manifest_metadata`; consumers that already have a
checkpoint-dir → operator map can call it once at the end of a training
sweep to write a single manifest covering every operator they trained.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint_loader import extract_manifest_metadata


__all__ = ["build_manifest"]


def build_manifest(
    task_name: str,
    operator_to_checkpoint_dir: dict[str, str | Path],
    output_path: str | Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build a manifest JSON for one task.

    Parameters
    ----------
    task_name:
        The task this manifest covers, e.g. ``"runtime_cost"`` or
        ``"startup_cost"``. This becomes the top-level key of the JSON.
    operator_to_checkpoint_dir:
        Mapping from operator type (e.g. ``"Hash Join"``) to the
        directory that holds its checkpoint (the directory passed to
        :func:`save_operator_checkpoint`).
    output_path:
        File to write the JSON to. Parent directories are created.

    Returns
    -------
    The same dict that gets written to disk: ``{task_name: {operator_type:
    entry_dict}}``. Each ``entry_dict`` contains ``checkpoint_dir`` plus
    the metadata fields. Returning the dict lets callers pipe it back
    into :func:`build_registry_from_manifest` without re-reading the
    file.
    """
    task_entries: dict[str, dict[str, Any]] = {}
    for operator_type, checkpoint_dir in operator_to_checkpoint_dir.items():
        task_entries[str(operator_type)] = extract_manifest_metadata(checkpoint_dir)

    manifest: dict[str, dict[str, dict[str, Any]]] = {str(task_name): task_entries}

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
