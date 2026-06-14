from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lego.cag.io import load_cag_template
from lego.model.cost_predictor import CostHead, CostHeadConfig
from lego.model.encoders import NodeEncoderConfig, ScalarNodeEncoder
from lego.model.graph_learner import GraphLearner, GraphLearnerConfig
from lego.model.graph_updater import GraphUpdateConfig, GraphUpdater
from lego.model.operator_encoder import OperatorEncoder, OperatorEncoderConfig
from lego.model.refinement_engine import RefinementEngine
from lego.training.checkpoint_io import load_operator_checkpoint
from .operator_inferencer import OperatorInferencer
from .registry import OperatorInferencerRegistry


logger = logging.getLogger(__name__)


# Fields surfaced from per-checkpoint metadata.json into the manifest entry.
MANIFEST_METADATA_FIELDS: tuple[str, ...] = (
    "pretraining_objective",
    "encoder_mode",
    "feature_ablation",
    "alpha",
    "beta",
    "gamma_conn",
    "gamma_sp",
    "tau_c",
    "T_max",
    "metric_type",
)


def _normalise_entry(value: str | dict[str, Any]) -> dict[str, Any]:
    """Coerce a manifest's per-operator entry into the new dict shape.

    Legacy manifests stored each operator entry as a bare string pointing at
    the checkpoint dir. Current manifests use a dict so they can carry extra
    fields such as ``pretraining_objective`` and ``alpha``. On read both
    shapes are accepted.
    """
    if isinstance(value, str):
        return {"checkpoint_dir": value}
    if isinstance(value, dict):
        if "checkpoint_dir" not in value:
            raise ValueError(
                "Manifest operator entry dict is missing required 'checkpoint_dir': "
                f"{value!r}"
            )
        # Defensive copy so mutating the loaded manifest cannot corrupt the
        # caller's view.
        return dict(value)
    raise TypeError(
        f"Manifest operator entry must be str or dict, got {type(value).__name__}"
    )


def extract_manifest_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read the saved checkpoint's metadata.json and return manifest fields.

    The fields returned mirror :data:`MANIFEST_METADATA_FIELDS` plus
    ``checkpoint_dir``. They are populated from the ``trainer_config`` and
    ``graph_learner_config`` blobs written into checkpoint metadata.
    Legacy ``OperatorTrainer`` checkpoints have no
    ``pretraining_objective`` field — for those we default to
    ``"cost_only"``.

    Missing optional fields are returned as ``None`` so legacy checkpoints
    round-trip cleanly through ``build_manifest`` without losing the
    ``checkpoint_dir`` pointer.
    """
    target_dir = Path(checkpoint_dir)
    metadata_path = target_dir / "metadata.json"
    if not metadata_path.is_file():
        # No metadata.json — return a stub with only the dir filled in so
        # the manifest can still record the pointer. This matches the
        # docstring: legacy / partial checkpoints load gracefully.
        return {"checkpoint_dir": str(target_dir), **{k: None for k in MANIFEST_METADATA_FIELDS}}

    with metadata_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    trainer_cfg = raw.get("trainer_config") or {}
    graph_learner_cfg = raw.get("graph_learner_config") or {}

    # Multi-objective checkpoints include pretraining_objective; cost-only
    # checkpoints do not.
    pretraining_objective = trainer_cfg.get("pretraining_objective")
    if pretraining_objective is None and trainer_cfg:
        pretraining_objective = "cost_only"

    # T_max: the training-time refinement budget.
    t_max = trainer_cfg.get("max_iter")

    return {
        "checkpoint_dir": str(target_dir),
        "pretraining_objective": pretraining_objective,
        "encoder_mode": trainer_cfg.get("encoder_mode") or ("iterative_graph" if trainer_cfg else None),
        "feature_ablation": raw.get("feature_ablation"),
        "alpha": trainer_cfg.get("alpha"),
        "beta": trainer_cfg.get("beta"),
        "gamma_conn": trainer_cfg.get("gamma_conn"),
        "gamma_sp": trainer_cfg.get("gamma_sp"),
        "tau_c": trainer_cfg.get("tau_c"),
        "T_max": t_max,
        "metric_type": graph_learner_cfg.get("metric_type"),
    }


def load_checkpoint_manifest(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load and normalise a checkpoint manifest.

    Returns a nested dict ``{task_name: {operator_type: entry_dict}}`` where
    each ``entry_dict`` is guaranteed to contain a ``"checkpoint_dir"`` key
    plus any metadata fields (``pretraining_objective``, ``alpha``,
    ``beta``, ``gamma_conn``, ``gamma_sp``, ``tau_c``, ``T_max``,
    ``metric_type``) that were present in the on-disk manifest. Legacy
    string-valued entries are accepted and coerced to ``{"checkpoint_dir":
    value}``.
    """
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for task_name, operator_map in manifest.items():
        normalized_task: dict[str, dict[str, Any]] = {}
        for operator, entry in operator_map.items():
            normalized_task[str(operator)] = _normalise_entry(entry)
        normalized[str(task_name)] = normalized_task
    return normalized


def build_registry_from_manifest(
    task_name: str,
    manifest: dict[str, dict[str, dict[str, Any] | str]],
    templates_dir: str | Path,
    device: str | None = None,
) -> OperatorInferencerRegistry:
    """Build an inferencer registry from a (possibly mixed-format) manifest.

    Each manifest entry can be either the legacy string form or the new
    dict form; ``_normalise_entry`` smooths over the difference. When an
    entry carries a non-``None`` ``pretraining_objective``, this is logged
    at INFO level so the audit trail of "which encoder was loaded for which
    operator with which objective" is captured in stdout. Entries missing
    that field (legacy manifests) trigger a single WARNING per operator.
    """
    registry = OperatorInferencerRegistry(task_name=task_name)
    checkpoint_map = manifest.get(task_name, {})
    for operator_type, raw_entry in checkpoint_map.items():
        entry = _normalise_entry(raw_entry)
        checkpoint_dir = entry["checkpoint_dir"]

        objective = entry.get("pretraining_objective")
        if objective is not None:
            logger.info(
                "Loading encoder task=%s operator=%s pretraining_objective=%s checkpoint_dir=%s",
                task_name,
                operator_type,
                objective,
                checkpoint_dir,
            )
        else:
            logger.warning(
                "Manifest entry for task=%s operator=%s has no pretraining_objective "
                "(legacy format); checkpoint_dir=%s",
                task_name,
                operator_type,
                checkpoint_dir,
            )

        registry.register(
            operator_type=operator_type,
            inferencer=load_inferencer_from_checkpoint(
                checkpoint_dir=checkpoint_dir,
                templates_dir=templates_dir,
                operator_type=operator_type,
                device=device,
            ),
            metadata=entry,
        )
    return registry


def load_inferencer_from_checkpoint(
    checkpoint_dir: str | Path,
    templates_dir: str | Path,
    operator_type: str | None = None,
    device: str | None = None,
) -> OperatorInferencer:
    checkpoint, metadata = load_operator_checkpoint(checkpoint_dir)
    resolved_operator = operator_type or metadata.get("operator_type")
    if not resolved_operator:
        resolved_operator = checkpoint.get("extra_metadata", {}).get("operator_type")
    if not resolved_operator:
        raise ValueError(f"Operator type is missing from checkpoint {checkpoint_dir}")

    template = _load_template_for_operator(
        operator_type=resolved_operator,
        templates_dir=templates_dir,
        metadata=metadata,
        checkpoint=checkpoint,
    )

    if "typed_node_encoder_config" in checkpoint:
        from lego.model.typed_node_encoder import TypedNodeEncoder, TypedNodeEncoderConfig

        node_encoder = TypedNodeEncoder(
            template.node_schema, TypedNodeEncoderConfig(**checkpoint["typed_node_encoder_config"])
        )
        node_encoder.load_state_dict(checkpoint["typed_node_encoder_state_dict"])
    else:
        node_encoder = ScalarNodeEncoder(NodeEncoderConfig(**checkpoint["node_encoder_config"]))
        node_encoder.load_state_dict(checkpoint["node_encoder_state_dict"])

    graph_learner = GraphLearner(GraphLearnerConfig(**checkpoint["graph_learner_config"]))
    graph_learner.load_state_dict(checkpoint["graph_learner_state_dict"])

    graph_updater = GraphUpdater(GraphUpdateConfig(**checkpoint["graph_updater_config"]))

    operator_encoder, cost_head = _load_encoder_and_cost_head(checkpoint)

    refinement_engine = RefinementEngine(
        node_encoder=node_encoder,
        graph_learner=graph_learner,
        graph_updater=graph_updater,
        operator_encoder=operator_encoder,
        cost_head=cost_head,
        encoder_mode=(checkpoint.get("trainer_config") or {}).get("encoder_mode", "iterative_graph"),
        device=device,
    )
    return OperatorInferencer(
        templates={resolved_operator: template},
        refinement_engine=refinement_engine,
    )


def _load_encoder_and_cost_head(checkpoint: dict[str, Any]) -> tuple[OperatorEncoder, CostHead]:
    """Build OperatorEncoder + CostHead from a checkpoint, handling both
    the split format (separate state_dicts) and the legacy ``cost_predictor_*``
    format that bundled the GNN backbone with the cost head.
    """
    if "operator_encoder_state_dict" in checkpoint and "cost_head_state_dict" in checkpoint:
        encoder_cfg = OperatorEncoderConfig(**checkpoint["operator_encoder_config"])
        head_cfg = CostHeadConfig(**checkpoint["cost_head_config"])
        operator_encoder = OperatorEncoder(encoder_cfg)
        operator_encoder.load_state_dict(checkpoint["operator_encoder_state_dict"])
        cost_head = CostHead(head_cfg)
        cost_head.load_state_dict(checkpoint["cost_head_state_dict"])
        return operator_encoder, cost_head

    # Legacy: cost_predictor bundled GNN layers + cost head in one state_dict.
    # Layer keys look like ``layers.{i}.{self,neighbor}_linear.{weight,bias}``;
    # head keys look like ``head.{0,3}.{weight,bias}``. Split by prefix and
    # rebuild the two new modules from a derived encoder/head config.
    legacy_cfg = checkpoint["cost_predictor_config"]
    encoder_cfg = OperatorEncoderConfig(
        hidden_dim=legacy_cfg["hidden_dim"],
        num_message_passing_layers=legacy_cfg["num_message_passing_layers"],
        pool_type=legacy_cfg["pool_type"],
        dropout=legacy_cfg["dropout"],
    )
    head_cfg = CostHeadConfig(
        input_dim=encoder_cfg.output_dim,
        hidden_dim=legacy_cfg["hidden_dim"],
        dropout=legacy_cfg["dropout"],
    )
    operator_encoder = OperatorEncoder(encoder_cfg)
    cost_head = CostHead(head_cfg)
    legacy_state = checkpoint["cost_predictor_state_dict"]
    encoder_state = {k[len("layers."):]: v for k, v in legacy_state.items() if k.startswith("layers.")}
    encoder_state = {f"layers.{k}": v for k, v in encoder_state.items()}
    head_state = {k[len("head."):]: v for k, v in legacy_state.items() if k.startswith("head.")}
    head_state = {f"head.{k}": v for k, v in head_state.items()}
    operator_encoder.load_state_dict(encoder_state)
    cost_head.load_state_dict(head_state)
    return operator_encoder, cost_head


def _load_template_for_operator(
    operator_type: str,
    templates_dir: str | Path,
    metadata: dict[str, Any],
    checkpoint: dict[str, Any],
):
    template_path = metadata.get("template_path")
    if not template_path:
        template_path = checkpoint.get("extra_metadata", {}).get("template_path")
    if not template_path:
        template_path = Path(templates_dir) / f"{operator_type.replace(' ', '_')}.pkl"
    return load_cag_template(template_path)
