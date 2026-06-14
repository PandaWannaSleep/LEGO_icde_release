from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lego.data.operator_context_extractor import OperatorContextExtractor, _unwrap_system_metrics
from lego.data.plan_preprocessor import iter_plan_nodes, normalize_plan_record
from .checkpoint_loader import build_registry_from_manifest, load_checkpoint_manifest
from .config import InferenceConfig
from .registry import OperatorInferencerRegistry


logger = logging.getLogger(__name__)

VALID_MANIFEST_IDS = frozenset({"no_pretrain", "contrastive_only", "cost_only", "full"})


@dataclass(frozen=True)
class PlanOperatorEmbeddingRow:
    row_index: int
    plan_node_index: int
    operator_type: str
    relation_name: str
    source_path: str
    parent_operator: str
    embedding_source: str


@dataclass(frozen=True)
class PlanEmbeddingResult:
    embeddings: np.ndarray
    rows: tuple[PlanOperatorEmbeddingRow, ...]
    fallback_operators: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedPlanNodeRef:
    plan_index: int
    plan_node_index: int
    operator_type: str
    relation_name: str
    source_path: str
    parent_operator: str


class LEGOOperatorEmbedder:
    def __init__(
        self,
        extractor: OperatorContextExtractor,
        registry: OperatorInferencerRegistry,
        *,
        _fallback_embedding_dim: int = 128,
    ):
        self.extractor = extractor
        self.registry = registry
        self._fallback_embedding_dim = _fallback_embedding_dim
        self.manifest_id: str | None = None
        self._log_loaded_encoders()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        templates_dir: str | Path,
        task_name: str,
        extractor: OperatorContextExtractor,
        *,
        device: str | None = None,
        embedding_dim: int = 128,
    ) -> "LEGOOperatorEmbedder":
        manifest = load_checkpoint_manifest(manifest_path)
        registry = build_registry_from_manifest(
            task_name=task_name,
            manifest=manifest,
            templates_dir=templates_dir,
            device=device,
        )
        embedder = cls(extractor, registry, _fallback_embedding_dim=embedding_dim)
        embedder.manifest_id = str(Path(manifest_path))
        return embedder

    @classmethod
    def from_manifest_id(
        cls,
        manifest_id: str,
        manifest_dir: str | Path,
        task_name: str,
        extractor: OperatorContextExtractor,
        *,
        templates_dir: str | Path | None = None,
        device: str | None = None,
        embedding_dim: int = 128,
    ) -> "LEGOOperatorEmbedder":
        """Factory: build an embedder from a manifest_id.

        manifest_id selects which pretraining objective's encoder to use.
        If manifest_id == "no_pretrain" OR if the manifest file is absent,
        returns a zero-fallback embedder (no registry loaded) and emits a
        UserWarning.

        The manifest file is expected at:
            manifest_dir / f"{manifest_id}_manifest.json"

        Parameters
        ----------
        manifest_id:
            One of "no_pretrain", "contrastive_only", "cost_only", "full".
        manifest_dir:
            Directory containing manifest JSON files.
        task_name:
            Task name key inside the manifest (e.g. "runtime_cost").
        extractor:
            Shared OperatorContextExtractor instance.
        embedding_dim:
            Fallback embedding dim when registry is empty (no_pretrain / missing manifest).
        """
        if manifest_id not in VALID_MANIFEST_IDS:
            raise ValueError(
                f"Invalid manifest_id {manifest_id!r}. "
                f"Must be one of {sorted(VALID_MANIFEST_IDS)}."
            )

        manifest_dir = Path(manifest_dir)

        if manifest_id == "no_pretrain":
            warnings.warn(
                f"manifest_id='no_pretrain': no real checkpoint exists. "
                f"Embedder will produce zero vectors of dimension {embedding_dim}.",
                UserWarning,
                stacklevel=2,
            )
            registry = OperatorInferencerRegistry(task_name=task_name)
            embedder = cls(extractor, registry, _fallback_embedding_dim=embedding_dim)
            embedder.manifest_id = manifest_id
            return embedder

        manifest_path = manifest_dir / f"{manifest_id}_manifest.json"
        if not manifest_path.exists():
            warnings.warn(
                f"Manifest file not found: {manifest_path}. "
                f"Embedder will produce zero vectors of dimension {embedding_dim}.",
                UserWarning,
                stacklevel=2,
            )
            registry = OperatorInferencerRegistry(task_name=task_name)
            embedder = cls(extractor, registry, _fallback_embedding_dim=embedding_dim)
            embedder.manifest_id = manifest_id
            return embedder

        if templates_dir is None:
            raise ValueError(
                "templates_dir is required when loading an existing LEGO manifest"
            )
        manifest = load_checkpoint_manifest(manifest_path)
        registry = build_registry_from_manifest(
            task_name=task_name,
            manifest=manifest,
            templates_dir=templates_dir,
            device=device,
        )
        embedder = cls(extractor, registry, _fallback_embedding_dim=embedding_dim)
        embedder.manifest_id = manifest_id
        return embedder

    def _log_loaded_encoders(self) -> None:
        """One INFO log line per registered encoder describing which
        pre-training objective produced it.

        Reads ``registry.metadata`` populated by
        :func:`build_registry_from_manifest` and prints one line per
        operator. Operators whose registry metadata lacks a
        ``pretraining_objective`` (legacy / hand-built registries) get a
        single line tagged ``unknown`` rather than being silently
        skipped — this keeps the audit trail honest.
        """
        if not self.registry.inferencers:
            return
        metadata = self.registry.metadata
        for operator_type in self.registry.inferencers.keys():
            entry = metadata.get(operator_type, {})
            objective = entry.get("pretraining_objective") if entry else None
            logger.info(
                "LEGOOperatorEmbedder loaded encoder task=%s operator=%s pretraining_objective=%s",
                self.registry.task_name,
                operator_type,
                objective if objective is not None else "unknown",
            )

    def embed_plan_record(
        self,
        plan_record: dict,
        inference_config: InferenceConfig,
        source_path: str | None = None,
        strict: bool = True,
    ) -> PlanEmbeddingResult:
        normalized_record = normalize_plan_record(plan_record)
        plan_root = normalized_record["planinfo"]["Plan"]
        plan_level_metrics = _unwrap_system_metrics(normalized_record.get("planinfo"))
        rows: list[PlanOperatorEmbeddingRow] = []
        embeddings: list[np.ndarray] = []
        fallback_operators: list[dict[str, Any]] = []

        expected_embedding_dim = self._resolve_embedding_dim()

        for plan_node_index, node in enumerate(iter_plan_nodes(plan_root)):
            context = self.extractor.extract_operator(
                node=node,
                query_text=normalized_record.get("query"),
                config=normalized_record.get("config", {}),
                table_heat_metrics=normalized_record.get("table_heat_metrics", {}),
                source_path=source_path,
                plan_level_metrics=plan_level_metrics,
            )
            registry_embedding = self.registry.embed(
                context=context,
                inference_config=inference_config,
                strict=strict,
            )
            if registry_embedding is None:
                info = {
                    "plan_node_index": plan_node_index,
                    "operator_type": context.operator_type,
                    "relation_name": context.metadata.relation_name or "",
                    "parent_operator": context.metadata.parent_operator,
                    "source_path": context.metadata.source_path or "",
                    "embedding_source": "fallback_zero",
                }
                if strict:
                    raise KeyError(
                        f"No embedding inferencer registered for operator type {context.operator_type!r} at plan node {plan_node_index}"
                    )
                fallback_operators.append(info)
                embedding = np.zeros((expected_embedding_dim,), dtype=np.float32)
                embedding_source = "fallback_zero"
            else:
                embedding = np.asarray(registry_embedding.embedding.embedding, dtype=np.float32)
                embedding_source = "model"
            if embedding.ndim != 1:
                raise ValueError(
                    f"Expected a 1D operator embedding for {context.operator_type!r}, got shape {embedding.shape}"
                )
            if int(embedding.shape[0]) != expected_embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {expected_embedding_dim}, got {embedding.shape[0]} for operator {context.operator_type!r}"
                )
            embeddings.append(embedding)
            rows.append(
                PlanOperatorEmbeddingRow(
                    row_index=len(rows),
                    plan_node_index=plan_node_index,
                    operator_type=context.operator_type,
                    relation_name=context.metadata.relation_name or "",
                    source_path=context.metadata.source_path or "",
                    parent_operator=context.metadata.parent_operator,
                    embedding_source=embedding_source,
                )
            )

        if embeddings:
            embedding_matrix = np.stack(embeddings, axis=0)
        else:
            embedding_matrix = np.zeros((0, expected_embedding_dim), dtype=np.float32)

        return PlanEmbeddingResult(
            embeddings=embedding_matrix,
            rows=tuple(rows),
            fallback_operators=tuple(fallback_operators),
            metadata={
                "task_name": self.registry.task_name,
                "source_path": source_path or "",
                "strict": strict,
                "plan_node_count": plan_node_index + 1 if "plan_node_index" in locals() else 0,
                "embedding_dim": expected_embedding_dim,
            },
        )

    def embed_plan_records(
        self,
        plan_records: list[dict],
        inference_config: InferenceConfig,
        source_paths: list[str | None] | None = None,
        strict: bool = True,
    ) -> list[PlanEmbeddingResult]:
        """Embed multiple plan records with per-operator-type batching.

        Each LEGO checkpoint is operator-type specific, so batching happens
        independently within every operator type and the resulting embeddings
        are scattered back to each plan's original postorder node index.
        """
        if not plan_records:
            return []
        if source_paths is None:
            source_paths = [None] * len(plan_records)
        if len(source_paths) != len(plan_records):
            raise ValueError(
                f"source_paths length {len(source_paths)} does not match plan_records length {len(plan_records)}"
            )

        expected_embedding_dim = self._resolve_embedding_dim()
        normalized_records: list[dict] = []
        per_plan_embeddings: list[list[np.ndarray | None]] = []
        per_plan_meta: list[list[dict[str, str | int]]] = []
        per_plan_sources: list[list[str]] = []
        fallback_by_plan: list[list[dict[str, Any]]] = []
        contexts_by_operator: dict[str, list[Any]] = {}
        refs_by_operator: dict[str, list[BatchedPlanNodeRef]] = {}

        for plan_index, (plan_record, source_path) in enumerate(zip(plan_records, source_paths)):
            normalized_record = normalize_plan_record(plan_record)
            normalized_records.append(normalized_record)
            plan_root = normalized_record["planinfo"]["Plan"]
            plan_level_metrics = _unwrap_system_metrics(normalized_record.get("planinfo"))

            embeddings: list[np.ndarray | None] = []
            metas: list[dict[str, str | int]] = []
            sources: list[str] = []
            fallbacks: list[dict[str, Any]] = []

            for plan_node_index, node in enumerate(iter_plan_nodes(plan_root)):
                context = self.extractor.extract_operator(
                    node=node,
                    query_text=normalized_record.get("query"),
                    config=normalized_record.get("config", {}),
                    table_heat_metrics=normalized_record.get("table_heat_metrics", {}),
                    source_path=source_path,
                    plan_level_metrics=plan_level_metrics,
                )
                relation_name = context.metadata.relation_name or ""
                parent_operator = context.metadata.parent_operator
                context_source_path = context.metadata.source_path or ""
                metas.append(
                    {
                        "plan_node_index": plan_node_index,
                        "operator_type": context.operator_type,
                        "relation_name": relation_name,
                        "source_path": context_source_path,
                        "parent_operator": parent_operator,
                    }
                )

                inferencer = self.registry.get(context.operator_type)
                if inferencer is None:
                    info = {
                        "plan_node_index": plan_node_index,
                        "operator_type": context.operator_type,
                        "relation_name": relation_name,
                        "parent_operator": parent_operator,
                        "source_path": context_source_path,
                        "embedding_source": "fallback_zero",
                    }
                    if strict:
                        raise KeyError(
                            f"No embedding inferencer registered for operator type {context.operator_type!r} at plan {plan_index} node {plan_node_index}"
                        )
                    embeddings.append(np.zeros((expected_embedding_dim,), dtype=np.float32))
                    sources.append("fallback_zero")
                    fallbacks.append(info)
                    continue

                embeddings.append(None)
                sources.append("model")
                contexts_by_operator.setdefault(context.operator_type, []).append(context)
                refs_by_operator.setdefault(context.operator_type, []).append(
                    BatchedPlanNodeRef(
                        plan_index=plan_index,
                        plan_node_index=plan_node_index,
                        operator_type=context.operator_type,
                        relation_name=relation_name,
                        source_path=context_source_path,
                        parent_operator=parent_operator,
                    )
                )

            per_plan_embeddings.append(embeddings)
            per_plan_meta.append(metas)
            per_plan_sources.append(sources)
            fallback_by_plan.append(fallbacks)

        for operator_type, contexts in contexts_by_operator.items():
            inferencer = self.registry.get(operator_type)
            if inferencer is None:
                continue
            batch_cag = inferencer.build_cag_batch(contexts, strict=strict)
            batch_result = inferencer.refinement_engine.refine_batch(
                batch_cag,
                inference_config=inference_config,
            )
            batch_embeddings = np.asarray(batch_result.final_pooled_embeddings, dtype=np.float32)
            refs = refs_by_operator[operator_type]
            if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(refs):
                raise ValueError(
                    f"Batch embedding shape mismatch for {operator_type!r}: got {batch_embeddings.shape}, expected first dimension {len(refs)}"
                )
            if int(batch_embeddings.shape[1]) != expected_embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {expected_embedding_dim}, got {batch_embeddings.shape[1]} for operator {operator_type!r}"
                )

            for embedding, ref in zip(batch_embeddings, refs):
                per_plan_embeddings[ref.plan_index][ref.plan_node_index] = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

        results: list[PlanEmbeddingResult] = []
        for plan_index, (source_path, normalized_record) in enumerate(zip(source_paths, normalized_records)):
            embeddings: list[np.ndarray] = []
            rows: list[PlanOperatorEmbeddingRow] = []
            for node_meta, embedding, embedding_source in zip(
                per_plan_meta[plan_index],
                per_plan_embeddings[plan_index],
                per_plan_sources[plan_index],
            ):
                if embedding is None:
                    if strict:
                        raise RuntimeError(
                            f"Missing batched embedding for plan {plan_index} node {node_meta['plan_node_index']}"
                        )
                    embedding = np.zeros((expected_embedding_dim,), dtype=np.float32)
                    embedding_source = "fallback_zero"
                embedding = np.asarray(embedding, dtype=np.float32)
                if embedding.ndim != 1:
                    raise ValueError(
                        f"Expected a 1D operator embedding for {node_meta['operator_type']!r}, got shape {embedding.shape}"
                    )
                if int(embedding.shape[0]) != expected_embedding_dim:
                    raise ValueError(
                        f"Embedding dim mismatch: expected {expected_embedding_dim}, got {embedding.shape[0]} for operator {node_meta['operator_type']!r}"
                    )
                embeddings.append(embedding)
                rows.append(
                    PlanOperatorEmbeddingRow(
                        row_index=len(rows),
                        plan_node_index=int(node_meta["plan_node_index"]),
                        operator_type=str(node_meta["operator_type"]),
                        relation_name=str(node_meta["relation_name"]),
                        source_path=str(node_meta["source_path"]),
                        parent_operator=str(node_meta["parent_operator"]),
                        embedding_source=embedding_source,
                    )
                )

            if embeddings:
                embedding_matrix = np.stack(embeddings, axis=0)
            else:
                embedding_matrix = np.zeros((0, expected_embedding_dim), dtype=np.float32)

            results.append(
                PlanEmbeddingResult(
                    embeddings=embedding_matrix,
                    rows=tuple(rows),
                    fallback_operators=tuple(fallback_by_plan[plan_index]),
                    metadata={
                        "task_name": self.registry.task_name,
                        "source_path": source_path or "",
                        "strict": strict,
                        "plan_node_count": len(per_plan_meta[plan_index]),
                        "embedding_dim": expected_embedding_dim,
                        "batch_plan_count": len(plan_records),
                    },
                )
            )

        return results

    def _resolve_embedding_dim(self) -> int:
        first_inferencer = next(iter(self.registry.inferencers.values()), None)
        if first_inferencer is None:
            return self._fallback_embedding_dim
        return int(first_inferencer.refinement_engine.operator_encoder.config.output_dim)
