from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lego.cag.builder import OperatorCAGBuilder
from lego.cag.instance import BatchOperatorCAG, OperatorCAG
from lego.cag.template import CAGTemplate
from lego.data.operator_context import OperatorContext
from lego.model.refinement_engine import RefinementEngine
from .config import InferenceConfig, InferenceMode


@dataclass(frozen=True)
class OperatorPrediction:
    operator_type: str
    predicted_cost: float
    final_adjacency: np.ndarray
    iterations: int
    mode: InferenceMode
    stop_reason: str
    metadata: dict[str, str | float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorEmbedding:
    operator_type: str
    embedding: np.ndarray
    iterations: int
    mode: InferenceMode
    stop_reason: str
    metadata: dict[str, str | float | int] = field(default_factory=dict)


class OperatorInferencer:
    def __init__(
        self,
        templates: dict[str, CAGTemplate],
        refinement_engine: RefinementEngine,
        cag_builder: OperatorCAGBuilder | None = None,
    ):
        self.templates = templates
        self.refinement_engine = refinement_engine
        self.cag_builder = cag_builder or OperatorCAGBuilder()

    def build_cag(self, context: OperatorContext, strict: bool = True) -> OperatorCAG:
        template = self.templates.get(context.operator_type)
        if template is None:
            raise KeyError(f"No CAG template registered for operator type {context.operator_type!r}")
        return self.cag_builder.build(context=context, template=template, strict=strict)

    def build_cag_batch(self, contexts: list[OperatorContext], strict: bool = True) -> BatchOperatorCAG:
        if not contexts:
            raise ValueError("Cannot build an empty CAG batch")

        operator_type = contexts[0].operator_type
        if any(context.operator_type != operator_type for context in contexts):
            raise ValueError("CAG batches must contain one operator type")

        template = self.templates.get(operator_type)
        if template is None:
            raise KeyError(f"No CAG template registered for operator type {operator_type!r}")

        cags = [
            self.cag_builder.build(context=context, template=template, strict=strict)
            for context in contexts
        ]
        return BatchOperatorCAG(
            operator_type=operator_type,
            contexts=tuple(contexts),
            template=template,
            node_values=np.stack([cag.node_values for cag in cags], axis=0),
            initial_adjacency=np.stack([cag.initial_adjacency for cag in cags], axis=0),
            current_adjacency=np.stack([cag.current_adjacency for cag in cags], axis=0),
        )

    def predict(self, context: OperatorContext, inference_config: InferenceConfig, strict: bool = True) -> OperatorPrediction:
        cag = self.build_cag(context=context, strict=strict)
        result = self.refinement_engine.refine(cag=cag, inference_config=inference_config)
        predicted_cost = max(float(result.predicted_cost), float(inference_config.min_prediction_floor))
        return OperatorPrediction(
            operator_type=context.operator_type,
            predicted_cost=predicted_cost,
            final_adjacency=result.final_adjacency,
            iterations=result.iterations,
            mode=result.mode,
            stop_reason=result.stop_reason,
            metadata=self._metadata_from_context(context),
        )

    def embed(self, context: OperatorContext, inference_config: InferenceConfig, strict: bool = True) -> OperatorEmbedding:
        cag = self.build_cag(context=context, strict=strict)
        result = self.refinement_engine.refine(cag=cag, inference_config=inference_config)
        return OperatorEmbedding(
            operator_type=context.operator_type,
            embedding=result.final_pooled_embedding,
            iterations=result.iterations,
            mode=result.mode,
            stop_reason=result.stop_reason,
            metadata=self._metadata_from_context(context),
        )

    def _metadata_from_context(self, context: OperatorContext) -> dict[str, str | float | int]:
        return {
            "relation_name": context.metadata.relation_name or "",
            "source_path": context.metadata.source_path or "",
            "parent_operator": context.metadata.parent_operator,
        }
