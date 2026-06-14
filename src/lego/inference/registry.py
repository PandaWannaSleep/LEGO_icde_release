from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lego.data.operator_context import OperatorContext
from .config import InferenceConfig
from .operator_inferencer import OperatorEmbedding, OperatorInferencer, OperatorPrediction


@dataclass(frozen=True)
class RegistryPrediction:
    task_name: str
    prediction: OperatorPrediction


@dataclass(frozen=True)
class RegistryEmbedding:
    task_name: str
    embedding: OperatorEmbedding


class OperatorInferencerRegistry:
    """
    Registry of per-operator inferencers for a single task.

    LEGO trains one model per operator/task pair. The scheduler keeps that
    organization explicit instead of assuming a single shared model across all
    operators.

    Per-operator manifest metadata is stashed in
    ``self.metadata[operator_type]`` when ``register`` is called with a
    ``metadata`` kwarg.
    """

    def __init__(self, task_name: str, inferencers: dict[str, OperatorInferencer] | None = None):
        self.task_name = task_name
        self.inferencers = inferencers or {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        operator_type: str,
        inferencer: OperatorInferencer,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.inferencers[operator_type] = inferencer
        if metadata is not None:
            self.metadata[operator_type] = dict(metadata)

    def supports(self, operator_type: str) -> bool:
        return operator_type in self.inferencers

    def get(self, operator_type: str) -> OperatorInferencer | None:
        return self.inferencers.get(operator_type)

    def predict(
        self,
        context: OperatorContext,
        inference_config: InferenceConfig,
        strict: bool = False,
    ) -> RegistryPrediction | None:
        inferencer = self.get(context.operator_type)
        if inferencer is None:
            return None
        prediction = inferencer.predict(context=context, inference_config=inference_config, strict=strict)
        return RegistryPrediction(task_name=self.task_name, prediction=prediction)

    def embed(
        self,
        context: OperatorContext,
        inference_config: InferenceConfig,
        strict: bool = False,
    ) -> RegistryEmbedding | None:
        inferencer = self.get(context.operator_type)
        if inferencer is None:
            return None
        embedding = inferencer.embed(context=context, inference_config=inference_config, strict=strict)
        return RegistryEmbedding(task_name=self.task_name, embedding=embedding)
