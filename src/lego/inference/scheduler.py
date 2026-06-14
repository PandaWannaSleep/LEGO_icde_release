from __future__ import annotations

import copy
from dataclasses import dataclass

from lego.data.operator_context_extractor import OperatorContextExtractor, _unwrap_system_metrics
from lego.data.plan_preprocessor import normalize_plan_record
from .config import InferenceConfig
from .plan_aggregator import PlanCost, PostgresPlanAggregator
from .registry import OperatorInferencerRegistry


@dataclass(frozen=True)
class SchedulerPrediction:
    total_cost: float
    root_cost: PlanCost
    annotated_plan: dict


class LEGOScheduler:
    """
    Plan-level scheduler boundary for LEGO inference.

    This scheduler does not know how CAGs are built internally and does not
    know how graph refinement works internally. It only orchestrates:

    1. Extract operator contexts from plan nodes.
    2. Call operator-level startup/runtime predictors.
    3. Aggregate child and local costs into plan-level costs.
    """

    def __init__(
        self,
        extractor: OperatorContextExtractor,
        runtime_registry: OperatorInferencerRegistry,
        startup_registry: OperatorInferencerRegistry | None = None,
        aggregator: PostgresPlanAggregator | None = None,
    ):
        self.extractor = extractor
        self.runtime_registry = runtime_registry
        self.startup_registry = startup_registry
        self.aggregator = aggregator or PostgresPlanAggregator()

    def predict_plan_record(
        self,
        plan_record: dict,
        inference_config: InferenceConfig,
        source_path: str | None = None,
    ) -> SchedulerPrediction:
        normalized_record = normalize_plan_record(plan_record)
        plan_root = copy.deepcopy(normalized_record["planinfo"]["Plan"])
        plan_level_metrics = _unwrap_system_metrics(normalized_record.get("planinfo"))
        root_cost = self._predict_node(
            node=plan_root,
            query_text=normalized_record.get("query"),
            config=normalized_record.get("config", {}),
            table_heat_metrics=normalized_record.get("table_heat_metrics", {}),
            inference_config=inference_config,
            source_path=source_path,
            plan_level_metrics=plan_level_metrics,
        )
        return SchedulerPrediction(
            total_cost=root_cost.total_cost,
            root_cost=root_cost,
            annotated_plan=plan_root,
        )

    def _predict_node(
        self,
        node: dict,
        query_text: str | None,
        config: dict,
        table_heat_metrics: dict,
        inference_config: InferenceConfig,
        source_path: str | None,
        plan_level_metrics: dict | None,
    ) -> PlanCost:
        normal_children = []
        subplan_children = []
        for child in node.get("Plans", []):
            if child.get("Parent Relationship") == "SubPlan" or "Subplan Name" in child:
                subplan_children.append(child)
            else:
                normal_children.append(child)

        child_costs = [
            self._predict_node(
                node=child,
                query_text=query_text,
                config=config,
                table_heat_metrics=table_heat_metrics,
                inference_config=inference_config,
                source_path=source_path,
                plan_level_metrics=plan_level_metrics,
            )
            for child in normal_children
        ]

        subplan_cost = self._predict_subplans(
            node=node,
            subplan_nodes=subplan_children,
            query_text=query_text,
            config=config,
            table_heat_metrics=table_heat_metrics,
            inference_config=inference_config,
            source_path=source_path,
            plan_level_metrics=plan_level_metrics,
        )

        local_runtime_cost = self._predict_local_cost(
            registry=self.runtime_registry,
            node=node,
            query_text=query_text,
            config=config,
            table_heat_metrics=table_heat_metrics,
            inference_config=inference_config,
            source_path=source_path,
            plan_level_metrics=plan_level_metrics,
        )
        local_startup_cost = 0.0
        if self.startup_registry is not None:
            local_startup_cost = self._predict_local_cost(
                registry=self.startup_registry,
                node=node,
                query_text=query_text,
                config=config,
                table_heat_metrics=table_heat_metrics,
                inference_config=inference_config,
                source_path=source_path,
                plan_level_metrics=plan_level_metrics,
            )

        plan_cost = self.aggregator.aggregate(
            node=node,
            local_startup_cost=local_startup_cost,
            local_runtime_cost=local_runtime_cost,
            child_costs=child_costs,
            subplan_cost=subplan_cost,
        )
        node["Startup Predict"] = plan_cost.startup_cost
        node["Total Predict"] = plan_cost.total_cost
        return plan_cost

    def _predict_subplans(
        self,
        node: dict,
        subplan_nodes: list[dict],
        query_text: str | None,
        config: dict,
        table_heat_metrics: dict,
        inference_config: InferenceConfig,
        source_path: str | None,
        plan_level_metrics: dict | None,
    ) -> PlanCost:
        startup_cost = 0.0
        total_cost = 0.0
        parent_loops = float(node.get("Actual Loops", 0.0) or 0.0)

        for subplan_node in subplan_nodes:
            subplan_cost = self._predict_node(
                node=subplan_node,
                query_text=query_text,
                config=config,
                table_heat_metrics=table_heat_metrics,
                inference_config=inference_config,
                source_path=source_path,
                plan_level_metrics=plan_level_metrics,
            )
            ratio = 0.0
            if parent_loops > 0.0:
                ratio = float(subplan_node.get("Actual Loops", 0.0) or 0.0) / parent_loops
            startup_cost += subplan_cost.startup_cost * ratio
            total_cost += subplan_cost.total_cost * ratio

        return PlanCost(startup_cost=startup_cost, total_cost=total_cost)

    def _predict_local_cost(
        self,
        registry: OperatorInferencerRegistry,
        node: dict,
        query_text: str | None,
        config: dict,
        table_heat_metrics: dict,
        inference_config: InferenceConfig,
        source_path: str | None,
        plan_level_metrics: dict | None,
    ) -> float:
        context = self.extractor.extract_operator(
            node=node,
            query_text=query_text,
            config=config,
            table_heat_metrics=table_heat_metrics,
            source_path=source_path,
            plan_level_metrics=plan_level_metrics,
        )
        registry_prediction = registry.predict(
            context=context,
            inference_config=inference_config,
            strict=False,
        )
        if registry_prediction is None:
            return 0.0
        return registry_prediction.prediction.predicted_cost
