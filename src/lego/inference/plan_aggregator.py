from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanCost:
    startup_cost: float
    total_cost: float


class PostgresPlanAggregator:
    """
    Plan-level cost aggregation rules for LEGO inference.

    The local operator predictors estimate operator-local startup/runtime terms.
    This class is responsible for composing those local terms with child plan
    costs following PostgreSQL operator semantics.
    """

    leaf_operators = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "CTE Scan"}
    join_operators = {"Hash Join", "Merge Join", "Nested Loop"}

    def aggregate(
        self,
        node: dict,
        local_startup_cost: float,
        local_runtime_cost: float,
        child_costs: list[PlanCost],
        subplan_cost: PlanCost | None = None,
    ) -> PlanCost:
        operator = node["Node Type"]
        subplan_cost = subplan_cost or PlanCost(0.0, 0.0)

        if operator in self.leaf_operators:
            return PlanCost(
                startup_cost=local_startup_cost + subplan_cost.startup_cost,
                total_cost=local_runtime_cost + subplan_cost.total_cost,
            )

        if operator == "Hash":
            left = child_costs[0]
            return PlanCost(
                startup_cost=left.startup_cost + subplan_cost.startup_cost,
                total_cost=left.total_cost + subplan_cost.total_cost,
            )

        if operator in {"Append", "Merge Append", "Subquery Scan"}:
            startup_cost = sum(cost.startup_cost for cost in child_costs) + subplan_cost.startup_cost
            total_cost = sum(cost.total_cost for cost in child_costs) + subplan_cost.total_cost
            return PlanCost(startup_cost=startup_cost, total_cost=total_cost)

        if operator == "Materialize":
            left = child_costs[0]
            loops = float(node.get("Actual Loops", 1.0) or 1.0)
            return PlanCost(
                startup_cost=local_startup_cost + left.startup_cost / loops + subplan_cost.startup_cost,
                total_cost=local_runtime_cost + left.total_cost / loops + subplan_cost.total_cost,
            )

        if operator == "Nested Loop":
            left, right = child_costs
            startup_cost = local_startup_cost + left.startup_cost + right.startup_cost
            total_cost = local_runtime_cost + left.total_cost + right.total_cost
            return PlanCost(
                startup_cost=startup_cost + subplan_cost.startup_cost,
                total_cost=total_cost + subplan_cost.total_cost,
            )

        if operator == "Merge Join":
            left, right = child_costs
            startup_cost = local_startup_cost + left.startup_cost + right.startup_cost
            total_cost = local_runtime_cost + left.total_cost + right.total_cost
            return PlanCost(
                startup_cost=startup_cost + subplan_cost.startup_cost,
                total_cost=total_cost + subplan_cost.total_cost,
            )

        if operator == "Hash Join":
            left, right = child_costs
            startup_cost = local_startup_cost + right.total_cost
            if float(node.get("RightRows", 0.0) or 0.0) == 0.0 and node.get("LeftOp") == "Seq Scan":
                startup_cost = startup_cost + left.startup_cost
                total_cost = local_runtime_cost + startup_cost + right.total_cost
            else:
                total_cost = local_runtime_cost + startup_cost + left.total_cost
            return PlanCost(
                startup_cost=startup_cost + subplan_cost.startup_cost,
                total_cost=total_cost + subplan_cost.total_cost,
            )

        if operator == "Sort":
            left = child_costs[0]
            startup_cost = local_startup_cost + left.total_cost
            total_cost = local_runtime_cost + startup_cost
            return PlanCost(
                startup_cost=startup_cost + subplan_cost.startup_cost,
                total_cost=total_cost + subplan_cost.total_cost,
            )

        if operator == "Aggregate":
            left = child_costs[0]
            strategy = node.get("Strategy")
            if strategy == "Sorted":
                startup_cost = local_startup_cost + left.startup_cost
                total_cost = local_runtime_cost + left.total_cost
            else:
                startup_cost = local_startup_cost + left.total_cost
                total_cost = local_runtime_cost + startup_cost
            return PlanCost(
                startup_cost=startup_cost + subplan_cost.startup_cost,
                total_cost=total_cost + subplan_cost.total_cost,
            )

        if child_costs:
            left = child_costs[0]
            return PlanCost(
                startup_cost=local_startup_cost + left.startup_cost + subplan_cost.startup_cost,
                total_cost=local_runtime_cost + left.total_cost + subplan_cost.total_cost,
            )

        return PlanCost(
            startup_cost=local_startup_cost + subplan_cost.startup_cost,
            total_cost=local_runtime_cost + subplan_cost.total_cost,
        )
