from __future__ import annotations

from dataclasses import dataclass

from lego.data.operator_context import OperatorContext


SUPPORTED_OPERATOR_TASKS: dict[str, set[str]] = {
    "Seq Scan": {"runtime_cost"},
    "Index Scan": {"runtime_cost"},
    "Index Only Scan": {"runtime_cost"},
    "Sort": {"startup_cost", "runtime_cost"},
    "Hash Join": {"startup_cost", "runtime_cost"},
    "Nested Loop": {"runtime_cost"},
    "Merge Join": {"runtime_cost"},
    "Aggregate": {"startup_cost", "runtime_cost"},
    "Gather": {"runtime_cost"},
    "Gather Merge": {"runtime_cost"},
    "Hash": {"runtime_cost"},
    "Limit": {"runtime_cost"},
    "CTE Scan": {"runtime_cost"},
}


@dataclass(frozen=True)
class TargetSpec:
    operator_type: str
    task_name: str


class LocalTargetBuilder:
    """
    Reconstruct the local operator targets used by the original LEGO artifact.

    These targets are not the raw plan-level times. They are the operator-local
    startup/runtime contributions that the scheduler later aggregates back into
    plan cost.
    """

    def supports(self, operator_type: str, task_name: str) -> bool:
        return task_name in SUPPORTED_OPERATOR_TASKS.get(operator_type, set())

    def build_target(self, context: OperatorContext, task_name: str) -> float | None:
        operator = context.operator_type
        if not self.supports(operator, task_name):
            return None

        node = context.metadata.raw_plan
        if node is None:
            return None

        features = context.feature_dict()
        info = _general_plan_info(node)
        actual_startup = context.labels.actual_startup_time
        actual_total = context.labels.actual_total_time
        loops = float(features.get("Loops", 0.0) or 0.0)

        y = 0.0
        if operator == "Sort" and task_name == "startup_cost":
            y = actual_startup - info["left_total_time"]
        elif operator == "Sort" and task_name == "runtime_cost":
            y = actual_total - actual_startup
        elif operator == "Hash Join" and task_name == "startup_cost":
            y = actual_startup - info["right_total_time"]
        elif operator == "Hash Join" and task_name == "runtime_cost":
            y = actual_total - actual_startup - info["left_total_time"]
        elif operator == "Seq Scan" and task_name == "runtime_cost":
            y = actual_total - info["initplan_cost_time"]
        elif operator == "Index Scan" and task_name == "runtime_cost":
            y = actual_total - info["initplan_cost_time"]
        elif operator == "Index Only Scan" and task_name == "runtime_cost":
            y = actual_total - info["initplan_cost_time"]
        elif operator == "Nested Loop" and task_name == "runtime_cost":
            left_rows = float(features.get("LeftRows", 0.0) or 0.0)
            right_rows = float(features.get("RightRows", 0.0) or 0.0)
            rows = float(features.get("Rows", 0.0) or 0.0)
            right_op = str(features.get("RightOp", ""))
            inner_unique = bool(features.get("InnerUnique", 0.0))

            base = info["left_total_time"] + info["right_startup_time"]
            if inner_unique:
                denominator = left_rows * right_rows
                inner_scan_frac = 0.0 if denominator == 0.0 else rows / denominator
            else:
                inner_scan_frac = 1.0

            if right_op not in {"Materialize", "Sort"}:
                base += info["right_startup_time"] * max(left_rows - 1.0, 0.0)
                base += left_rows * (info["right_total_time"] - info["right_startup_time"]) * inner_scan_frac
            y = actual_total - base
        elif operator == "Merge Join" and task_name == "runtime_cost":
            y = actual_total - (info["left_total_time"] + info["right_total_time"])
        elif operator == "Gather" and task_name == "runtime_cost":
            # Gather collects rows from parallel workers. Its local cost is the
            # coordination/collection overhead: wall-clock total minus the child's
            # cumulative parallel time (which PG already sums across workers).
            y = actual_total - info["left_total_time"]
        elif operator == "Gather Merge" and task_name == "runtime_cost":
            # Gather Merge has the same collection role as Gather, with extra
            # merge-ordering overhead.
            y = actual_total - info["left_total_time"]
        elif operator == "Hash" and task_name == "runtime_cost":
            # Hash is a pure build node: it scans its child and builds a hash table.
            # actual_startup ≈ actual_total (no output rows until probed by Hash Join).
            # Local cost = hashing overhead = startup minus child scan time.
            y = actual_startup - info["left_total_time"]
        elif operator == "Limit" and task_name == "runtime_cost":
            # Limit usually forwards a prefix of its child output; local work is
            # the residual overhead after the child has produced that prefix.
            y = actual_total - info["left_total_time"]
        elif operator == "CTE Scan" and task_name == "runtime_cost":
            # CTE materialization work appears as init/subplan time. The scan
            # node's local runtime is the read/iteration residual.
            y = actual_total - info["initplan_cost_time"]
        elif operator == "Aggregate" and task_name == "startup_cost":
            strategy = str(features.get("Strategy", ""))
            if strategy == "Sorted":
                y = actual_startup - info["left_startup_time"]
            else:
                y = actual_startup - info["left_total_time"]
        elif operator == "Aggregate" and task_name == "runtime_cost":
            strategy = str(features.get("Strategy", ""))
            y = actual_total - actual_startup
            if strategy == "Sorted":
                y = actual_total - info["left_total_time"]
        else:
            return None

        y = max(y, 0.0)
        if task_name == "runtime_cost":
            y = y * loops - info["subplan_cost_time"]
        else:
            y = y * loops - info["subplan_startup_time"] - info["initplan_cost_time"]
        return max(y, 0.0)


def _general_plan_info(node: dict) -> dict[str, float]:
    normal_children = [child for child in node.get("Plans", []) if "Subplan Name" not in child]
    subplan_children = [
        child
        for child in node.get("Plans", [])
        if "Subplan Name" in child and child.get("Parent Relationship") == "SubPlan"
    ]

    subplan_startup_time = sum(float(child.get("Actual Total Time", 0.0) or 0.0) for child in subplan_children)
    subplan_cost_time = sum(
        float(child.get("Actual Total Time", 0.0) or 0.0) * float(child.get("Actual Loops", 0.0) or 0.0)
        for child in subplan_children
    )
    initplan_cost_time = sum(
        float(child.get("Actual Total Time", 0.0) or 0.0) for child in node.get("InitPlan", [])
    )

    left = normal_children[0] if len(normal_children) >= 1 else {}
    right = normal_children[1] if len(normal_children) >= 2 else {}

    return {
        "left_startup_time": float(left.get("Actual Startup Time", 0.0) or 0.0),
        "left_total_time": float(left.get("Actual Total Time", 0.0) or 0.0),
        "right_startup_time": float(right.get("Actual Startup Time", 0.0) or 0.0),
        "right_total_time": float(right.get("Actual Total Time", 0.0) or 0.0),
        "subplan_startup_time": subplan_startup_time,
        "subplan_cost_time": subplan_cost_time,
        "initplan_cost_time": initplan_cost_time,
    }
