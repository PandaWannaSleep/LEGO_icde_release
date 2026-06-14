from __future__ import annotations

import copy
import json
import queue
import re
from typing import Any, Iterator


def iter_plan_nodes(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if "Plans" in node:
        for child in node["Plans"]:
            yield from iter_plan_nodes(child)
    yield node


def add_parent_info(plan_record: dict[str, Any]) -> None:
    plan_root = plan_record["Plan"]
    plan_root["parent"] = "none"

    nodes: queue.Queue[dict[str, Any]] = queue.Queue()
    nodes.put(plan_root)
    while not nodes.empty():
        node = nodes.get()
        parent_op = node["Node Type"]
        for child in node.get("Plans", []):
            child["parent"] = parent_op
            nodes.put(child)


def add_initplan_info(plan_record: dict[str, Any]) -> None:
    nodes = list(iter_plan_nodes(plan_record["Plan"]))
    init_plan: dict[str, dict[str, Any]] = {}
    cte_plan: dict[str, dict[str, Any]] = {}

    for node in nodes:
        if "Subplan Name" not in node:
            continue
        if node.get("Parent Relationship") == "InitPlan" and "InitPlan" in node["Subplan Name"]:
            key = re.findall(r"\$\d+", node["Subplan Name"])[0]
            init_plan[key] = node
        elif node.get("Parent Relationship") == "InitPlan" and "CTE" in node["Subplan Name"]:
            key = node["Subplan Name"].split(" ")[1]
            cte_plan[key] = node

    for node in nodes:
        if node.get("Actual Total Time", 0) != 0 and "Filter" in node:
            keys = list(init_plan.keys())
            outputs = ",".join(node.get("Output", []))
            for key in keys:
                if key not in f"{node['Filter']}{outputs}":
                    continue
                node.setdefault("InitPlan", [])
                duplicate = any(
                    child.get("Subplan Name") == init_plan[key]["Subplan Name"]
                    for child in node["InitPlan"]
                )
                if not duplicate:
                    node["InitPlan"].append(init_plan[key])
                    init_plan.pop(key)

        if node.get("Actual Total Time", 0) != 0 and node.get("Node Type") == "CTE Scan":
            cte_name = node.get("CTE Name")
            if cte_name not in cte_plan:
                continue
            node.setdefault("InitPlan", [])
            duplicate = any(
                child.get("Subplan Name", "").split(" ")[1] == cte_name
                for child in node["InitPlan"]
                if "Subplan Name" in child
            )
            if not duplicate:
                node["InitPlan"].append(cte_plan[cte_name])
                cte_plan.pop(cte_name)


def actual_rows_modify(plan_root: dict[str, Any]) -> None:
    def _walk(node: dict[str, Any], parent: dict[str, Any]) -> None:
        if node["Node Type"] in {"Index Scan", "Index Only Scan"} and node.get("Actual Loops", 0) > 1:
            node["Actual Rows"] = parent["Actual Rows"] / node["Actual Loops"]
        for child in node.get("Plans", []):
            _walk(child, node)

    for child in plan_root.get("Plans", []):
        _walk(child, plan_root)


def normalize_plan_record(record: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(record, str):
        parsed = json.loads(record.strip())
    else:
        parsed = copy.deepcopy(record)

    if not parsed.get("planinfo") or "Plan" not in parsed["planinfo"]:
        raise ValueError("Plan Info Error")

    add_parent_info(parsed["planinfo"])
    add_initplan_info(parsed["planinfo"])
    actual_rows_modify(parsed["planinfo"]["Plan"])

    parsed.setdefault("config", {})
    parsed.setdefault("table_heat_metrics", {})
    if "config" in parsed and "settings" in parsed:
        parsed["config"].update(parsed["settings"])
    elif "setting" in parsed:
        parsed["config"] = parsed["setting"]

    return parsed

