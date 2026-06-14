"""Operator taxonomy used by the LEGO artifact."""

LEAF_OPERATORS = (
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "CTE Scan",
)

JOIN_OPERATORS = (
    "Hash Join",
    "Merge Join",
    "Nested Loop",
)

AGGREGATE_OPERATORS = ("Aggregate",)
SORT_OPERATORS = ("Sort",)
OTHER_OPERATORS = ("Hash", "Materialize", "Subquery Scan")

SUPPORTED_OPERATORS = (
    LEAF_OPERATORS
    + JOIN_OPERATORS
    + AGGREGATE_OPERATORS
    + SORT_OPERATORS
    + OTHER_OPERATORS
)

PREDICTABLE_OPERATORS = (
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Sort",
    "Hash Join",
    "Nested Loop",
    "Merge Join",
    "Aggregate",
)

SUPPORTED_COST_TYPES = ("startup_cost", "runtime_cost")

DEFAULT_PARENT_OPERATOR = "none"
DEFAULT_CHILD_OPERATOR = "none"
DEFAULT_STRATEGY = "none"

OPERATOR_CATEGORICAL_FEATURES = {
    "ParentOp",
    "LeftOp",
    "RightOp",
    "Strategy",
}

