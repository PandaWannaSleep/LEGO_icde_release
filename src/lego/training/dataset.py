from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from lego.cag.builder import OperatorCAGBuilder
from lego.cag.template import CAGTemplate
from lego.data.operator_context import OperatorContext
from lego.data.operator_context_extractor import OperatorContextExtractor
from lego.data.plan_loader import iter_plan_records_with_source
from .metrics import summarize_numeric_values
from .targets import LocalTargetBuilder


@dataclass(frozen=True)
class OperatorTrainingExample:
    operator_type: str
    task_name: str
    node_values: np.ndarray
    initial_adjacency: np.ndarray
    target: float
    context: OperatorContext


class OperatorDataset(Dataset[OperatorTrainingExample]):
    def __init__(
        self,
        examples: list[OperatorTrainingExample],
        skipped_by_file: dict[str, int] | None = None,
        target_summary: dict[str, float | int] | None = None,
    ):
        self.examples = examples
        self.skipped_by_file = skipped_by_file or {}
        self.target_summary = target_summary or summarize_numeric_values([])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> OperatorTrainingExample:
        return self.examples[index]

    @classmethod
    def from_plan_files(
        cls,
        plan_files: list[str] | list[Path],
        operator_type: str,
        task_name: str,
        extractor: OperatorContextExtractor,
        template: CAGTemplate,
        target_builder: LocalTargetBuilder,
        limit_plans: int = 0,
        strict: bool = False,
        skip_invalid_plans: bool = False,
    ) -> "OperatorDataset":
        cag_builder = OperatorCAGBuilder()
        examples: list[OperatorTrainingExample] = []
        skipped_by_file: dict[str, int] = defaultdict(int)

        def on_skip(path: Path, _line_number: int, _reason: str) -> None:
            skipped_by_file[str(path)] += 1

        for plan_index, (source_path, plan_record) in enumerate(
            iter_plan_records_with_source(
                list(plan_files),
                strict=not skip_invalid_plans,
                on_skip=on_skip if skip_invalid_plans else None,
            ),
            start=1,
        ):
            contexts = extractor.extract_plan(plan_record, source_path=str(source_path))
            for context in contexts:
                if context.operator_type != operator_type:
                    continue
                target = target_builder.build_target(context=context, task_name=task_name)
                if target is None:
                    continue
                cag = cag_builder.build(context=context, template=template, strict=strict)
                examples.append(
                    OperatorTrainingExample(
                        operator_type=operator_type,
                        task_name=task_name,
                        node_values=cag.node_values.astype(np.float32, copy=True),
                        initial_adjacency=cag.initial_adjacency.astype(np.float32, copy=True),
                        target=float(target),
                        context=context,
                    )
                )
            if limit_plans and plan_index >= limit_plans:
                break

        return cls(
            examples,
            skipped_by_file=dict(skipped_by_file),
            target_summary=summarize_numeric_values([example.target for example in examples]),
        )
