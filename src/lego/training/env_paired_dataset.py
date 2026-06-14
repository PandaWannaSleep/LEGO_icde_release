"""Environment-paired operator dataset for multi-objective pre-training.

This module supports the ``MultiObjectiveTrainer`` data pipeline:

  * **Paired mode** — each item is an :class:`EnvPair` of two observations
    belonging to the same positive anchor. Files may identify anchors by plan
    position or by a structured positive anchor key. Used by the contrastive
    head.
  * **Single mode** — each item is an :class:`OperatorTrainingExample`,
    structurally identical to legacy :class:`OperatorDataset` items. Used by
    the cost (LogMAE) head, and may also draw from anchored views to widen
    the per-op-type pool.

Both consumption modes are produced from the same on-disk JSONL files:

  * ``pairs.jsonl`` — either one plan-position anchor per line with a
    ``views`` dict, or streamed ``record_kind="positive_anchor_view"`` lines
    that share an ``anchor_id``. Streamed records permit an anchor
    to contain multiple observations from a runtime condition without writing
    a single massive JSON object. Anchors with fewer than two valid views
    cannot form a pair and are dropped from ``anchors``.
  * ``singles.jsonl`` — one single-instance record per line.

The dataset stores already-built ``(node_values, initial_adjacency, target)``
tensors in memory (computed once at load time). Random pair selection inside
an anchor is done on every ``__getitem__`` call, so the contrastive trainer
sees fresh ``(view_i, view_j)`` draws across epochs.

The optional sibling fields the extractor would normally pull from a
``planinfo`` envelope (``query``, ``config``, ``table_heat_metrics``,
plan-level system metrics) may be carried alongside ``view`` / ``views``;
they are forwarded to ``OperatorContextExtractor.extract_operator`` if
present. Their absence is tolerated — table heat falls back to zeros and
config defaults to empty, mirroring legacy behaviour.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lego.cag.builder import OperatorCAGBuilder
from lego.cag.template import CAGTemplate
from lego.data.operator_context_extractor import (
    OperatorContextExtractor,
    _unwrap_system_metrics,
)
from .collate import collate_operator_examples  # re-exported for trainer convenience
from .dataset import OperatorTrainingExample
from .metrics import summarize_numeric_values
from .targets import LocalTargetBuilder

logger = logging.getLogger(__name__)


__all__ = [
    "EnvPair",
    "EnvPairedOperatorDataset",
    "collate_env_pairs",
    "collate_operator_examples",
]


PAIR_GROUP_WEIGHT_POLICIES = (
    "uniform",
    "sqrt_cross_pairs",
    "sqrt_all_pairs",
)


# --------------------------------------------------------------------------- #
# Dataclass shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnvPair:
    """A pair of views sharing a positive anchor.

    Two sampled observations may originate from different queries or plan
    locations. The contrastive head treats them as a positive pair.

    ``view_a`` and ``view_b`` re-use :class:`OperatorTrainingExample` so the
    same tensor layout (``node_values``, ``initial_adjacency``, ``target``)
    applies to both single and paired training.
    """

    operator_type: str
    anchor_id: str
    view_a: OperatorTrainingExample
    view_b: OperatorTrainingExample


# --------------------------------------------------------------------------- #
# Internal anchor representation
# --------------------------------------------------------------------------- #


@dataclass
class _Anchor:
    anchor_id: str
    operator_type: str
    # view identifier -> built example (node values + adjacency + target).
    # View identifiers usually correspond to runtime conditions.
    views: dict[str, OperatorTrainingExample]
    # view identifier -> runtime condition label. Legacy files use the same
    # label for both the key and the condition; streamed records may contain
    # multiple views from the same condition.
    view_levels: dict[str, str]
    view_ids: list[str] = field(init=False)
    views_by_level: dict[str, list[str]] = field(init=False)
    possible_pair_count: int = field(init=False)
    cross_condition_pair_count: int = field(init=False)
    cross_condition_level_pairs: list[tuple[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.view_ids = sorted(self.views.keys())
        by_level: dict[str, list[str]] = defaultdict(list)
        for view_id in self.view_ids:
            by_level[self.view_levels.get(view_id, view_id)].append(view_id)
        self.views_by_level = dict(by_level)
        count = len(self.views)
        self.possible_pair_count = count * (count - 1) // 2
        counts = [len(view_ids) for view_ids in self.views_by_level.values()]
        self.cross_condition_pair_count = sum(
            left * right
            for idx, left in enumerate(counts)
            for right in counts[idx + 1 :]
        )
        levels = sorted(level for level, view_ids in self.views_by_level.items() if view_ids)
        self.cross_condition_level_pairs = [
            (left, right)
            for idx, left in enumerate(levels)
            for right in levels[idx + 1 :]
        ]

    def has_cross_condition_pair(self) -> bool:
        return self.cross_condition_pair_count > 0


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class EnvPairedOperatorDataset(Dataset):
    """Per-op-type dataset with paired and single consumption modes.

    Construct via :py:meth:`from_files`. The trainer instantiates two
    instances (one ``mode="paired"``, one ``mode="single"``) over the same
    underlying files when it wants to drive both heads from one source of
    truth.

    Attributes
    ----------
    operator_type:
        The operator type this dataset is filtered to (e.g. ``"Hash Join"``).
    task_name:
        Cost-head task label (e.g. ``"runtime_cost"``). Stored on every built
        example for loss routing.
    mode:
        Either ``"paired"`` or ``"single"`` — selects which underlying list
        ``__getitem__`` walks.
    anchors:
        List of anchors that retained ``>= 2`` valid views (paired-eligible).
    singles:
        List of single-instance examples (drawn from ``singles.jsonl`` plus,
        optionally, every retained anchor view). The trainer's cost-head
        DataLoader uses this list.
    dropped_singletons:
        Number of anchors that loaded but had ``< 2`` valid views and were
        therefore dropped from ``anchors``. Useful telemetry for the data
        collection pipeline.
    skipped_by_file:
        Per-file count of records that failed to build (JSON decode error,
        extractor failure, missing target, etc.). Mirrors legacy
        :class:`OperatorDataset` naming.
    target_summary:
        Numeric summary of every cost target loaded into ``singles``. Used by
        the trainer's diagnostics block.
    per_op_type_counts:
        Maps ``operator_type → count`` for *this dataset's mode*. Since we
        filter by op-type at construction the dict has at most one key, but
        the API matches what the trainer needs to build its cross-dataset weighted
        sampler.
    """

    PAIRED = "paired"
    SINGLE = "single"
    _MODES = frozenset({PAIRED, SINGLE})

    def __init__(
        self,
        operator_type: str,
        task_name: str,
        anchors: list[_Anchor],
        singles: list[OperatorTrainingExample],
        mode: str = PAIRED,
        skipped_by_file: dict[str, int] | None = None,
        dropped_singletons: int = 0,
        paired_epoch_size: int = 0,
        pair_group_weight: str = "uniform",
        pair_group_weight_cap: float = 0.0,
        positive_cross_condition_only: bool = False,
        rng: random.Random | None = None,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {sorted(self._MODES)}, got {mode!r}")
        if paired_epoch_size < 0:
            raise ValueError(f"paired_epoch_size must be >= 0, got {paired_epoch_size}")
        if pair_group_weight not in PAIR_GROUP_WEIGHT_POLICIES:
            raise ValueError(
                f"pair_group_weight must be one of {PAIR_GROUP_WEIGHT_POLICIES}, "
                f"got {pair_group_weight!r}"
            )
        if pair_group_weight_cap < 0:
            raise ValueError(
                f"pair_group_weight_cap must be >= 0, got {pair_group_weight_cap}"
            )
        self.operator_type = operator_type
        self.task_name = task_name
        original_anchor_count = len(anchors)
        if positive_cross_condition_only:
            anchors = [anchor for anchor in anchors if anchor.has_cross_condition_pair()]
        self.anchors = anchors
        self.singles = singles
        # ``__init__`` already validated ``mode`` above; assign to the backing
        # field directly to skip the setter's redundant re-validation.
        self._mode = mode
        self.skipped_by_file = skipped_by_file or {}
        self.dropped_singletons = dropped_singletons
        self.paired_epoch_size = paired_epoch_size
        self.pair_group_weight = pair_group_weight
        self.pair_group_weight_cap = pair_group_weight_cap
        self.positive_cross_condition_only = positive_cross_condition_only
        # A non-deterministic per-instance RNG so DataLoader workers and
        # repeated epochs see fresh pair draws by default; tests pass a
        # seeded ``random.Random`` to assert determinism.
        self._rng = rng if rng is not None else random.Random()
        self._anchor_cum_weights = self._build_anchor_cum_weights()
        self._cross_condition_dropped = original_anchor_count - len(anchors)

        target_values = [example.target for example in singles]
        self.target_summary = summarize_numeric_values(target_values)

    # ------------------------------------------------------------------ #
    # PyTorch Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        if self.mode == self.PAIRED:
            if self.paired_epoch_size > 0:
                return self.paired_epoch_size
            return len(self.anchors)
        return len(self.singles)

    def __getitem__(self, index: int) -> EnvPair | OperatorTrainingExample:
        if self.mode == self.PAIRED:
            anchor = self._sample_anchor(index)
            view_a_id, view_b_id = self._sample_view_pair(anchor)
            return EnvPair(
                operator_type=anchor.operator_type,
                anchor_id=anchor.anchor_id,
                view_a=anchor.views[view_a_id],
                view_b=anchor.views[view_b_id],
            )
        return self.singles[index]

    def _sample_anchor(self, index: int) -> _Anchor:
        if not self.anchors:
            raise RuntimeError("paired-mode dataset has no eligible anchors")
        if self.paired_epoch_size <= 0:
            return self.anchors[index]
        if not self._anchor_cum_weights:
            return self.anchors[self._rng.randrange(len(self.anchors))]
        total = self._anchor_cum_weights[-1]
        threshold = self._rng.random() * total
        anchor_index = bisect.bisect_right(self._anchor_cum_weights, threshold)
        if anchor_index >= len(self.anchors):
            anchor_index = len(self.anchors) - 1
        return self.anchors[anchor_index]

    def _sample_view_pair(self, anchor: _Anchor) -> tuple[str, str]:
        if self.positive_cross_condition_only:
            by_level = anchor.views_by_level
            if anchor.cross_condition_level_pairs:
                left, right = self._rng.choice(anchor.cross_condition_level_pairs)
                return self._rng.choice(by_level[left]), self._rng.choice(by_level[right])

        view_ids = anchor.view_ids
        if len(view_ids) < 2:
            # Should not happen — anchors with <2 views are dropped at load.
            raise RuntimeError(
                f"anchor {anchor.anchor_id!r} has <2 views in paired-mode dataset"
            )
        view_a_id, view_b_id = self._rng.sample(view_ids, 2)
        return view_a_id, view_b_id

    def _build_anchor_cum_weights(self) -> list[float]:
        cumulative: list[float] = []
        running = 0.0
        for anchor in self.anchors:
            if self.pair_group_weight == "sqrt_cross_pairs":
                base = math.sqrt(max(anchor.cross_condition_pair_count, 1))
            elif self.pair_group_weight == "sqrt_all_pairs":
                base = math.sqrt(max(anchor.possible_pair_count, 1))
            else:
                base = 1.0
            if self.pair_group_weight_cap > 0:
                base = min(base, self.pair_group_weight_cap)
            running += max(base, 0.0)
            cumulative.append(running)
        return cumulative

    # ------------------------------------------------------------------ #
    # Trainer-facing properties
    # ------------------------------------------------------------------ #

    @property
    def mode(self) -> str:
        """Current consumption mode (``"paired"`` or ``"single"``)."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in self._MODES:
            raise ValueError(
                f"Invalid mode {value!r}; expected one of {sorted(self._MODES)}"
            )
        self._mode = value

    @property
    def per_op_type_counts(self) -> dict[str, int]:
        """Count of items in this dataset's current mode, keyed by op-type.

        The trainer can fold these dicts together across per-op-type datasets
        to build a cross-dataset ``WeightedRandomSampler``. Within one op-type
        dataset uniform sampling is fine, so this returns a single-key dict.
        """
        return {self.operator_type: len(self)}

    @property
    def pair_sampling_summary(self) -> dict[str, Any]:
        """Telemetry describing paired-mode dynamic sampling."""
        cross_pairs = sum(anchor.cross_condition_pair_count for anchor in self.anchors)
        all_pairs = sum(anchor.possible_pair_count for anchor in self.anchors)
        return {
            "operator_type": self.operator_type,
            "loaded_anchor_groups": len(self.anchors),
            "paired_epoch_size": self.paired_epoch_size or len(self.anchors),
            "pair_group_weight": self.pair_group_weight,
            "pair_group_weight_cap": self.pair_group_weight_cap,
            "positive_cross_condition_only": self.positive_cross_condition_only,
            "cross_condition_filtered_anchor_groups": self._cross_condition_dropped,
            "candidate_positive_pairs": all_pairs,
            "candidate_cross_condition_pairs": cross_pairs,
        }

    # ------------------------------------------------------------------ #
    # Loader
    # ------------------------------------------------------------------ #

    @classmethod
    def from_files(
        cls,
        *,
        operator_type: str,
        task_name: str,
        template: CAGTemplate,
        extractor: OperatorContextExtractor,
        target_builder: LocalTargetBuilder,
        pairs_files: Sequence[str | Path] = (),
        singles_files: Sequence[str | Path] = (),
        limit_pairs: int = 0,
        limit_singles: int = 0,
        skip_invalid: bool = False,
        include_anchor_views_in_singles: bool = True,
        paired_epoch_size: int = 0,
        pair_group_weight: str = "uniform",
        pair_group_weight_cap: float = 0.0,
        positive_cross_condition_only: bool = False,
        mode: str = PAIRED,
        strict: bool = False,
        rng_seed: int | None = None,
    ) -> "EnvPairedOperatorDataset":
        """Construct from on-disk JSONL.

        Parameters
        ----------
        operator_type:
            Filter — only records whose ``operator_type`` matches are kept.
            LEGO trains one encoder per op-type, so the dataset is per-type.
        task_name:
            Target task name for :class:`LocalTargetBuilder` (e.g.
            ``"runtime_cost"``). Records whose target builds to ``None``
            (unsupported op/task pair, malformed plan node) are skipped.
        template:
            CAG template for ``operator_type``. Drives node ordering,
            categorical encoders, normalization stats, and initial adjacency.
        extractor:
            :class:`OperatorContextExtractor` re-used to turn a view's
            plan-node JSON into an :class:`OperatorContext`.
        target_builder:
            :class:`LocalTargetBuilder` used to derive the cost label.
        pairs_files, singles_files:
            Lists of JSONL paths. Either may be empty — for example, a
            paired-only dataset would pass only ``pairs_files``.
        limit_pairs, limit_singles:
            ``0`` disables the limit; otherwise stops loading after that many
            anchors / single records have been *built* (post-filter). Note:
            ``limit_singles`` is checked against the total ``singles`` list
            length, which (when ``include_anchor_views_in_singles=True``)
            already includes anchor views inflated from the paired pool.
            Inflated anchor views are themselves capped indirectly by
            ``limit_pairs`` rather than ``limit_singles``: they are appended
            during the pairs phase and contribute to the running ``singles``
            count, but the singles-loading loop will simply break out
            immediately if the inflation pool alone has already met or
            exceeded ``limit_singles``. Practically this means callers who
            want a known number of records from ``singles_files`` should
            either disable inflation (``include_anchor_views_in_singles=
            False``) or set ``limit_singles`` to the *combined* desired
            total.
        skip_invalid:
            If ``True``, JSON decode errors and extractor/target failures are
            counted in ``skipped_by_file`` and the record is dropped. If
            ``False`` (default), the first invalid record raises.
        include_anchor_views_in_singles:
            If ``True`` (default), every successfully built anchor view is
            *also* appended to ``singles``. The cost head can then train on
            the combined paired + unpaired pool. Set to ``False`` if you want
            strict separation.

            When ``True``, the same anchor view appears in both the paired
            (contrastive head) and single (cost head) pools.
        paired_epoch_size:
            Paired-mode epoch budget. ``0`` preserves legacy behaviour:
            one sampled pair per loaded anchor per epoch. A positive value
            makes ``__len__`` return this fixed budget and samples anchors
            dynamically on every ``__getitem__`` call, which keeps large
            positive-anchor groups tractable without materialising every
            candidate pair.
        pair_group_weight:
            Anchor-group weighting for dynamic paired sampling. ``uniform``
            samples groups equally, while ``sqrt_cross_pairs`` and
            ``sqrt_all_pairs`` compress large groups by the square root of
            their candidate pair counts.
        pair_group_weight_cap:
            Optional upper bound applied after the group-weight transform.
            ``0`` disables capping.
        positive_cross_condition_only:
            If ``True``, paired samples draw two views from different
            runtime conditions and anchors with no cross-condition pair are
            filtered out. This matches the existing plan-position protocol
            more closely for fair ablations.
        mode:
            Initial mode for the returned dataset (``"paired"`` or
            ``"single"``). Mode can be re-assigned post-hoc by the trainer
            (e.g. ``dataset.mode = "single"``); the setter validates the
            new value against ``_MODES`` and raises ``ValueError`` on an
            unknown mode.
        strict:
            Forwarded to :class:`OperatorCAGBuilder.build` — if ``True``,
            unknown categorical values raise; if ``False``, they fall back
            to ``0``.
        rng_seed:
            If given, seeds the per-instance ``random.Random`` so paired
            ``__getitem__`` draws are reproducible.
        """
        cag_builder = OperatorCAGBuilder()
        skipped_by_file: dict[str, int] = defaultdict(int)
        anchors: list[_Anchor] = []
        singles: list[OperatorTrainingExample] = []
        dropped_singletons = 0
        streamed_anchor_views: dict[str, dict[str, OperatorTrainingExample]] = defaultdict(dict)
        streamed_anchor_levels: dict[str, dict[str, str]] = defaultdict(dict)
        streamed_anchor_types: dict[str, str] = {}

        # ----------------- pairs ------------------------------------- #
        for path in pairs_files:
            path = Path(path)
            anchors_built_for_path = 0
            for line_number, raw_line in cls._iter_lines(path, skip_invalid, skipped_by_file):
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    if not skip_invalid:
                        raise ValueError(
                            f"{path}:{line_number}: JSON decode error: {exc.msg}"
                        ) from exc
                    skipped_by_file[str(path)] += 1
                    continue

                op_type = record.get("operator_type")
                if op_type != operator_type:
                    continue

                anchor_id = record.get("anchor_id") or f"{path.name}:{line_number}"
                if record.get("record_kind") == "positive_anchor_view":
                    view_payload = record.get("view")
                    example = cls._build_example_from_view(
                        view_payload=view_payload,
                        operator_type=operator_type,
                        task_name=task_name,
                        template=template,
                        extractor=extractor,
                        target_builder=target_builder,
                        cag_builder=cag_builder,
                        source_path=f"{path}:{line_number}#view",
                        anchor_query=record.get("query"),
                        anchor_config=record.get("config") or {},
                        anchor_heat=record.get("table_heat_metrics") or {},
                        strict=strict,
                    )
                    if example is None:
                        skipped_by_file[str(path)] += 1
                        continue
                    view_id = str(record.get("view_id") or f"{path.name}:{line_number}")
                    streamed_anchor_views[anchor_id][view_id] = example
                    streamed_anchor_levels[anchor_id][view_id] = str(
                        record.get("concurrency_level") or view_id.split(":", 1)[0]
                    )
                    streamed_anchor_types[anchor_id] = operator_type
                    continue

                views_payload = record.get("views") or {}
                if not isinstance(views_payload, dict):
                    if not skip_invalid:
                        raise ValueError(
                            f"{path}:{line_number}: 'views' must be a dict, got {type(views_payload).__name__}"
                        )
                    skipped_by_file[str(path)] += 1
                    continue

                # Sibling context fields. Each view may also override these,
                # but anchor-level defaults are the common case.
                anchor_query = record.get("query")
                anchor_config = record.get("config") or {}
                anchor_heat = record.get("table_heat_metrics") or {}

                built_views: dict[str, OperatorTrainingExample] = {}
                view_examples_for_singles: list[OperatorTrainingExample] = []
                for level, view_payload in views_payload.items():
                    example = cls._build_example_from_view(
                        view_payload=view_payload,
                        operator_type=operator_type,
                        task_name=task_name,
                        template=template,
                        extractor=extractor,
                        target_builder=target_builder,
                        cag_builder=cag_builder,
                        source_path=f"{path}:{line_number}#views.{level}",
                        anchor_query=anchor_query,
                        anchor_config=anchor_config,
                        anchor_heat=anchor_heat,
                        strict=strict,
                    )
                    if example is None:
                        skipped_by_file[str(path)] += 1
                        continue
                    built_views[level] = example
                    view_examples_for_singles.append(example)

                if len(built_views) < 2:
                    dropped_singletons += 1
                    if include_anchor_views_in_singles:
                        # A solo view is still a usable single-instance datum.
                        singles.extend(view_examples_for_singles)
                    continue

                anchors.append(
                    _Anchor(
                        anchor_id=anchor_id,
                        operator_type=operator_type,
                        views=built_views,
                        view_levels={str(level): str(level) for level in built_views},
                    )
                )
                anchors_built_for_path += 1
                if include_anchor_views_in_singles:
                    singles.extend(view_examples_for_singles)

                if limit_pairs and len(anchors) >= limit_pairs:
                    break
            # outer for-line ends; check pair limit across files
            if limit_pairs and len(anchors) >= limit_pairs:
                break

        # Paper-policy pair files are streamed one observation per line. Form
        # their in-memory anchors after all matching observations are built.
        if not limit_pairs or len(anchors) < limit_pairs:
            for anchor_id in sorted(streamed_anchor_views):
                built_views = streamed_anchor_views[anchor_id]
                if len(built_views) < 2:
                    dropped_singletons += 1
                    if include_anchor_views_in_singles:
                        singles.extend(built_views.values())
                    continue
                anchors.append(
                    _Anchor(
                        anchor_id=anchor_id,
                        operator_type=streamed_anchor_types[anchor_id],
                        views=built_views,
                        view_levels=streamed_anchor_levels[anchor_id],
                    )
                )
                if include_anchor_views_in_singles:
                    singles.extend(built_views.values())
                if limit_pairs and len(anchors) >= limit_pairs:
                    break

        # ----------------- singles ----------------------------------- #
        for path in singles_files:
            path = Path(path)
            for line_number, raw_line in cls._iter_lines(path, skip_invalid, skipped_by_file):
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    if not skip_invalid:
                        raise ValueError(
                            f"{path}:{line_number}: JSON decode error: {exc.msg}"
                        ) from exc
                    skipped_by_file[str(path)] += 1
                    continue

                op_type = record.get("operator_type")
                if op_type != operator_type:
                    continue

                view_payload = record.get("view")
                if view_payload is None:
                    if not skip_invalid:
                        raise ValueError(
                            f"{path}:{line_number}: missing 'view' field"
                        )
                    skipped_by_file[str(path)] += 1
                    continue

                anchor_query = record.get("query")
                anchor_config = record.get("config") or {}
                anchor_heat = record.get("table_heat_metrics") or {}

                example = cls._build_example_from_view(
                    view_payload=view_payload,
                    operator_type=operator_type,
                    task_name=task_name,
                    template=template,
                    extractor=extractor,
                    target_builder=target_builder,
                    cag_builder=cag_builder,
                    source_path=f"{path}:{line_number}",
                    anchor_query=anchor_query,
                    anchor_config=anchor_config,
                    anchor_heat=anchor_heat,
                    strict=strict,
                )
                if example is None:
                    skipped_by_file[str(path)] += 1
                    continue

                singles.append(example)
                if limit_singles and len(singles) >= limit_singles:
                    break
            if limit_singles and len(singles) >= limit_singles:
                break

        if dropped_singletons:
            logger.info(
                "EnvPairedOperatorDataset[%s]: dropped %d singleton anchor(s) "
                "(fewer than 2 valid views).",
                operator_type,
                dropped_singletons,
            )

        rng = random.Random(rng_seed) if rng_seed is not None else None
        return cls(
            operator_type=operator_type,
            task_name=task_name,
            anchors=anchors,
            singles=singles,
            mode=mode,
            skipped_by_file=dict(skipped_by_file),
            dropped_singletons=dropped_singletons,
            rng=rng,
            paired_epoch_size=paired_epoch_size,
            pair_group_weight=pair_group_weight,
            pair_group_weight_cap=pair_group_weight_cap,
            positive_cross_condition_only=positive_cross_condition_only,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iter_lines(
        path: Path,
        skip_invalid: bool,
        skipped_by_file: dict[str, int],
    ) -> Iterable[tuple[int, str]]:
        """Yield ``(line_number, line)`` pairs for non-empty lines in *path*."""
        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError:
            if skip_invalid:
                skipped_by_file[str(path)] += 1
                return
            raise
        try:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                yield line_number, stripped
        finally:
            handle.close()

    @staticmethod
    def _build_example_from_view(
        *,
        view_payload: Any,
        operator_type: str,
        task_name: str,
        template: CAGTemplate,
        extractor: OperatorContextExtractor,
        target_builder: LocalTargetBuilder,
        cag_builder: OperatorCAGBuilder,
        source_path: str,
        anchor_query: str | None,
        anchor_config: dict[str, Any],
        anchor_heat: dict[str, Any],
        strict: bool,
    ) -> OperatorTrainingExample | None:
        """Turn one view's plan-node JSON into a training example, or ``None`` on failure."""
        if not isinstance(view_payload, dict):
            return None

        # The view *may* be either a bare plan-node dict OR a wrapper carrying
        # ``{"node": ..., "query": ..., "config": ..., "plan_metrics": ...}``.
        # Tolerate both shapes to keep upstream pair-assembler wiggle room.
        if "Node Type" in view_payload:
            node = view_payload
            view_query = anchor_query
            view_config = anchor_config
            view_heat = anchor_heat
            plan_level_metrics = _unwrap_system_metrics(view_payload)
        else:
            node = view_payload.get("node") or view_payload.get("plan_node")
            if not isinstance(node, dict) or "Node Type" not in node:
                return None
            view_query = view_payload.get("query", anchor_query)
            view_config = view_payload.get("config", anchor_config) or {}
            view_heat = view_payload.get("table_heat_metrics", anchor_heat) or {}
            plan_level_metrics = (
                _unwrap_system_metrics(view_payload)
                or _unwrap_system_metrics(view_payload.get("planinfo"))
            )

        if node.get("Node Type") != operator_type:
            return None

        try:
            context = extractor.extract_operator(
                node,
                query_text=view_query,
                config=view_config,
                table_heat_metrics=view_heat,
                source_path=source_path,
                plan_level_metrics=plan_level_metrics,
            )
        except (KeyError, ValueError, TypeError):
            return None

        try:
            target = target_builder.build_target(context=context, task_name=task_name)
        except (KeyError, ValueError, TypeError):
            return None
        if target is None:
            return None

        try:
            cag = cag_builder.build(context=context, template=template, strict=strict)
        except (KeyError, ValueError) as exc:
            # Strict mode raises through; lenient just drops.
            if strict:
                raise
            logger.debug("CAG build failure for %s: %s", source_path, exc)
            return None

        return OperatorTrainingExample(
            operator_type=operator_type,
            task_name=task_name,
            node_values=cag.node_values.astype(np.float32, copy=True),
            initial_adjacency=cag.initial_adjacency.astype(np.float32, copy=True),
            target=float(target),
            context=context,
        )


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #


def collate_env_pairs(batch: list[EnvPair]) -> dict[str, object]:
    """Stack a batch of :class:`EnvPair` into parallel ``view_a`` / ``view_b`` tensors.

    Output keys:

    - ``view_a_node_values``      : ``(B, N)`` float32
    - ``view_a_initial_adjacency``: ``(B, N, N)`` float32
    - ``view_a_targets``          : ``(B,)`` float32
    - ``view_b_node_values``      : ``(B, N)`` float32
    - ``view_b_initial_adjacency``: ``(B, N, N)`` float32
    - ``view_b_targets``          : ``(B,)`` float32
    - ``anchor_ids``              : list[str] of length ``B``
    - ``view_a_contexts``         : list[OperatorContext]
    - ``view_b_contexts``         : list[OperatorContext]

    Mirrors :func:`collate_operator_examples` so the trainer can feed a
    batched dict into the same encoder/head modules.
    """
    if not batch:
        raise ValueError("collate_env_pairs received an empty batch")

    a_node = np.stack([pair.view_a.node_values for pair in batch], axis=0)
    a_adj = np.stack([pair.view_a.initial_adjacency for pair in batch], axis=0)
    b_node = np.stack([pair.view_b.node_values for pair in batch], axis=0)
    b_adj = np.stack([pair.view_b.initial_adjacency for pair in batch], axis=0)

    return {
        "view_a_node_values": torch.from_numpy(a_node).to(torch.float32),
        "view_a_initial_adjacency": torch.from_numpy(a_adj).to(torch.float32),
        "view_a_targets": torch.tensor(
            [pair.view_a.target for pair in batch], dtype=torch.float32
        ),
        "view_b_node_values": torch.from_numpy(b_node).to(torch.float32),
        "view_b_initial_adjacency": torch.from_numpy(b_adj).to(torch.float32),
        "view_b_targets": torch.tensor(
            [pair.view_b.target for pair in batch], dtype=torch.float32
        ),
        "anchor_ids": [pair.anchor_id for pair in batch],
        "view_a_contexts": [pair.view_a.context for pair in batch],
        "view_b_contexts": [pair.view_b.context for pair in batch],
    }
