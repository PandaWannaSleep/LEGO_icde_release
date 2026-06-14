"""Joint contrastive, cost, and graph-regularized pretraining.

The trainer wires together:

* :class:`lego.model.typed_node_encoder.TypedNodeEncoder` for per-type
  categorical, log-scale, and z-score node embedding.
* :class:`lego.model.graph_learner.GraphLearner` for per-operator
  interaction weights.
* :class:`lego.model.graph_updater.GraphUpdater` for initial-graph blending
  and iterative refinement.
* :class:`lego.model.operator_encoder.OperatorEncoder` for message passing
  and graph-level readout into ``z``.
* Two heads consuming the same ``z``:

  - :class:`lego.model.cost_predictor.CostHead` →
    ``L_cost = LogMAE(h(z), c)``
  - :class:`lego.model.projection_head.ProjectionHead` →
    ``L_con = InfoNCE(g(z_a), g(z_b))`` over env-paired anchors.

* A graph-regularisation term on the post-graph-updater adjacency::

      L_reg = -gamma_conn / n  · 1ᵀ log(A 1)
              + gamma_sp / n^2 · ||A||_F^2

  (connectivity log-barrier + Frobenius sparsity), averaged across the batch.

The total loss is::

    L = alpha · L_con + beta · L_cost + gamma · L_reg

The trainer drives the iterative refinement loop ``T_max`` steps per
forward pass. ``L_reg`` is computed on the final refined adjacency.

Two DataLoaders feed the loop — one paired (contrastive) and one single
(cost). Each training step alternates between the two: a paired batch
contributes ``L_con + L_reg``; a single batch contributes ``L_cost +
L_reg``. Each batch produces an independent backward + optimizer step;
this is simpler than a single fused backward and keeps the two objectives'
gradients separate.

Dataset wiring note (``include_anchor_views_in_singles``)
---------------------------------------------------------

The :class:`lego.training.env_paired_dataset.EnvPairedOperatorDataset`
can inflate anchor views into the singles pool. That is convenient for small
data regimes but can let the same view train the cost head twice. To keep the
cost head's gradient signal attributable to unpaired records,
:meth:`MultiObjectiveTrainer.fit` accepts already-built datasets from the
caller. Callers can construct the two
:class:`EnvPairedOperatorDataset` instances themselves with
``include_anchor_views_in_singles=False`` so the contrastive and cost pools
are disjoint.

Ablation hooks (``pretraining_objective``)
------------------------------------------

The ``MultiObjectiveTrainerConfig.pretraining_objective`` field selects
one of four ablation modes used by the H.5 study:

* ``"no_pretrain"`` — :meth:`fit` returns immediately with empty history;
  the modules retain their initial random weights. Used as the
  random-init baseline.
* ``"contrastive_only"`` — only paired batches drive training; ``beta``
  is internally forced to ``0`` so the cost head sees no gradient.
* ``"cost_only"`` — only single batches drive training; ``alpha`` is
  internally forced to ``0`` so the projection head sees no gradient.
* ``"full"`` — both branches active; ``alpha`` and ``beta`` use their
  configured values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lego.model.cost_predictor import CostHead, CostHeadConfig
from lego.model.graph_learner import GraphLearner, GraphLearnerConfig
from lego.model.graph_updater import GraphUpdateConfig, GraphUpdater
from lego.model.operator_encoder import OperatorEncoder, OperatorEncoderConfig
from lego.model.projection_head import ProjectionHead, ProjectionHeadConfig
from lego.model.typed_node_encoder import TypedNodeEncoder, TypedNodeEncoderConfig

from .collate import collate_operator_examples
from .env_paired_dataset import collate_env_pairs
from .metrics import qerror_summary, summarize_numeric_values


__all__ = [
    "MultiObjectiveTrainer",
    "MultiObjectiveTrainerConfig",
    "ENCODER_MODES",
    "PRETRAINING_OBJECTIVES",
    "graph_reg_loss",
    "info_nce_loss",
]


PRETRAINING_OBJECTIVES = (
    "no_pretrain",
    "contrastive_only",
    "cost_only",
    "full",
)

ENCODER_MODES = (
    "iterative_graph",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MultiObjectiveTrainerConfig:
    """Hyper-parameters for :class:`MultiObjectiveTrainer`.

    Attributes
    ----------
    alpha:
        Weight on the contrastive (InfoNCE) loss term.
    beta:
        Weight on the cost (LogMAE) loss term.
    gamma:
        Outer weight on the graph-regularisation block. The block itself
        is a sum of two pieces (connectivity log-barrier with
        ``gamma_conn`` and Frobenius sparsity with ``gamma_sp``), and
        ``gamma`` then scales the whole thing.
    gamma_conn:
        Connectivity log-barrier sub-weight (default ``1.0``).
    gamma_sp:
        Frobenius sparsity sub-weight (default ``1.0``).
    tau_c:
        InfoNCE temperature.
    max_iter:
        Iterative-refinement budget for the encoder backbone. The forward
        pass runs up to this many message-passing iterations. ``T_max``
        and ``max_iter`` are aliases — both populate this field.
    inference_max_iter:
        Iterative-refinement budget at inference / validation time.
    learning_rate, weight_decay, epochs:
        Standard AdamW hyperparameters.
    batch_size:
        Default batch size. ``pair_batch_size`` and ``single_batch_size``
        override per-mode if set.
    pair_batch_size, single_batch_size:
        Per-mode batch sizes; default to ``batch_size`` when unset (=0).
    num_workers:
        Per-DataLoader worker count. ``0`` keeps unit tests synchronous.
    device:
        ``"cpu"`` or a CUDA device string.
    pretraining_objective:
        One of :data:`PRETRAINING_OBJECTIVES`. See module docstring for
        per-mode behaviour.
    encoder_mode:
        One of :data:`ENCODER_MODES`. The release path uses
        ``iterative_graph``.
    eps_adj:
        Mean-absolute-difference threshold used to early-stop the
        iterative refinement loop. Mirrors :class:`OperatorTrainer`.
    """

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.01
    gamma_conn: float = 1.0
    gamma_sp: float = 1.0
    tau_c: float = 0.5
    max_iter: int = 3
    inference_max_iter: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    batch_size: int = 16
    pair_batch_size: int = 0
    single_batch_size: int = 0
    num_workers: int = 0
    device: str = "cpu"
    pretraining_objective: str = "full"
    encoder_mode: str = "iterative_graph"
    eps_adj: float = 4e-5

    def __post_init__(self) -> None:
        if self.pretraining_objective not in PRETRAINING_OBJECTIVES:
            raise ValueError(
                f"pretraining_objective must be one of {PRETRAINING_OBJECTIVES}, "
                f"got {self.pretraining_objective!r}"
            )
        if self.encoder_mode not in ENCODER_MODES:
            raise ValueError(
                f"encoder_mode must be one of {ENCODER_MODES}, got {self.encoder_mode!r}"
            )
        if self.tau_c <= 0:
            raise ValueError(f"tau_c must be > 0, got {self.tau_c}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.inference_max_iter < 1:
            raise ValueError(
                f"inference_max_iter must be >= 1, got {self.inference_max_iter}"
            )

    @property
    def effective_pair_batch_size(self) -> int:
        return self.pair_batch_size or self.batch_size

    @property
    def effective_single_batch_size(self) -> int:
        return self.single_batch_size or self.batch_size


# --------------------------------------------------------------------------- #
# Loss utilities
# --------------------------------------------------------------------------- #


def info_nce_loss(
    u_a: torch.Tensor,
    u_b: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, float]:
    """SimCLR-style symmetric InfoNCE loss.

    Builds a ``2N × 2N`` similarity matrix from the stacked anchor /
    positive embeddings. Positives are at offsets ``(i, i+N)`` and
    ``(i+N, i)``; the diagonal (self-similarity) is masked out with
    ``-inf`` so it cannot win the cross-entropy.

    Both inputs are assumed L2-normalized (the projection head defaults
    to L2-normalising its output). If they are not, the temperature
    scaling still works but the geometric interpretation of cosine
    similarity does not.

    Parameters
    ----------
    u_a, u_b:
        Float tensors of shape ``[N, d]``.
    temperature:
        Strictly positive scalar; smaller values sharpen the softmax.

    Returns
    -------
    (loss, top1_accuracy):
        Scalar loss tensor (mean cross-entropy across ``2N`` anchors)
        and the float fraction of anchors whose top-1 nearest neighbour
        is the true positive.
    """
    if u_a.shape != u_b.shape:
        raise ValueError(
            f"info_nce_loss expects matching shapes, got u_a={tuple(u_a.shape)} "
            f"u_b={tuple(u_b.shape)}"
        )
    if u_a.dim() != 2:
        raise ValueError(
            f"info_nce_loss expects 2-D inputs [N, d], got {tuple(u_a.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    n = u_a.size(0)
    if n < 2:
        # With fewer than 2 anchors there are no negatives — InfoNCE is
        # ill-defined. Return a 0-loss so the trainer can drop the batch
        # gracefully without diverging.
        zero = u_a.new_zeros(())
        return zero, 1.0 if n == 1 else 0.0

    # Stack into a 2N tensor; row i is anchor i, row i+N is its positive.
    z = torch.cat([u_a, u_b], dim=0)  # [2N, d]
    similarity = torch.matmul(z, z.transpose(0, 1)) / temperature  # [2N, 2N]

    # Mask out the diagonal (each row's similarity to itself).
    diag_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(diag_mask, float("-inf"))

    # Targets: row i's positive is at column i+N; row i+N's is at column i.
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)

    loss = F.cross_entropy(similarity, targets)

    # Top-1 accuracy — fraction of anchors whose argmax (over the masked
    # similarity matrix) lands on the true positive.
    with torch.no_grad():
        predictions = similarity.argmax(dim=1)
        correct = (predictions == targets).float().mean().item()

    return loss, correct


def graph_reg_loss(
    adjacency: torch.Tensor,
    gamma_conn: float,
    gamma_sp: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Connectivity log-barrier + Frobenius sparsity regulariser.

    For an adjacency tensor ``A ∈ R^{B × n × n}``::

        connectivity = -gamma_conn / n  · mean_b mean_i log( (A 1)_{b,i} ).clamp(min=eps)
        sparsity     =  gamma_sp / n^2 · mean_b ||A_b||_F^2
        L_reg        = connectivity + sparsity

    Both terms average over the batch dimension. The connectivity term
    penalises rows whose total in-flow approaches zero (a disconnected
    node), while the Frobenius term encourages overall sparsity.

    Parameters
    ----------
    adjacency:
        Tensor of shape ``[n, n]`` or ``[B, n, n]``.
    gamma_conn, gamma_sp:
        Per-term weights (the outer ``gamma`` from the trainer config is
        applied by the caller, not here).
    eps:
        Floor for ``log`` to keep gradients finite when a row is
        all-zero.
    """
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0)
    if adjacency.dim() != 3:
        raise ValueError(
            f"graph_reg_loss expects [n,n] or [B,n,n], got {tuple(adjacency.shape)}"
        )

    _, n, n2 = adjacency.shape
    if n != n2:
        raise ValueError(
            f"graph_reg_loss expects a square matrix, got [B, {n}, {n2}]"
        )

    # Per-row in-flow: sum each row to a single scalar -> [B, n].
    row_sum = adjacency.sum(dim=-1).clamp(min=eps)
    connectivity = -(gamma_conn / float(n)) * torch.log(row_sum).mean()

    # ||A||_F^2 / n^2 averaged across the batch.
    frob_sq = adjacency.pow(2).sum(dim=(-1, -2))  # [B]
    sparsity = (gamma_sp / float(n * n)) * frob_sq.mean()

    return connectivity + sparsity


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class MultiObjectiveTrainer:
    """Joint contrastive + cost + graph-regularised pre-trainer.

    See module docstring for the high-level design and how
    ``pretraining_objective`` selects an ablation mode.
    """

    def __init__(
        self,
        typed_node_encoder: TypedNodeEncoder,
        typed_node_encoder_config: TypedNodeEncoderConfig,
        graph_learner: GraphLearner,
        graph_learner_config: GraphLearnerConfig,
        graph_updater: GraphUpdater,
        graph_updater_config: GraphUpdateConfig,
        operator_encoder: OperatorEncoder,
        operator_encoder_config: OperatorEncoderConfig,
        cost_head: CostHead,
        cost_head_config: CostHeadConfig,
        projection_head: ProjectionHead,
        projection_head_config: ProjectionHeadConfig,
        trainer_config: MultiObjectiveTrainerConfig,
    ):
        self.typed_node_encoder = typed_node_encoder
        self.typed_node_encoder_config = typed_node_encoder_config
        self.graph_learner = graph_learner
        self.graph_learner_config = graph_learner_config
        self.graph_updater = graph_updater
        self.graph_updater_config = graph_updater_config
        self.operator_encoder = operator_encoder
        self.operator_encoder_config = operator_encoder_config
        self.cost_head = cost_head
        self.cost_head_config = cost_head_config
        self.projection_head = projection_head
        self.projection_head_config = projection_head_config
        self.config = trainer_config
        self.device = torch.device(trainer_config.device)

        for module in (
            self.typed_node_encoder,
            self.graph_learner,
            self.operator_encoder,
            self.cost_head,
            self.projection_head,
        ):
            module.to(self.device)

        parameters = (
            list(self.typed_node_encoder.parameters())
            + list(self.graph_learner.parameters())
            + list(self.operator_encoder.parameters())
            + list(self.cost_head.parameters())
            + list(self.projection_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=trainer_config.learning_rate,
            weight_decay=trainer_config.weight_decay,
        )

        # Resolve effective alpha / beta given the ablation mode. The
        # configured ``alpha`` / ``beta`` are kept on ``self.config``; the
        # effective values used during loss assembly are stored
        # separately so ``build_checkpoint`` round-trips the user-facing
        # config rather than the masked one.
        objective = trainer_config.pretraining_objective
        if objective == "no_pretrain":
            self._alpha = 0.0
            self._beta = 0.0
        elif objective == "contrastive_only":
            self._alpha = trainer_config.alpha
            self._beta = 0.0
        elif objective == "cost_only":
            self._alpha = 0.0
            self._beta = trainer_config.beta
        else:  # "full"
            self._alpha = trainer_config.alpha
            self._beta = trainer_config.beta

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        paired_dataset: Dataset | None,
        single_dataset: Dataset | None,
        valid_paired_dataset: Dataset | None = None,
        valid_single_dataset: Dataset | None = None,
    ) -> list[dict[str, Any]]:
        """Run multi-objective pre-training.

        Parameters
        ----------
        paired_dataset:
            Dataset returning :class:`EnvPair` items. Required for
            ``"full"`` and ``"contrastive_only"`` modes; ignored otherwise.
        single_dataset:
            Dataset returning :class:`OperatorTrainingExample` items.
            Required for ``"full"`` and ``"cost_only"`` modes; ignored
            otherwise.
        valid_paired_dataset, valid_single_dataset:
            Optional validation datasets logged at the end of each epoch.

        Returns
        -------
        list of per-epoch dicts containing ``loss_con``, ``loss_cost``,
        ``loss_reg``, ``total_loss``, ``contrastive_top1_accuracy``,
        ``qerror_summary``, plus epoch timing.

        For ``pretraining_objective == "no_pretrain"`` returns an empty
        list — the modules retain their initial random weights and act
        as the random-encoder baseline.
        """
        objective = self.config.pretraining_objective
        if objective == "no_pretrain":
            return []

        use_paired = objective in {"full", "contrastive_only"}
        use_single = objective in {"full", "cost_only"}

        if use_paired and (paired_dataset is None or len(paired_dataset) == 0):
            raise ValueError(
                f"pretraining_objective={objective!r} requires a non-empty paired_dataset"
            )
        if use_single and (single_dataset is None or len(single_dataset) == 0):
            raise ValueError(
                f"pretraining_objective={objective!r} requires a non-empty single_dataset"
            )

        paired_loader = (
            self._build_paired_loader(paired_dataset, shuffle=True) if use_paired else None
        )
        single_loader = (
            self._build_single_loader(single_dataset, shuffle=True) if use_single else None
        )
        valid_paired_loader = (
            self._build_paired_loader(valid_paired_dataset, shuffle=False)
            if use_paired and valid_paired_dataset is not None and len(valid_paired_dataset) > 0
            else None
        )
        valid_single_loader = (
            self._build_single_loader(valid_single_dataset, shuffle=False)
            if use_single and valid_single_dataset is not None and len(valid_single_dataset) > 0
            else None
        )

        history: list[dict[str, Any]] = []
        fit_start = perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = perf_counter()
            train_metrics = self._run_train_epoch(paired_loader, single_loader)
            record: dict[str, Any] = {"epoch": epoch, "train": train_metrics}

            valid_metrics = self._run_valid_epoch(valid_paired_loader, valid_single_loader)
            if valid_metrics is not None:
                record["valid"] = valid_metrics

            record["epoch_duration_sec"] = perf_counter() - epoch_start
            history.append(record)

        if history:
            history[-1]["fit_duration_sec"] = perf_counter() - fit_start
        return history

    def predict_single(
        self,
        single_dataset: Dataset,
    ) -> tuple[list[float], list[float]]:
        """Run cost-head inference over a single-mode dataset.

        Mirrors :meth:`OperatorTrainer.predict_dataset`. Used by the
        validation loop and diagnostics.
        """
        loader = self._build_single_loader(single_dataset, shuffle=False)
        all_predictions: list[float] = []
        all_targets: list[float] = []

        self._set_eval()
        with torch.no_grad():
            for batch in loader:
                _, _, predictions = self._forward_single(
                    batch, max_iter=self.config.inference_max_iter
                )
                targets = batch["targets"].to(self.device)
                all_predictions.extend(predictions.detach().cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())
        return all_predictions, all_targets

    def build_checkpoint(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Serialise every learnable module + its config + the trainer config.

        The projection head's state dict is written too so it can be reused
        for warm starts. Inference does not use it for embedding extraction.
        """
        return {
            "typed_node_encoder_state_dict": self.typed_node_encoder.state_dict(),
            "graph_learner_state_dict": self.graph_learner.state_dict(),
            "operator_encoder_state_dict": self.operator_encoder.state_dict(),
            "cost_head_state_dict": self.cost_head.state_dict(),
            "projection_head_state_dict": self.projection_head.state_dict(),
            "typed_node_encoder_config": asdict(self.typed_node_encoder_config),
            "graph_learner_config": asdict(self.graph_learner_config),
            "graph_updater_config": asdict(self.graph_updater_config),
            "operator_encoder_config": asdict(self.operator_encoder_config),
            "cost_head_config": asdict(self.cost_head_config),
            "projection_head_config": asdict(self.projection_head_config),
            "trainer_config": asdict(self.config),
            "extra_metadata": extra_metadata or {},
        }

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #

    def _forward_single(
        self,
        batch: dict[str, Any],
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one (single-mode) forward pass.

        Returns ``(z, A_final, cost_predictions)``.
        """
        node_values = batch["node_values"].to(self.device)
        initial_adjacency = batch["initial_adjacency"].to(self.device)
        z, a_final, propagated = self._encode(
            node_values, initial_adjacency, max_iter=max_iter
        )
        predictions = self.cost_head(z)
        return z, a_final, predictions

    def _forward_paired(
        self,
        batch: dict[str, Any],
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one (paired-mode) forward pass — both views serially.

        Returns ``(u_a, u_b, A_final_a, A_final_b)`` for InfoNCE +
        regularisation.
        """
        a_node = batch["view_a_node_values"].to(self.device)
        a_adj = batch["view_a_initial_adjacency"].to(self.device)
        b_node = batch["view_b_node_values"].to(self.device)
        b_adj = batch["view_b_initial_adjacency"].to(self.device)

        z_a, a_final_a, _ = self._encode(a_node, a_adj, max_iter=max_iter)
        z_b, a_final_b, _ = self._encode(b_node, b_adj, max_iter=max_iter)

        u_a = self.projection_head(z_a)
        u_b = self.projection_head(z_b)
        return u_a, u_b, a_final_a, a_final_b

    def _encode(
        self,
        node_values: torch.Tensor,
        initial_adjacency: torch.Tensor,
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run typed encoder + graph learner + iterative refinement loop.

        Returns ``(z, A_final, propagated_states)``. The explicit
        ``max_iter`` lets validation and inference use a smaller refinement
        budget than training.
        """
        budget = max_iter if max_iter is not None else self.config.max_iter

        node_states = self.typed_node_encoder(node_values)

        learned = self.graph_learner(node_states)
        a_first = self.graph_updater.build_first_graph(initial_adjacency, learned)
        z, propagated = self.operator_encoder.encode(node_states, a_first)
        a_curr = a_first

        for _ in range(2, budget + 1):
            learned = self.graph_learner(propagated)
            a_next = self.graph_updater.build_iterative_graph(
                initial_adjacency=initial_adjacency,
                learned_adjacency=learned,
                first_refined_adjacency=a_first,
            )
            delta = self.graph_updater.adjacency_delta(
                a_curr.detach(), a_next.detach()
            )
            z, propagated = self.operator_encoder.encode(propagated, a_next)
            a_curr = a_next
            if delta <= self.config.eps_adj:
                break

        return z, a_curr, propagated

    def _graph_reg_or_zero(self, adjacency: torch.Tensor) -> torch.Tensor:
        return graph_reg_loss(adjacency, self.config.gamma_conn, self.config.gamma_sp)

    # ------------------------------------------------------------------ #
    # Training / validation epochs
    # ------------------------------------------------------------------ #

    def _run_train_epoch(
        self,
        paired_loader: DataLoader | None,
        single_loader: DataLoader | None,
    ) -> dict[str, Any]:
        self._set_train()

        loss_con_sum = 0.0
        loss_cost_sum = 0.0
        loss_reg_sum = 0.0
        total_loss_sum = 0.0
        top1_sum = 0.0
        paired_steps = 0
        single_steps = 0

        cost_predictions: list[float] = []
        cost_targets: list[float] = []

        paired_iter = iter(paired_loader) if paired_loader is not None else None
        single_iter = iter(single_loader) if single_loader is not None else None

        # Alternate between paired and single batches. Each loop iteration
        # tries to pull one batch from each iterator, so both pools get
        # roughly equal step counts. When one exhausts before the other
        # the loop continues until the second is also drained.
        while paired_iter is not None or single_iter is not None:
            if paired_iter is not None:
                try:
                    p_batch = next(paired_iter)
                except StopIteration:
                    paired_iter = None
                else:
                    metrics = self._train_step_paired(p_batch)
                    loss_con_sum += metrics["loss_con"]
                    loss_reg_sum += metrics["loss_reg"]
                    total_loss_sum += metrics["total_loss"]
                    top1_sum += metrics["top1_accuracy"]
                    paired_steps += 1

            if single_iter is not None:
                try:
                    s_batch = next(single_iter)
                except StopIteration:
                    single_iter = None
                else:
                    metrics = self._train_step_single(s_batch)
                    loss_cost_sum += metrics["loss_cost"]
                    loss_reg_sum += metrics["loss_reg"]
                    total_loss_sum += metrics["total_loss"]
                    cost_predictions.extend(metrics["predictions"])
                    cost_targets.extend(metrics["targets"])
                    single_steps += 1

        loss_reg_count = paired_steps + single_steps
        return {
            "loss_con": loss_con_sum / max(paired_steps, 1),
            "loss_cost": loss_cost_sum / max(single_steps, 1),
            "loss_reg": loss_reg_sum / max(loss_reg_count, 1),
            "total_loss": total_loss_sum / max(loss_reg_count, 1),
            "contrastive_top1_accuracy": top1_sum / max(paired_steps, 1),
            "qerror_summary": qerror_summary(cost_predictions, cost_targets),
            "prediction_summary": summarize_numeric_values(cost_predictions),
            "target_summary": summarize_numeric_values(cost_targets),
            "paired_steps": paired_steps,
            "single_steps": single_steps,
        }

    def _train_step_paired(self, batch: dict[str, Any]) -> dict[str, float]:
        """One contrastive batch -> (L_con + L_reg) backward."""
        self.optimizer.zero_grad(set_to_none=True)
        u_a, u_b, a_a, a_b = self._forward_paired(batch)

        loss_con, top1 = info_nce_loss(u_a, u_b, self.config.tau_c)
        # Average L_reg over both views' adjacencies.
        reg_a = self._graph_reg_or_zero(a_a)
        reg_b = self._graph_reg_or_zero(a_b)
        loss_reg = 0.5 * (reg_a + reg_b)

        total = self._alpha * loss_con + self.config.gamma * loss_reg
        total.backward()
        self.optimizer.step()
        return {
            "loss_con": float(loss_con.detach().item()),
            "loss_reg": float(loss_reg.detach().item()),
            "total_loss": float(total.detach().item()),
            "top1_accuracy": float(top1),
        }

    def _train_step_single(self, batch: dict[str, Any]) -> dict[str, Any]:
        """One cost batch -> (L_cost + L_reg) backward."""
        self.optimizer.zero_grad(set_to_none=True)
        targets = batch["targets"].to(self.device)
        _, a_final, predictions = self._forward_single(batch)

        loss_cost = self._logmae(predictions, targets)
        loss_reg = self._graph_reg_or_zero(a_final)

        total = self._beta * loss_cost + self.config.gamma * loss_reg
        total.backward()
        self.optimizer.step()
        return {
            "loss_cost": float(loss_cost.detach().item()),
            "loss_reg": float(loss_reg.detach().item()),
            "total_loss": float(total.detach().item()),
            "predictions": predictions.detach().cpu().tolist(),
            "targets": targets.detach().cpu().tolist(),
        }

    def _run_valid_epoch(
        self,
        paired_loader: DataLoader | None,
        single_loader: DataLoader | None,
    ) -> dict[str, Any] | None:
        if paired_loader is None and single_loader is None:
            return None

        self._set_eval()
        loss_con_sum = 0.0
        loss_cost_sum = 0.0
        loss_reg_sum = 0.0
        top1_sum = 0.0
        paired_steps = 0
        single_steps = 0
        cost_predictions: list[float] = []
        cost_targets: list[float] = []

        with torch.no_grad():
            if paired_loader is not None:
                for batch in paired_loader:
                    u_a, u_b, a_a, a_b = self._forward_paired(
                        batch, max_iter=self.config.inference_max_iter
                    )
                    loss_con, top1 = info_nce_loss(u_a, u_b, self.config.tau_c)
                    reg_a = self._graph_reg_or_zero(a_a)
                    reg_b = self._graph_reg_or_zero(a_b)
                    loss_con_sum += float(loss_con.detach().item())
                    loss_reg_sum += float(0.5 * (reg_a + reg_b).detach().item())
                    top1_sum += float(top1)
                    paired_steps += 1

            if single_loader is not None:
                for batch in single_loader:
                    targets = batch["targets"].to(self.device)
                    _, a_final, predictions = self._forward_single(
                        batch, max_iter=self.config.inference_max_iter
                    )
                    loss_cost = self._logmae(predictions, targets)
                    loss_reg = self._graph_reg_or_zero(a_final)
                    loss_cost_sum += float(loss_cost.detach().item())
                    loss_reg_sum += float(loss_reg.detach().item())
                    single_steps += 1
                    cost_predictions.extend(predictions.detach().cpu().tolist())
                    cost_targets.extend(targets.detach().cpu().tolist())

        loss_reg_count = paired_steps + single_steps
        return {
            "loss_con": loss_con_sum / max(paired_steps, 1),
            "loss_cost": loss_cost_sum / max(single_steps, 1),
            "loss_reg": loss_reg_sum / max(loss_reg_count, 1),
            "contrastive_top1_accuracy": top1_sum / max(paired_steps, 1),
            "qerror_summary": qerror_summary(cost_predictions, cost_targets),
            "prediction_summary": summarize_numeric_values(cost_predictions),
            "target_summary": summarize_numeric_values(cost_targets),
            "paired_steps": paired_steps,
            "single_steps": single_steps,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_paired_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.effective_pair_batch_size,
            shuffle=shuffle,
            collate_fn=collate_env_pairs,
            num_workers=self.config.num_workers,
        )

    def _build_single_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.effective_single_batch_size,
            shuffle=shuffle,
            collate_fn=collate_operator_examples,
            num_workers=self.config.num_workers,
        )

    def _set_train(self) -> None:
        self.typed_node_encoder.train()
        self.graph_learner.train()
        self.operator_encoder.train()
        self.cost_head.train()
        self.projection_head.train()

    def _set_eval(self) -> None:
        self.typed_node_encoder.eval()
        self.graph_learner.eval()
        self.operator_encoder.eval()
        self.cost_head.eval()
        self.projection_head.eval()

    @staticmethod
    def _logmae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        safe_pred = torch.clamp(predictions, min=0.0)
        safe_tgt = torch.clamp(targets, min=0.0)
        return F.l1_loss(torch.log1p(safe_pred), torch.log1p(safe_tgt))
