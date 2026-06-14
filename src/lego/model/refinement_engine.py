from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
import torch

from lego.cag.instance import OperatorCAG
from .cost_predictor import CostHead

if TYPE_CHECKING:
    # Defer to break the lego.model -> lego.inference -> lego.model cycle.
    # Used only as a parameter type, never at module init.
    from lego.inference.config import InferenceConfig
from .encoders import ScalarNodeEncoder
from .graph_learner import GraphLearner
from .graph_updater import GraphUpdater
from .operator_encoder import OperatorEncoder

InferenceMode = Literal["single_step", "iterative"]


@dataclass(frozen=True)
class RefinementTraceStep:
    iteration: int
    adjacency_delta: float
    predicted_cost: float


@dataclass(frozen=True)
class RefinementResult:
    predicted_cost: float
    final_adjacency: np.ndarray
    final_pooled_embedding: np.ndarray
    iterations: int
    mode: InferenceMode
    stop_reason: str
    trace: tuple[RefinementTraceStep, ...] = ()


@dataclass(frozen=True)
class BatchRefinementResult:
    predicted_costs: np.ndarray
    final_adjacencies: np.ndarray
    final_pooled_embeddings: np.ndarray
    iterations: int
    mode: InferenceMode
    stop_reason: str


class RefinementEngine:
    """Iterative graph refinement engine.

    Holds the encoder backbone (``OperatorEncoder``) plus the cost head
    (``CostHead``) and runs the inference loop:

      1. encode raw node values (``ScalarNodeEncoder``);
      2. learn adjacency over current node states (``GraphLearner``);
      3. fuse with the data-driven prior (``GraphUpdater``);
      4. propagate + pool through ``OperatorEncoder.encode`` -> ``z``;
      5. cost head reads ``z`` (kept for cost-only deployments and trace
         reporting); embedding consumers use ``z`` directly without going
         through the cost head.
    """

    def __init__(
        self,
        node_encoder: ScalarNodeEncoder,
        graph_learner: GraphLearner,
        graph_updater: GraphUpdater,
        operator_encoder: OperatorEncoder,
        cost_head: CostHead,
        encoder_mode: str = "iterative_graph",
        device: str | torch.device | None = None,
    ):
        self.node_encoder = node_encoder
        self.graph_learner = graph_learner
        self.graph_updater = graph_updater
        self.operator_encoder = operator_encoder
        self.cost_head = cost_head
        self.encoder_mode = encoder_mode
        self.device = torch.device(device or "cpu")

        self.node_encoder.to(self.device)
        self.graph_learner.to(self.device)
        self.operator_encoder.to(self.device)
        self.cost_head.to(self.device)

    @torch.no_grad()
    def refine(self, cag: OperatorCAG, inference_config: InferenceConfig) -> RefinementResult:
        node_values = torch.from_numpy(cag.node_values).to(self.device)
        initial_adjacency = torch.from_numpy(cag.initial_adjacency).to(self.device)

        self.node_encoder.eval()
        self.graph_learner.eval()
        self.operator_encoder.eval()
        self.cost_head.eval()

        node_states = self.node_encoder(node_values)
        learned_adjacency = self.graph_learner(node_states)
        current_adjacency = self.graph_updater.build_first_graph(initial_adjacency, learned_adjacency)
        z, node_states = self.operator_encoder.encode(node_states, current_adjacency)
        predicted_cost = self.cost_head(z)

        trace = [
            RefinementTraceStep(
                iteration=1,
                adjacency_delta=0.0,
                predicted_cost=float(predicted_cost.squeeze(0).item()),
            )
        ]

        if inference_config.mode == "single_step":
            return RefinementResult(
                predicted_cost=float(predicted_cost.squeeze(0).item()),
                final_adjacency=current_adjacency.squeeze(0).cpu().numpy(),
                final_pooled_embedding=z.squeeze(0).cpu().numpy(),
                iterations=1,
                mode="single_step",
                stop_reason="single_step",
                trace=tuple(trace),
            )

        first_refined_adjacency = current_adjacency
        iterations = 1
        stop_reason = "max_iter"

        for iteration in range(2, inference_config.max_iter + 1):
            learned_adjacency = self.graph_learner(node_states)
            next_adjacency = self.graph_updater.build_iterative_graph(
                initial_adjacency=initial_adjacency,
                learned_adjacency=learned_adjacency,
                first_refined_adjacency=first_refined_adjacency,
            )
            adjacency_delta = self.graph_updater.adjacency_delta(current_adjacency, next_adjacency)
            z, node_states = self.operator_encoder.encode(node_states, next_adjacency)
            predicted_cost = self.cost_head(z)
            trace.append(
                RefinementTraceStep(
                    iteration=iteration,
                    adjacency_delta=adjacency_delta,
                    predicted_cost=float(predicted_cost.squeeze(0).item()),
                )
            )

            current_adjacency = next_adjacency
            iterations = iteration
            if adjacency_delta <= inference_config.eps_adj:
                stop_reason = "adjacency_converged"
                break

        return RefinementResult(
            predicted_cost=float(predicted_cost.squeeze(0).item()),
            final_adjacency=current_adjacency.squeeze(0).cpu().numpy(),
            final_pooled_embedding=z.squeeze(0).cpu().numpy(),
            iterations=iterations,
            mode="iterative",
            stop_reason=stop_reason,
            trace=tuple(trace),
        )

    @torch.no_grad()
    def refine_batch(self, cag_batch, inference_config: InferenceConfig) -> BatchRefinementResult:
        """Refine a same-operator CAG batch in one forward pass.

        The downstream inference path uses ``single_step`` mode. Iterative
        batch refinement can be added later, but keeping this API scoped avoids
        silently changing the convergence semantics for mixed-length batches.
        """
        if inference_config.mode != "single_step":
            raise ValueError("refine_batch currently supports only single_step inference")

        node_values = torch.from_numpy(cag_batch.node_values).to(self.device)
        initial_adjacency = torch.from_numpy(cag_batch.initial_adjacency).to(self.device)

        self.node_encoder.eval()
        self.graph_learner.eval()
        self.operator_encoder.eval()
        self.cost_head.eval()

        node_states = self.node_encoder(node_values)
        learned_adjacency = self.graph_learner(node_states)
        current_adjacency = self.graph_updater.build_first_graph(initial_adjacency, learned_adjacency)
        z, _node_states = self.operator_encoder.encode(node_states, current_adjacency)
        predicted_costs = self.cost_head(z)

        return BatchRefinementResult(
            predicted_costs=predicted_costs.detach().cpu().numpy(),
            final_adjacencies=current_adjacency.detach().cpu().numpy(),
            final_pooled_embeddings=z.detach().cpu().numpy(),
            iterations=1,
            mode="single_step",
            stop_reason="single_step",
        )
