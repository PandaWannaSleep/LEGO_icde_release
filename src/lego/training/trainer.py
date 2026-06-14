from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from lego.model.cost_predictor import CostHead, CostHeadConfig
from lego.model.encoders import NodeEncoderConfig, ScalarNodeEncoder
from lego.model.graph_learner import GraphLearner, GraphLearnerConfig
from lego.model.graph_updater import GraphUpdateConfig, GraphUpdater
from lego.model.operator_encoder import OperatorEncoder, OperatorEncoderConfig
from .collate import collate_operator_examples
from .dataset import OperatorDataset
from .metrics import qerror_summary, summarize_numeric_values


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    inference_mode: str = "iterative"
    max_iter: int = 5
    eps_adj: float = 4e-5
    loss_name: str = "logmae"
    device: str = "cpu"


class _LogMAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_loss = nn.L1Loss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        safe_predictions = torch.clamp(predictions, min=0.0)
        safe_targets = torch.clamp(targets, min=0.0)
        return self.base_loss(torch.log1p(safe_predictions), torch.log1p(safe_targets))


class _LogSmoothL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_loss = nn.SmoothL1Loss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        safe_predictions = torch.clamp(predictions, min=0.0)
        safe_targets = torch.clamp(targets, min=0.0)
        return self.base_loss(torch.log1p(safe_predictions), torch.log1p(safe_targets))


class OperatorTrainer:
    """Cost-only trainer kept for checkpoint compatibility."""

    def __init__(
        self,
        node_encoder: ScalarNodeEncoder,
        node_encoder_config: NodeEncoderConfig,
        graph_learner: GraphLearner,
        graph_learner_config: GraphLearnerConfig,
        graph_updater: GraphUpdater,
        graph_updater_config: GraphUpdateConfig,
        operator_encoder: OperatorEncoder,
        operator_encoder_config: OperatorEncoderConfig,
        cost_head: CostHead,
        cost_head_config: CostHeadConfig,
        trainer_config: TrainerConfig,
    ):
        self.node_encoder = node_encoder
        self.node_encoder_config = node_encoder_config
        self.graph_learner = graph_learner
        self.graph_learner_config = graph_learner_config
        self.graph_updater = graph_updater
        self.graph_updater_config = graph_updater_config
        self.operator_encoder = operator_encoder
        self.operator_encoder_config = operator_encoder_config
        self.cost_head = cost_head
        self.cost_head_config = cost_head_config
        self.config = trainer_config
        self.device = torch.device(trainer_config.device)

        self.node_encoder.to(self.device)
        self.graph_learner.to(self.device)
        self.operator_encoder.to(self.device)
        self.cost_head.to(self.device)

        parameters = (
            list(self.node_encoder.parameters())
            + list(self.graph_learner.parameters())
            + list(self.operator_encoder.parameters())
            + list(self.cost_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=trainer_config.learning_rate,
            weight_decay=trainer_config.weight_decay,
        )
        self.loss_fn = self._build_loss(trainer_config.loss_name)

    def fit(
        self,
        train_dataset: OperatorDataset,
        valid_dataset: OperatorDataset | None = None,
    ) -> list[dict[str, Any]]:
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_operator_examples,
        )
        valid_loader = None
        if valid_dataset is not None and len(valid_dataset) > 0:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_operator_examples,
            )

        history: list[dict[str, Any]] = []
        fit_start = perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = perf_counter()
            train_metrics = self._run_epoch(train_loader, train=True)
            record: dict[str, Any] = {"epoch": epoch, "train": train_metrics}
            if valid_loader is not None:
                record["valid"] = self._run_epoch(valid_loader, train=False)
            record["epoch_duration_sec"] = perf_counter() - epoch_start
            history.append(record)
        total_duration_sec = perf_counter() - fit_start
        if history:
            history[-1]["fit_duration_sec"] = total_duration_sec
        return history

    def predict_dataset(self, dataset: OperatorDataset) -> tuple[list[float], list[float]]:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_operator_examples,
        )
        all_predictions: list[float] = []
        all_targets: list[float] = []
        self.node_encoder.eval()
        self.graph_learner.eval()
        self.operator_encoder.eval()
        self.cost_head.eval()

        with torch.no_grad():
            for batch in loader:
                targets = batch["targets"].to(self.device)
                predictions, _ = self._forward_batch(batch)
                all_predictions.extend(predictions.detach().cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())
        return all_predictions, all_targets

    def build_checkpoint(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "node_encoder_state_dict": self.node_encoder.state_dict(),
            "graph_learner_state_dict": self.graph_learner.state_dict(),
            "operator_encoder_state_dict": self.operator_encoder.state_dict(),
            "cost_head_state_dict": self.cost_head.state_dict(),
            "operator_encoder_config": asdict(self.operator_encoder_config),
            "cost_head_config": asdict(self.cost_head_config),
            "node_encoder_config": asdict(self.node_encoder_config),
            "graph_learner_config": asdict(self.graph_learner_config),
            "graph_updater_config": asdict(self.graph_updater_config),
            "trainer_config": asdict(self.config),
            "extra_metadata": extra_metadata or {},
        }

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, Any]:
        if train:
            self.node_encoder.train()
            self.graph_learner.train()
            self.operator_encoder.train()
            self.cost_head.train()
        else:
            self.node_encoder.eval()
            self.graph_learner.eval()
            self.operator_encoder.eval()
            self.cost_head.eval()

        total_loss = 0.0
        batch_count = 0
        all_predictions: list[float] = []
        all_targets: list[float] = []

        for batch in loader:
            targets = batch["targets"].to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                predictions, _ = self._forward_batch(batch)
                loss = self.loss_fn(predictions, targets)

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += float(loss.detach().item())
            batch_count += 1
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

        qerror = qerror_summary(all_predictions, all_targets)
        prediction_summary = summarize_numeric_values(all_predictions)
        target_summary = summarize_numeric_values(all_targets)
        return {
            "loss": total_loss / max(batch_count, 1),
            "qerror": qerror,
            "prediction_summary": prediction_summary,
            "target_summary": target_summary,
        }

    def _forward_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        node_values = batch["node_values"].to(self.device)
        initial_adjacency = batch["initial_adjacency"].to(self.device)

        node_states = self.node_encoder(node_values)
        learned_adjacency = self.graph_learner(node_states)
        current_adjacency = self.graph_updater.build_first_graph(initial_adjacency, learned_adjacency)
        z, node_states = self.operator_encoder.encode(node_states, current_adjacency)
        predictions = self.cost_head(z)

        if self.config.inference_mode == "single_step":
            return predictions, current_adjacency

        first_refined_adjacency = current_adjacency
        for _ in range(2, self.config.max_iter + 1):
            learned_adjacency = self.graph_learner(node_states)
            next_adjacency = self.graph_updater.build_iterative_graph(
                initial_adjacency=initial_adjacency,
                learned_adjacency=learned_adjacency,
                first_refined_adjacency=first_refined_adjacency,
            )
            z, node_states = self.operator_encoder.encode(node_states, next_adjacency)
            predictions = self.cost_head(z)
            adjacency_delta = self.graph_updater.adjacency_delta(current_adjacency.detach(), next_adjacency.detach())
            current_adjacency = next_adjacency
            if adjacency_delta <= self.config.eps_adj:
                break
        return predictions, current_adjacency

    def _build_loss(self, loss_name: str) -> nn.Module:
        normalized = loss_name.lower()
        if normalized == "mse":
            return nn.MSELoss()
        if normalized == "smooth_l1":
            return nn.SmoothL1Loss()
        if normalized in {"logmae", "log_mae"}:
            return _LogMAELoss()
        if normalized == "log_smooth_l1":
            return _LogSmoothL1Loss()
        raise ValueError(f"Unsupported loss: {loss_name}")
