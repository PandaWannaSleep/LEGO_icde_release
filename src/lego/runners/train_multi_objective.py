"""Train multi-objective operator embedding with real collected data.

Usage:
    PYTHONPATH=src conda run -n paramgraph python -m lego.runners.train_multi_objective \\
        --pairs-file data/job_light/processed/pairs.jsonl \\
        --singles-file data/job_light/splits/train.jsonl \\
        --valid-singles-file data/job_light/splits/valid.jsonl \\
        --operator-type "Index Scan" \\
        --task runtime_cost \\
        --output-dir runs/multi_obj_$(date +%m%d-%H%M) \\
        --epochs 50 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml

from lego.data.operator_context_extractor import OperatorContextExtractor
from lego.cag.io import load_cag_template
from lego.inference.manifest_writer import build_manifest
from lego.training.env_paired_dataset import EnvPairedOperatorDataset
from lego.training.targets import LocalTargetBuilder
from lego.training.multi_objective_trainer import (
    ENCODER_MODES,
    MultiObjectiveTrainer,
    MultiObjectiveTrainerConfig,
    PRETRAINING_OBJECTIVES,
)
from lego.model.cost_predictor import CostHead, CostHeadConfig
from lego.model.graph_learner import GraphLearner, GraphLearnerConfig
from lego.model.graph_updater import GraphUpdater, GraphUpdateConfig
from lego.model.operator_encoder import OperatorEncoder, OperatorEncoderConfig
from lego.model.projection_head import ProjectionHead, ProjectionHeadConfig
from lego.model.typed_node_encoder import TypedNodeEncoder, TypedNodeEncoderConfig
from lego.training.checkpoint_io import save_operator_checkpoint


logger = logging.getLogger(__name__)


def _operator_slug(operator_type: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", operator_type.lower()).strip("_")


def _read_config(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _get(config: dict, *keys: str, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _operator_config(config: dict, operator_type: str | None) -> dict:
    if not operator_type:
        return {}
    for item in config.get("operators", []) or []:
        if item.get("type") == operator_type:
            return item
    return {}


def _fill_if_missing(args: argparse.Namespace, name: str, value) -> None:
    if getattr(args, name) is None and value is not None:
        setattr(args, name, value)


def _apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    config = _read_config(args.config)
    op_cfg = _operator_config(config, args.operator_type)

    _fill_if_missing(args, "pairs_file", _get(config, "data", "pairs_file"))
    _fill_if_missing(args, "singles_file", _get(config, "data", "train_singles_file"))
    _fill_if_missing(args, "valid_singles_file", _get(config, "data", "valid_singles_file"))
    _fill_if_missing(args, "task", _get(config, "experiment", "task"))
    _fill_if_missing(args, "pretraining_objective", _get(config, "experiment", "pretraining_objective"))

    templates_dir = _get(config, "initial_graph", "templates_dir")
    template_file = op_cfg.get("template_file")
    if args.template_path is None and templates_dir and template_file:
        args.template_path = str(Path(templates_dir) / template_file)

    checkpoint_root = _get(config, "artifacts", "checkpoint_root")
    if args.output_dir is None and checkpoint_root and args.operator_type:
        args.output_dir = str(Path(checkpoint_root) / f"synthetic_pak_lw_{_operator_slug(args.operator_type)}")

    _fill_if_missing(args, "device", op_cfg.get("device"))
    _fill_if_missing(args, "paired_epoch_size", op_cfg.get("paired_epoch_size"))

    _fill_if_missing(args, "hidden_dim", _get(config, "model", "hidden_dim"))
    _fill_if_missing(args, "num_message_passing_layers", _get(config, "model", "num_message_passing_layers"))
    _fill_if_missing(args, "pool_type", _get(config, "model", "pool_type"))
    _fill_if_missing(args, "projection_dim", _get(config, "model", "projection_dim"))
    _fill_if_missing(args, "post_mp_layers", _get(config, "model", "post_mp_layers"))
    _fill_if_missing(args, "post_mp_hidden_dim", _get(config, "model", "post_mp_hidden_dim"))
    _fill_if_missing(args, "post_mp_output_dim", _get(config, "model", "post_mp_output_dim"))
    _fill_if_missing(args, "encoder_mode", _get(config, "model", "encoder_mode"))

    _fill_if_missing(args, "graph_metric_type", _get(config, "graph_learning", "graph_metric_type"))
    _fill_if_missing(args, "graph_num_heads", _get(config, "graph_learning", "graph_num_heads"))
    _fill_if_missing(args, "graph_hyper_hidden_dim", _get(config, "graph_learning", "graph_hyper_hidden_dim"))
    _fill_if_missing(args, "max_iter", _get(config, "graph_learning", "max_iter"))
    _fill_if_missing(args, "inference_max_iter", _get(config, "graph_learning", "inference_max_iter"))

    _fill_if_missing(args, "epochs", _get(config, "optimization", "epochs"))
    _fill_if_missing(args, "seed", _get(config, "optimization", "seed"))
    _fill_if_missing(args, "pair_batch_size", _get(config, "optimization", "pair_batch_size"))
    _fill_if_missing(args, "single_batch_size", _get(config, "optimization", "single_batch_size"))
    _fill_if_missing(args, "learning_rate", _get(config, "optimization", "learning_rate"))
    _fill_if_missing(args, "weight_decay", _get(config, "optimization", "weight_decay"))
    _fill_if_missing(args, "alpha", _get(config, "optimization", "alpha"))
    _fill_if_missing(args, "beta", _get(config, "optimization", "beta"))
    _fill_if_missing(args, "gamma", _get(config, "optimization", "gamma"))

    _fill_if_missing(args, "positive_cross_condition_only", _get(config, "pair_sampling", "positive_cross_condition_only"))
    _fill_if_missing(args, "pair_group_weight", _get(config, "pair_sampling", "pair_group_weight"))
    _fill_if_missing(args, "pair_group_weight_cap", _get(config, "pair_sampling", "pair_group_weight_cap"))

    fallbacks = {
        "epochs": 50,
        "device": "cuda:0",
        "seed": 42,
        "hidden_dim": 128,
        "num_message_passing_layers": 3,
        "pool_type": "concat",
        "projection_dim": 64,
        "graph_metric_type": "learned_weight",
        "graph_num_heads": 1,
        "graph_hyper_hidden_dim": 0,
        "post_mp_layers": 0,
        "post_mp_hidden_dim": 64,
        "post_mp_output_dim": 0,
        "max_iter": 5,
        "inference_max_iter": 10,
        "pair_batch_size": 32,
        "single_batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.1,
        "pretraining_objective": "full",
        "encoder_mode": "iterative_graph",
        "limit_pairs": 0,
        "limit_singles": 0,
        "paired_epoch_size": 0,
        "pair_group_weight": "uniform",
        "pair_group_weight_cap": 0.0,
        "positive_cross_condition_only": False,
        "log_level": "INFO",
    }
    for name, value in fallbacks.items():
        _fill_if_missing(args, name, value)

    required = ["pairs_file", "singles_file", "operator_type", "task", "output_dir", "template_path"]
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        parser.error(
            "missing required arguments after applying config: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if args.config and not op_cfg:
        parser.error(f"--operator-type {args.operator_type!r} is not listed in {args.config}")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-objective operator embedding")
    p.add_argument("--config", help="YAML experiment config. CLI arguments override config values.")
    p.add_argument("--pairs-file", help="Paired data JSONL file")
    p.add_argument("--singles-file", help="Training singles JSONL file")
    p.add_argument("--valid-singles-file", help="Validation singles JSONL file (optional)")
    p.add_argument("--operator-type", help='e.g. "Index Scan"')
    p.add_argument("--task", choices=["runtime_cost", "startup_cost"])
    p.add_argument("--output-dir", help="Output directory for checkpoint")
    p.add_argument(
        "--template-path",
        help="Serialized MI-initialized CAG template.",
    )

    # Training hyperparameters
    p.add_argument("--epochs", type=int)
    p.add_argument("--device")
    p.add_argument("--seed", type=int)

    # Model architecture
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--num-message-passing-layers", type=int)
    p.add_argument("--pool-type", choices=["mean", "sum", "max", "concat", "attention"])
    p.add_argument("--projection-dim", type=int)
    p.add_argument(
        "--graph-metric-type",
        choices=["learned_weight", "weighted_cosine", "attention", "instance_conditional"],
        help="Graph learner metric.",
    )
    p.add_argument(
        "--graph-num-heads",
        type=int,
        help="Number of graph learner heads for weighted_cosine/attention. Use 1 for single-head weighted cosine.",
    )
    p.add_argument(
        "--graph-hyper-hidden-dim",
        type=int,
        help="Hidden dim for instance_conditional graph learner hypernetwork (0=default hidden_dim).",
    )

    # Post-MP MLP (from HomoParamGCN)
    p.add_argument("--post-mp-layers", type=int, help="Post-pooling MLP layers (0=disabled)")
    p.add_argument("--post-mp-hidden-dim", type=int)
    p.add_argument("--post-mp-output-dim", type=int, help="z output dim (0=auto from pool_type)")

    # Training dynamics
    p.add_argument("--max-iter", type=int, help="Graph learner iterations during training")
    p.add_argument("--inference-max-iter", type=int, help="Graph learner iterations during inference")
    p.add_argument("--pair-batch-size", type=int)
    p.add_argument("--single-batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--weight-decay", type=float)

    # Loss weights
    p.add_argument("--alpha", type=float, help="Contrastive loss weight")
    p.add_argument("--beta", type=float, help="Cost loss weight")
    p.add_argument("--gamma", type=float, help="Graph regularization loss weight")
    p.add_argument(
        "--pretraining-objective",
        choices=PRETRAINING_OBJECTIVES,
        help=(
            "Operator pretraining loss branches: full=contrastive+cost, "
            "contrastive_only=paired contrastive branch, cost_only=cost regression branch."
        ),
    )
    p.add_argument(
        "--encoder-mode",
        choices=ENCODER_MODES,
        help=(
            "Operator embedding encoder mode. The release path uses "
            "iterative graph refinement over the MI-initialized topology."
        ),
    )
    # Data loading
    p.add_argument("--limit-pairs", type=int, help="Limit number of pairs (0=all)")
    p.add_argument("--limit-singles", type=int, help="Limit number of singles (0=all)")
    p.add_argument(
        "--paired-epoch-size",
        type=int,
        help=(
            "Fixed paired-mode samples per epoch (0=one per loaded anchor). "
            "Use this for positive-anchor-key data to keep training scale "
            "matched with the plan-position protocol."
        ),
    )
    p.add_argument(
        "--pair-group-weight",
        choices=["uniform", "sqrt_cross_pairs", "sqrt_all_pairs"],
        help="Anchor-group weighting policy when --paired-epoch-size is positive.",
    )
    p.add_argument(
        "--pair-group-weight-cap",
        type=float,
        help="Optional cap on dynamic anchor-group sampling weights (0=disabled).",
    )
    p.add_argument(
        "--positive-cross-condition-only",
        action="store_true",
        default=None,
        help="Sample positive views from different runtime conditions only.",
    )

    p.add_argument("--log-level")
    args = p.parse_args(argv)
    return _apply_config_defaults(args, p)


def build_trainer(template, args: argparse.Namespace) -> tuple[MultiObjectiveTrainer, dict]:
    """Build trainer and return (trainer, configs_dict)."""

    n_nodes = template.node_schema.size

    # TypedNodeEncoder
    typed_cfg = TypedNodeEncoderConfig(hidden_dim=args.hidden_dim)
    typed_node_encoder = TypedNodeEncoder(template.node_schema, typed_cfg)

    # GraphLearner
    learner_kwargs = {
        "hidden_dim": args.hidden_dim,
        "metric_type": args.graph_metric_type,
        "num_heads": args.graph_num_heads,
    }
    if args.graph_metric_type in {"learned_weight", "instance_conditional"}:
        learner_kwargs["node_count"] = n_nodes
    if args.graph_hyper_hidden_dim > 0:
        learner_kwargs["hyper_hidden_dim"] = args.graph_hyper_hidden_dim
    learner_cfg = GraphLearnerConfig(**learner_kwargs)
    graph_learner = GraphLearner(learner_cfg)

    # GraphUpdater
    updater_cfg = GraphUpdateConfig()
    graph_updater = GraphUpdater(updater_cfg)

    # OperatorEncoder
    encoder_cfg = OperatorEncoderConfig(
        encoder_mode="graph",
        hidden_dim=args.hidden_dim,
        num_message_passing_layers=args.num_message_passing_layers,
        pool_type=args.pool_type,
        post_mp_layers=args.post_mp_layers,
        post_mp_hidden_dim=args.post_mp_hidden_dim,
        post_mp_output_dim=args.post_mp_output_dim,
    )
    operator_encoder = OperatorEncoder(encoder_cfg)

    # CostHead
    cost_cfg = CostHeadConfig(
        input_dim=encoder_cfg.output_dim,
        hidden_dim=args.hidden_dim,
    )
    cost_head = CostHead(cost_cfg)

    # ProjectionHead
    proj_cfg = ProjectionHeadConfig(
        input_dim=encoder_cfg.output_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.projection_dim,
        l2_normalize=True,
    )
    projection_head = ProjectionHead(proj_cfg)

    # MultiObjectiveTrainer
    trainer_cfg = MultiObjectiveTrainerConfig(
        pretraining_objective=args.pretraining_objective,
        encoder_mode=args.encoder_mode,
        max_iter=args.max_iter,
        inference_max_iter=args.inference_max_iter,
        pair_batch_size=args.pair_batch_size,
        single_batch_size=args.single_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        epochs=args.epochs,
        device=args.device,
    )

    trainer = MultiObjectiveTrainer(
        typed_node_encoder=typed_node_encoder,
        typed_node_encoder_config=typed_cfg,
        graph_learner=graph_learner,
        graph_learner_config=learner_cfg,
        graph_updater=graph_updater,
        graph_updater_config=updater_cfg,
        operator_encoder=operator_encoder,
        operator_encoder_config=encoder_cfg,
        cost_head=cost_head,
        cost_head_config=cost_cfg,
        projection_head=projection_head,
        projection_head_config=proj_cfg,
        trainer_config=trainer_cfg,
    )

    configs_dict = {
        "typed_node_encoder_config": typed_cfg.__dict__,
        "graph_learner_config": learner_cfg.__dict__,
        "graph_updater_config": updater_cfg.__dict__,
        "operator_encoder_config": encoder_cfg.__dict__,
        "cost_head_config": cost_cfg.__dict__,
        "projection_head_config": proj_cfg.__dict__,
        "trainer_config": trainer_cfg.__dict__,
    }

    return trainer, configs_dict


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Multi-Objective Operator Embedding Training ===")
    logger.info(
        "operator_type=%s task=%s pretraining_objective=%s",
        args.operator_type,
        args.task,
        args.pretraining_objective,
    )
    logger.info("encoder_mode=%s", args.encoder_mode)
    logger.info("pairs_file=%s", args.pairs_file)
    logger.info("singles_file=%s", args.singles_file)
    logger.info("output_dir=%s", output_dir)

    # 1. Build template and datasets
    saved_template_path: Path | None = None
    template = load_cag_template(args.template_path)
    if template.operator_type != args.operator_type:
        raise ValueError(
            f"Template operator_type={template.operator_type!r} does not match "
            f"--operator-type={args.operator_type!r}"
        )
    saved_template_path = Path(args.template_path)
    initial_edge_count = int((template.initial_adjacency != 0).sum())
    logger.info(
        "initial_graph_method=%s template_path=%s node_count=%d initial_edge_count=%d",
        template.build_method,
        saved_template_path,
        template.node_schema.size,
        initial_edge_count,
    )
    extractor = OperatorContextExtractor()
    target_builder = LocalTargetBuilder()

    logger.info("Loading paired dataset...")
    paired_dataset = EnvPairedOperatorDataset.from_files(
        operator_type=args.operator_type,
        task_name=args.task,
        template=template,
        extractor=extractor,
        target_builder=target_builder,
        pairs_files=[args.pairs_file],
        include_anchor_views_in_singles=False,
        mode="paired",
        limit_pairs=args.limit_pairs,
        paired_epoch_size=args.paired_epoch_size,
        pair_group_weight=args.pair_group_weight,
        pair_group_weight_cap=args.pair_group_weight_cap,
        positive_cross_condition_only=args.positive_cross_condition_only,
        rng_seed=args.seed,
    )

    logger.info("Loading training singles dataset...")
    single_dataset = EnvPairedOperatorDataset.from_files(
        operator_type=args.operator_type,
        task_name=args.task,
        template=template,
        extractor=extractor,
        target_builder=target_builder,
        singles_files=[args.singles_file],
        include_anchor_views_in_singles=False,
        mode="single",
        limit_singles=args.limit_singles,
        rng_seed=args.seed,
    )

    # Optional validation dataset
    valid_dataset = None
    if args.valid_singles_file:
        logger.info("Loading validation singles dataset...")
        valid_dataset = EnvPairedOperatorDataset.from_files(
            operator_type=args.operator_type,
            task_name=args.task,
            template=template,
            extractor=extractor,
            target_builder=target_builder,
            singles_files=[args.valid_singles_file],
            include_anchor_views_in_singles=False,
            mode="single",
            rng_seed=args.seed,
        )

    logger.info(
        "Datasets loaded: %d anchors / %d train singles / %d valid singles",
        len(paired_dataset.anchors),
        len(single_dataset.singles),
        len(valid_dataset.singles) if valid_dataset else 0,
    )
    logger.info("Pair sampling: %s", json.dumps(paired_dataset.pair_sampling_summary, sort_keys=True))

    # 2. Build trainer
    logger.info("Building trainer...")
    trainer, configs_dict = build_trainer(template, args)

    # 3. Train
    logger.info("Starting training for %d epochs...", args.epochs)
    fit_start = perf_counter()
    history = trainer.fit(
        paired_dataset=paired_dataset,
        single_dataset=single_dataset,
    )
    fit_duration = perf_counter() - fit_start

    logger.info("Training completed in %.2f seconds", fit_duration)

    # 4. Save checkpoint
    logger.info("Saving checkpoint...")
    checkpoint = trainer.build_checkpoint(
        extra_metadata={
            "operator_type": args.operator_type,
            "task": args.task,
            "initial_graph_method": template.build_method,
            "template_path": str(saved_template_path) if saved_template_path else args.template_path,
            "encoder_mode": args.encoder_mode,
            "feature_ablation": "full",
        }
    )

    metadata = {
        "operator_type": args.operator_type,
        "task": args.task,
        "train_paired_size": len(paired_dataset.anchors),
        "train_paired_epoch_size": len(paired_dataset),
        "pair_sampling": paired_dataset.pair_sampling_summary,
        "train_single_size": len(single_dataset.singles),
        "history": history,
        "training_duration_sec": fit_duration,
        "initial_graph_method": template.build_method,
        "initial_graph_template_path": str(saved_template_path) if saved_template_path else args.template_path,
        "initial_graph_edge_count": initial_edge_count,
        "initial_graph_metadata": template.metadata,
        "encoder_mode": args.encoder_mode,
        "feature_ablation": "full",
        "node_schema": {
            "behavior_nodes": list(template.node_schema.behavior_nodes),
            "resource_nodes": list(template.node_schema.resource_nodes),
            "table_heat_nodes": list(template.node_schema.table_heat_nodes),
            "node_count": template.node_schema.size,
        },
        **configs_dict,
    }

    save_operator_checkpoint(
        output_dir=output_dir,
        checkpoint=checkpoint,
        metadata=metadata,
    )

    # 5. Build manifest
    logger.info("Building manifest...")
    manifest_path = output_dir / "manifest.json"
    build_manifest(
        task_name=args.task,
        operator_to_checkpoint_dir={args.operator_type: output_dir},
        output_path=manifest_path,
    )

    # 6. Print summary
    final_epoch = history[-1]
    logger.info("=== Training Summary ===")
    logger.info("Final epoch: %d", final_epoch["epoch"])
    logger.info("Final train loss_con: %.4f", final_epoch["train"]["loss_con"])
    logger.info("Final train loss_cost: %.4f", final_epoch["train"]["loss_cost"])
    logger.info("Final train loss_reg: %.4f", final_epoch["train"]["loss_reg"])
    logger.info("Checkpoint saved to: %s", output_dir / "model.pt")
    logger.info("Manifest saved to: %s", manifest_path)

    print(f"output_dir={output_dir}")
    print(f"checkpoint={output_dir / 'model.pt'}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
