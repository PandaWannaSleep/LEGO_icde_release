from __future__ import annotations

import argparse
from pathlib import Path

from lego.cag.io import load_cag_template
from lego.data.operator_context_extractor import OperatorContextExtractor
from lego.data.plan_loader import iter_plan_records
from lego.data.schema_stats import load_schema_stats
from lego.inference import InferenceConfig
from lego.inference.checkpoint_loader import load_inferencer_from_checkpoint
from lego.inference.operator_inferencer import OperatorInferencer
from lego.model.cost_predictor import CostHead, CostHeadConfig
from lego.model.encoders import NodeEncoderConfig, ScalarNodeEncoder
from lego.model.graph_learner import GraphLearner, GraphLearnerConfig
from lego.model.graph_updater import GraphUpdateConfig, GraphUpdater
from lego.model.operator_encoder import OperatorEncoder, OperatorEncoderConfig
from lego.model.refinement_engine import RefinementEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect operator-level inference")
    parser.add_argument("--plan-file", required=True, help="Path to a JSONL plan file")
    parser.add_argument("--db-name", required=True, help="Database name, e.g. imdb")
    parser.add_argument("--legacy-root", help="Path to the legacy artifact root")
    parser.add_argument("--schema-cache", help="Explicit schema stats cache artifact (.pickle or .json)")
    parser.add_argument("--templates-dir", required=True, help="Directory containing serialized CAG templates")
    parser.add_argument("--checkpoint-dir", help="Optional trained LEGO checkpoint directory")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--mode", choices=("single_step", "iterative"), default="single_step")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of plans to inspect")
    return parser.parse_args()


def _load_templates(templates_dir: Path) -> dict[str, object]:
    templates = {}
    for path in sorted(templates_dir.glob("*.pkl")):
        template = load_cag_template(path)
        templates[template.operator_type] = template
    if not templates:
        raise ValueError(f"No templates found in {templates_dir}")
    return templates


def main() -> None:
    args = parse_args()
    templates = _load_templates(Path(args.templates_dir))
    first_template = next(iter(templates.values()))

    if args.checkpoint_dir:
        first_operator = next(iter(templates))
        inferencer = load_inferencer_from_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            templates_dir=args.templates_dir,
            operator_type=first_operator,
            device=args.device,
        )
        inferencer = OperatorInferencer(templates=templates, refinement_engine=inferencer.refinement_engine)
    else:
        node_count = len(first_template.node_order)
        node_encoder = ScalarNodeEncoder(NodeEncoderConfig(node_count=node_count, hidden_dim=64))
        graph_learner = GraphLearner(GraphLearnerConfig(hidden_dim=64, metric_type="weighted_cosine", num_heads=4))
        graph_updater = GraphUpdater(GraphUpdateConfig(graph_skip_conn=0.5, update_adj_ratio=0.2, include_self=False))
        operator_encoder_cfg = OperatorEncoderConfig(hidden_dim=64, num_message_passing_layers=2, pool_type="mean")
        operator_encoder = OperatorEncoder(operator_encoder_cfg)
        cost_head = CostHead(CostHeadConfig(input_dim=operator_encoder_cfg.output_dim, hidden_dim=64))

        refinement_engine = RefinementEngine(
            node_encoder=node_encoder,
            graph_learner=graph_learner,
            graph_updater=graph_updater,
            operator_encoder=operator_encoder,
            cost_head=cost_head,
            device=args.device,
        )
        inferencer = OperatorInferencer(templates=templates, refinement_engine=refinement_engine)

    schema_stats = load_schema_stats(
        db_name=args.db_name,
        schema_cache=args.schema_cache,
        legacy_root=args.legacy_root,
    )
    extractor = OperatorContextExtractor(schema_stats=schema_stats)
    inference_config = InferenceConfig(mode=args.mode, max_iter=args.max_iter, eps_adj=4e-5, return_final_adj=True)

    inspected = 0
    for plan_index, plan_record in enumerate(iter_plan_records([args.plan_file]), start=1):
        contexts = extractor.extract_plan(plan_record, source_path=args.plan_file)
        for context in contexts:
            if context.operator_type not in templates:
                continue
            prediction = inferencer.predict(context=context, inference_config=inference_config, strict=False)
            print(
                f"operator={prediction.operator_type} mode={prediction.mode} "
                f"predicted_cost={prediction.predicted_cost:.6f} iterations={prediction.iterations} "
                f"stop_reason={prediction.stop_reason} adj_shape={prediction.final_adjacency.shape}"
            )
            inspected += 1
            break
        if args.limit and plan_index >= args.limit:
            break

    print(f"inspected_predictions={inspected}")


if __name__ == "__main__":
    main()
