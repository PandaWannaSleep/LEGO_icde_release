# LEGO Operator Embedding

This README shows the minimal pretraining path. `configs/default.yml` is used
only by the operator encoder training command. Data preparation, template
construction, manifest generation, and inference use explicit command-line
arguments.

## Positive Pair Assembly

Build positive pairs and single-view records from the irregular raw plan file.
The positive-anchor key itself does not use `concurrency_level`; it groups
operator views by operator identity, pipeline path, and cardinality bucket.
`concurrency_level` is preserved as record metadata only.

```bash
$PY -m lego.runners.assemble_pairs \
  --irregular-file data/synthetic/raw/synthetic_irregular_0614.jsonl \
  --pairs-path data/synthetic/processed/irregular/pairs_positive_anchor_key.jsonl \
  --singles-path data/synthetic/processed/irregular/singles_positive_anchor_key.jsonl \
  --data-quality-path data/synthetic/processed/irregular/data_quality_positive_anchor_key.json \
  --pairing-policy positive_anchor_key \
  --skip-invalid-plans
```

Split single-view records by query template:

```bash
$PY -m lego.runners.split_pairs \
  --singles-path data/synthetic/processed/irregular/singles_positive_anchor_key.jsonl \
  --train-path data/synthetic/splits/irregular/train.jsonl \
  --valid-path data/synthetic/splits/irregular/valid.jsonl \
  --test-path data/synthetic/splits/irregular/test.jsonl \
  --train-ratio 0.7 \
  --valid-ratio 0.1 \
  --seed 42
```

The positive-anchor-key policy matches operator views using operator identity,
pipeline-breaker-to-node path, and cardinality bucket. For irregular pretraining,
paired sampling draws any two views under the same anchor; it is not restricted
to different runtime conditions.

## Initial Graph Templates

The default release workflow builds MI-initialized CAG templates from the
irregular raw plan records and then learns graph weights during iterative
refinement:

```bash
$PY -m lego.runners.build_templates \
  --plan-file data/synthetic/raw/synthetic_irregular_0614.jsonl \
  --db-name imdb \
  --schema-cache artifacts/schema/scheme_imdb_histogram_info.pickle \
  --output-dir runs/positive_anchor_key_mi_initialized/templates \
  --mi-threshold 0.4 \
  --skip-invalid-plans
```



## Operator Encoder Training

Train one encoder per operator type. This is the only stage that reads
`configs/default.yml`. The file contains the training input paths, template
directory, checkpoint root, operator-specific settings, model hyperparameters,
graph-learning settings, optimization settings, and pair-sampling settings.
Only `--operator-type` is required for selecting the operator entry.

```bash
$PY -m lego.runners.train_multi_objective \
  --config configs/default.yml \
  --operator-type "Index Scan"
```

To train all default operator encoders sequentially:

```bash
for op in "Seq Scan" "Index Scan" "Index Only Scan" "Nested Loop" "Hash Join" "Hash" "Aggregate" "Gather"; do
  $PY -m lego.runners.train_multi_objective \
    --config configs/default.yml \
    --operator-type "$op"
done
```

Command-line arguments override the training config when needed. For example, use
`--device cuda:0` to temporarily move one operator to another GPU, or
`--output-dir runs/debug_index_scan` for a short debugging run.

The full objective uses the contrastive branch, cost regression branch, and
graph regularization branch. The trainer alternates paired batches for
contrastive updates and single-view batches for cost-regression updates.

## Manifest Generation

After all operator encoders finish, build one manifest for downstream loading:

```bash
$PY -m lego.runners.build_checkpoint_manifest \
  --task runtime_cost \
  --checkpoint-root runs/positive_anchor_key_mi_initialized/checkpoints \
  --checkpoint-prefix synthetic_pak_lw \
  --output-path runs/positive_anchor_key_mi_initialized/full_manifest.json
```

If checkpoint directories include a shared run suffix, pass it through
`--tag`.

## Loading a Trained Model

Use `LEGOOperatorEmbedder.from_manifest` to load all operator encoders from a
manifest and a template directory:

```python
from lego.data.operator_context_extractor import OperatorContextExtractor
from lego.data.schema_stats import load_schema_stats
from lego.inference.config import InferenceConfig
from lego.inference.operator_embedding_api import LEGOOperatorEmbedder

schema_stats = load_schema_stats(
    db_name="imdb",
    schema_cache="artifacts/schema/scheme_imdb_histogram_info.pickle",
)
extractor = OperatorContextExtractor(schema_stats=schema_stats)

embedder = LEGOOperatorEmbedder.from_manifest(
    manifest_path="runs/positive_anchor_key_mi_initialized/full_manifest.json",
    templates_dir="runs/positive_anchor_key_mi_initialized/templates",
    task_name="runtime_cost",
    extractor=extractor,
    device="cuda:0",
)

inference_config = InferenceConfig(
    mode="iterative",
    max_iter=10,
    eps_adj=4e-5,
    return_final_adj=False,
)
```

## Embedding One Operator

If you already have an operator plan node, extract its `OperatorContext` and
call the per-operator registry:

```python
context = extractor.extract_operator(
    node=plan_node,
    query_text=query_text,
    config=plan_record.get("config", {}),
    table_heat_metrics=plan_record.get("table_heat_metrics", {}),
    source_path="query.jsonl:1",
)

registry_result = embedder.registry.embed(
    context=context,
    inference_config=inference_config,
    strict=True,
)

operator_embedding = registry_result.embedding.embedding
```

`operator_embedding` is a one-dimensional NumPy array. With the default training
configuration, its dimension is `32`.

## Embedding Operators in One Query Plan

For one query plan record, use:

```python
result = embedder.embed_plan_record(
    plan_record,
    inference_config=inference_config,
    source_path="query.jsonl:1",
    strict=True,
)

matrix = result.embeddings
rows = result.rows
```

`matrix` has shape `(num_plan_nodes, embedding_dim)`. `rows` stores the mapping
from each row back to the plan node index, operator type, relation name, parent
operator, and embedding source.

## Batched Query-Plan Embedding

For multiple query plans, use the batched API:

```python
results = embedder.embed_plan_records(
    plan_records,
    inference_config=inference_config,
    source_paths=[f"query.jsonl:{i}" for i in range(len(plan_records))],
    strict=True,
)
```

The batched path groups nodes by operator type, runs each operator encoder in
batch, and scatters embeddings back to the original query-plan order.

## Output Artifacts

A successful pretraining run produces:

```text
runs/positive_anchor_key_mi_initialized/templates/*.pkl
runs/positive_anchor_key_mi_initialized/checkpoints/*/model.pt
runs/positive_anchor_key_mi_initialized/checkpoints/*/metadata.json
runs/positive_anchor_key_mi_initialized/checkpoints/*/manifest.json
runs/positive_anchor_key_mi_initialized/full_manifest.json
```

Keep these artifacts together. The manifest points to checkpoint directories,
and inference also needs the template directory used during training.
