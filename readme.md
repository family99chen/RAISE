<p align="center">
  <img src="assets/RAISE_logo.png" width="500">
</p>

<h3 align="center">RAG Intelligence Search Engine</h3>
<h4 align="center">Hyper-Parameter Search & Evaluation Toolkit for Text and Multimodal RAG</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  &nbsp;
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
  &nbsp;
  <img src="https://img.shields.io/badge/algorithms-17-blue?style=for-the-badge" alt="Algorithms">
  &nbsp;
  <img src="https://img.shields.io/badge/datasets-7-orange?style=for-the-badge" alt="Datasets">
  &nbsp;
  <img src="https://img.shields.io/badge/modalities-text%20%7C%20multimodal-blueviolet?style=for-the-badge" alt="Modalities">
</p>

---

**RAISE** is an open-source toolkit for automated hyper-parameter search and evaluation of Retrieval-Augmented Generation (RAG) pipelines. It provides a unified framework to systematically explore the configuration space of RAG systems — spanning chunking strategies, retrieval models, rerankers, pruners, and generators — and identify optimal configurations through a diverse suite of search algorithms.

**Key Features:**
- **17 search algorithms** covering Bayesian, evolutionary, bandit-based, local search, and reinforcement learning paradigms
- **Unified API & CLI** for seamless integration into research workflows
- **Text & multimodal pipelines** with pluggable components
- **7 built-in benchmarks** from widely-used QA datasets
- **Reproducible experiments** with layered YAML configuration and seed control

---

## Supported Algorithms (17)

RAISE includes a comprehensive suite of hyper-parameter search algorithms organized into five categories:

<table>
<thead>
<tr>
<th>Category</th>
<th>Algorithm</th>
<th>Module</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><b>Bayesian / Model-based</b></td>
<td>TPE</td>
<td><code>tpe</code></td>
<td>Tree-structured Parzen Estimator</td>
</tr>
<tr>
<td>Cross-Entropy</td>
<td><code>cross_entropy</code></td>
<td>Cross-Entropy Method for combinatorial optimization</td>
</tr>
<tr>
<td rowspan="5"><b>Evolutionary</b></td>
<td>Regularized Evolution</td>
<td><code>regularized_evolution</code></td>
<td>Aging-based evolutionary search</td>
</tr>
<tr>
<td>GRPO</td>
<td><code>grpo</code></td>
<td>Group Relative Policy Optimization with neural policy</td>
</tr>
<tr>
<td>Doctor-GRPO</td>
<td><code>doctor_grpo</code></td>
<td>Stabilized GRPO profile with conservative clipping and stronger KL control</td>
</tr>
<tr>
<td>PPO</td>
<td><code>ppo</code></td>
<td>Clipped Proximal Policy Optimization with multi-epoch policy updates</td>
</tr>
<tr>
<td>Reinforce++</td>
<td><code>reinforce_pp</code></td>
<td>REINFORCE-style policy gradient profile with lightweight regularization</td>
</tr>
<tr>
<td rowspan="4"><b>Local Search</b></td>
<td>Greedy</td>
<td><code>greedy</code></td>
<td>Component-wise greedy selection</td>
</tr>
<tr>
<td>Coordinate Descent</td>
<td><code>coordinate_descent</code></td>
<td>Cyclic coordinate-wise optimization</td>
</tr>
<tr>
<td>Simulated Annealing</td>
<td><code>simulated_annealing</code></td>
<td>Metropolis–Hastings with cooling schedule</td>
</tr>
<tr>
<td>Iterative Local Search</td>
<td><code>iterative_local_search</code></td>
<td>Perturbation + local search with restarts</td>
</tr>
<tr>
<td rowspan="2"><b>Bandit-based</b></td>
<td>MAB-TS</td>
<td><code>mab_ts</code></td>
<td>Multi-Armed Bandit with Thompson Sampling</td>
</tr>
<tr>
<td>MAB-UCB</td>
<td><code>mab_ucb</code></td>
<td>Multi-Armed Bandit with Upper Confidence Bound</td>
</tr>
<tr>
<td><b>Resource-Adaptive</b></td>
<td>Successive Halving</td>
<td><code>successive_halving</code></td>
<td>Budget-aware early stopping</td>
</tr>
<tr>
<td rowspan="3"><b>Baselines</b></td>
<td>Random Search</td>
<td><code>randomalgo</code></td>
<td>Uniform random sampling baseline</td>
</tr>
<tr>
<td>Upper Bound</td>
<td><code>upperbound</code></td>
<td>Exhaustive evaluation oracle</td>
</tr>
<tr>
<td>Theoretical UB</td>
<td><code>thupperbound</code></td>
<td>Theoretical upper bound estimation</td>
</tr>
</tbody>
</table>

---

## Supported Datasets (7)

| Dataset | Source | Modality | QA Pairs | Corpus Docs |
|:--------|:-------|:--------:|:--------:|:-----------:|
| `triviaqa` | TriviaQA (Joshi et al.) | Text | 20 | 272 |
| `hotpotqa` | HotpotQA (Yang et al.) | Text | 20 | 791 |
| `msmarco` | MS MARCO (Nguyen et al.) | Text | 20 | 503 |
| `squadv2` | SQuAD v2 (Rajpurkar et al.) | Text | 20 | 300 |
| `longbench-multifield` | LongBench-MultiField (Bai et al.) | Text | 20 | 150 |
| `longbench-qasper` | LongBench-Qasper (Bai et al.) | Text | 20 | 20 |
| `scienceqa` | ScienceQA (Lu et al.) | Multimodal | 20 | 20 |

> Each dataset ships with a `qa_transfer.py` script that downloads and converts raw data from HuggingFace into the unified `qa.json` + `corpus.json` format.

---

## RAG Pipeline Components

RAISE decomposes a RAG pipeline into configurable stages, each with its own hyper-parameter search space:

```
Query ──► Rewriter ──► Chunking ──► Retriever ──► Reranker ──► Pruner ──► Generator ──► Answer
              │            │            │             │           │            │
              ▼            ▼            ▼             ▼           ▼            ▼
          prompt_id    chunk_size    model_url      model_url  prompt_id    model_url
                       chunk_overlap   topk           topk                  model_name
                                    bm25_weight
```

For **multimodal** pipelines, a CLIP-based image retrieval stage is added between Retriever and Reranker.

---

## Quick Start

### Installation
```bash
pip install -e .
```

### Evaluate a RAG Configuration

The last argument is `eval_mode`:
- **`avg`** — aggregated metrics only (compact, used by search algorithms)
- **`per_item`** — per-query scores only (for debugging individual questions)
- **`both`** — both aggregated + per-query (default, most informative)

```bash
# Text evaluation
python -m raisex.cli.eval_cli \
  data/datasets/triviaqa/qa.json \
  data/datasets/triviaqa/corpus.json \
  configs/demo.yaml both

# Multimodal evaluation
python -m raisex.cli.eval_cli multimodal \
  data/datasets/scienceqa/qa.json \
  data/datasets/scienceqa/corpus.json \
  configs/demo.yaml both
```

### Run Hyper-Parameter Search
```bash
# Single algorithm
python -m raisex.cli.algo_cli --algorithm greedy \
  --qa_json data/datasets/triviaqa/qa.json \
  --corpus_json data/datasets/triviaqa/corpus.json \
  --config_yaml configs/algorithms/default.yaml \
  --eval_mode avg

# Compare multiple algorithms
python -m raisex.cli.algo_cli --algorithms randomalgo,greedy,tpe,simulated_annealing \
  --qa_json data/datasets/triviaqa/qa.json \
  --corpus_json data/datasets/triviaqa/corpus.json \
  --config_yaml configs/algorithms/default.yaml \
  --eval_mode avg
```

### Python API
```python
from raisex.api.public import evaluate_rag, run_algorithms

# Single evaluation
result = evaluate_rag(
    qa_json_path="data/datasets/hotpotqa/qa.json",
    corpus_json_path="data/datasets/hotpotqa/corpus.json",
    config_path="configs/demo.yaml",
    eval_mode="both",
)
print(result["eval_report"])

# Run search algorithms
results = run_algorithms(
    qa_json_path="data/datasets/triviaqa/qa.json",
    corpus_json_path="data/datasets/triviaqa/corpus.json",
    config_path="configs/algorithms/default.yaml",
    algorithms=["greedy", "tpe", "grpo"],
)
```

---

## API Reference

All public functions are available via a single import:

```python
from raisex.api.public import (
    find_search_space,
    check_config_valid,
    evaluate_rag,
    evaluate_rag_multimodal,
    run_algorithms,
)
```

### `find_search_space`

Retrieve the full hyper-parameter search space, including all selectable values, prompt templates, and a ready-to-use YAML template.

```python
space = find_search_space(config_path=None, multimodal=False)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `config_path` | `str \| None` | `None` | Path to search space YAML. If `None`, resolves from env var `RAGSEARCH_CONFIG` or default `configs/search_space/text.yaml` |
| `multimodal` | `bool` | `False` | If `True`, use multimodal search space schema |

**Returns:** `Dict[str, Any]`

| Key | Type | Description |
|:----|:-----|:------------|
| `description` | `list[str]` | Human-readable description of the search space and its constraints |
| `search_space` | `dict` | Complete search space definition with all allowed values per component |
| `prompt_templates` | `dict` | Available prompt templates (keyed by ID) |
| `selection_template_text` | `str` | A ready-to-use YAML template string with default selections |
| `response_format` | `dict` | Expected response format hint |

<details>
<summary><b>Example</b></summary>

```python
space = find_search_space()
print(space["search_space"]["retrieve"])
# {'model_url': ['models/all-MiniLM-L6-v2'], 'topk': [1, 3, 5], ...}

print(space["selection_template_text"])
# rewriter:
#   model_url: "http://localhost:9000/v1"
#   prompt_template_id: "1"
# chunking:
#   chunk_size: 2048
# ...
```

</details>

---

### `check_config_valid`

Validate a user-provided selection YAML against the search space schema. Returns whether the config is valid and a list of error messages if not.

```python
result = check_config_valid(config_path, multimodal=False)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `config_path` | `str` | *(required)* | Path to the selection YAML file to validate |
| `multimodal` | `bool` | `False` | If `True`, validate against multimodal schema |

**Returns:** `Dict[str, Any]`

| Key | Type | Description |
|:----|:-----|:------------|
| `is_valid` | `bool` | `True` if the config passes all validation checks |
| `errors` | `list[str]` | List of validation error messages (empty if valid) |

<details>
<summary><b>Example</b></summary>

```python
result = check_config_valid("configs/demo.yaml")
if result["is_valid"]:
    print("Config is valid!")
else:
    for err in result["errors"]:
        print(f"  Error: {err}")
```

</details>

---

### `evaluate_rag` / `evaluate_rag_multimodal`

Run a full RAG pipeline (chunking → retrieval → reranking → generation → evaluation) with a given configuration and return metrics.

```python
result = evaluate_rag(qa_json_path, corpus_json_path, config_path, eval_mode="both")
result = evaluate_rag_multimodal(qa_json_path, corpus_json_path, config_path, eval_mode="both")
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `qa_json_path` | `str` | *(required)* | Path to QA dataset JSON (list of `{query, references}`) |
| `corpus_json_path` | `str` | *(required)* | Path to corpus JSON (list of `{id, content}`) |
| `config_path` | `str` | *(required)* | Path to selection YAML with chosen hyper-parameters |
| `eval_mode` | `str` | `"both"` | Evaluation mode: `"avg"`, `"per_item"`, or `"both"` (see below) |

**`eval_mode` options:**

| Value | Description |
|:------|:------------|
| `"avg"` | Returns only aggregated metrics averaged across all QA pairs (e.g. `ExactMatch=0.65, F1=0.72, ...`). Lightweight and sufficient for algorithm search where only the overall score matters. |
| `"per_item"` | Returns only per-query scores — a list where each entry contains individual metric scores, LLM-as-a-Judge reasoning, and the corresponding answer/references for that question. |
| `"both"` | Returns both aggregated metrics **and** per-query details. Use this when you need the overall score as well as the ability to inspect individual predictions. |

**Returns:** `Dict[str, Any]`

| Key | Type | Description |
|:----|:-----|:------------|
| `eval_report` | `dict` | Evaluation report containing `metrics` (aggregated scores) and/or `per_item` (per-query scores), depending on `eval_mode` |
| `outputs` | `list[dict]` | Per-query pipeline outputs including `answer`, retrieved contexts, etc. |
| `chunking` | `dict` | Chunking metadata |

If the config is invalid, returns `{"error": "invalid_config", "errors": [...]}` instead.

<details>
<summary><b>Example</b></summary>

```python
result = evaluate_rag(
    qa_json_path="data/datasets/hotpotqa/qa.json",
    corpus_json_path="data/datasets/hotpotqa/corpus.json",
    config_path="configs/demo.yaml",
    eval_mode="both",
)

if "error" in result:
    print("Config invalid:", result["errors"])
else:
    metrics = result["eval_report"]["metrics"]
    print(f"F1: {metrics.get('F1')}, ROUGE-L: {metrics.get('ROUGE-L')}")
```

</details>

---

### `run_algorithms`

Run one or more search algorithms as subprocesses and collect their results.

```python
results = run_algorithms(
    qa_json_path, corpus_json_path, config_path,
    algorithms=None, eval_mode="both", score_weights="", extra_args=None, cwd=None,
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `qa_json_path` | `str` | *(required)* | Path to QA dataset JSON |
| `corpus_json_path` | `str` | *(required)* | Path to corpus JSON |
| `config_path` | `str` | *(required)* | Path to algorithm config YAML |
| `algorithms` | `list[str] \| None` | `None` | Algorithm module names. If `None`, runs all 16 default algorithms |
| `eval_mode` | `str` | `"both"` | Evaluation mode |
| `score_weights` | `str` | `""` | Composite score weights (e.g. `"llmaaj1.0,bertf12.0,rougel1.5"`) |
| `extra_args` | `dict[str, list[str]] \| None` | `None` | Per-algorithm extra CLI args |
| `cwd` | `str \| None` | `None` | Working directory for subprocess execution |

**Returns:** `Dict[str, Any]`

| Key | Type | Description |
|:----|:-----|:------------|
| `results` | `list[dict]` | Per-algorithm results, each containing `algorithm`, `returncode`, `stdout`, `stderr`, `cmd` |

---

## Writing Custom Algorithms

RAISE is designed so you can easily write your own search algorithm. Your algorithm samples configurations from the search space, calls `evaluate_rag` to score them, and tracks the best result.

### Step 1: Import the environment functions

```python
from raisex.core.evaluation_service import evaluate_rag, evaluate_rag_multimodal
from raisex.api.public import find_search_space, check_config_valid
```

### Step 2: Get the search space

```python
space = find_search_space("configs/search_space/text.yaml")
search_space = space["search_space"]

# search_space is a nested dict, e.g.:
# {
#   "rewriter":  {"model_url": [...], "prompt_template_id": ["1","2","3"]},
#   "chunking":  {"chunk_size": [512, 1024, 2048], "chunk_overlap": [0, 128]},
#   "retrieve":  {"model_url": [...], "topk": [1,3,5,10], "bm25_weight": [0.0, 0.25, 0.5]},
#   "reranker":  {"model_url": [...], "topk": [1,3,5]},
#   "pruner":    {"model_url": [...], "prompt_template_id": [...]},
#   "generator": {"model_url": [...]},
# }
```

### Step 3: Sample a configuration and write it to YAML

```python
import random, yaml, tempfile

def sample_config(search_space):
    config = {}
    for component, params in search_space.items():
        config[component] = {}
        for param, values in params.items():
            config[component][param] = random.choice(values)
    return config

config = sample_config(search_space)

# Write to a temporary YAML file
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    yaml.safe_dump(config, f)
    config_path = f.name
```

### Step 4: Validate and evaluate

```python
# Optional: validate before evaluation
check = check_config_valid(config_path)
assert check["is_valid"], check["errors"]

# Evaluate the configuration
result = evaluate_rag(
    qa_json_path="data/datasets/triviaqa/qa.json",
    corpus_json_path="data/datasets/triviaqa/corpus.json",
    config_path=config_path,
    eval_mode="avg",
)

metrics = result["eval_report"]["metrics"]
score = metrics.get("F1", 0.0)
print(f"Score: {score}")
```

### Step 5: Loop and optimize

```python
best_score = -1
best_config = None

for trial in range(100):
    config = sample_config(search_space)
    # ... write to YAML, evaluate, track best ...
    if score > best_score:
        best_score = score
        best_config = config
```

> Refer to the built-in algorithms under `src/raisex/search/algorithms/` for complete, production-ready examples including argument parsing, progress tracking, and result reporting.

---

## Repository Layout

```
raise/
├── src/raisex/
│   ├── api/                  # Public Python API
│   ├── cli/                  # CLI entry points (eval_cli, algo_cli)
│   ├── core/                 # Config loading, validation, orchestration
│   ├── pipelines/
│   │   ├── text/             # Text RAG pipeline & evaluation
│   │   └── multimodal/       # Multimodal RAG pipeline & evaluation
│   ├── search/
│   │   └── algorithms/       # 14 search algorithm implementations
│   └── llmfactory/           # LLM provider abstraction (API / local)
├── configs/
│   ├── search_space/         # Hyper-parameter search space definitions
│   ├── algorithms/           # Algorithm run configurations
│   └── experiments/          # Reproducible experiment configs
├── data/
│   └── datasets/             # 7 built-in benchmark datasets
├── experiments/              # Large-scale benchmark scripts & analysis
├── docs/                     # Documentation (10 chapters)
├── pyproject.toml
└── requirements.txt
```

---

## Entry Points

| Interface | Command / Import | Description |
|:----------|:-----------------|:------------|
| **Python API** | `from raisex.api.public import ...` | Programmatic access to evaluation and search |
| **Eval CLI** | `python -m raisex.cli.eval_cli` | Evaluate a single RAG configuration |
| **Algo CLI** | `python -m raisex.cli.algo_cli` | Run one or more search algorithms |
| **Console** | `raise-eval` / `raise-algo` | Installed console entry points |

---

## Documentation

| Chapter | Topic |
|:--------|:------|
| [01-overview](docs/01-overview.md) | Architecture overview |
| [02-quickstart](docs/02-quickstart.md) | Installation & first run |
| [03-project-structure](docs/03-project-structure.md) | Module responsibilities |
| [04-api-reference](docs/04-api-reference.md) | Python API contract |
| [05-cli-reference](docs/05-cli-reference.md) | CLI usage & flags |
| [06-config-system](docs/06-config-system.md) | Configuration resolution |
| [07-experiments](docs/07-experiments.md) | Running benchmarks |
| [08-compatibility-matrix](docs/08-compatibility-matrix.md) | Supported interfaces |
| [09-migration-guide](docs/09-migration-guide.md) | Upgrading from legacy layout |
| [10-faq](docs/10-faq-troubleshooting.md) | FAQ & troubleshooting |

---

## License

This project is licensed under the [MIT License](LICENSE).
