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
  <img src="https://img.shields.io/badge/algorithms-14-blue?style=for-the-badge" alt="Algorithms">
  &nbsp;
  <img src="https://img.shields.io/badge/datasets-7-orange?style=for-the-badge" alt="Datasets">
  &nbsp;
  <img src="https://img.shields.io/badge/modalities-text%20%7C%20multimodal-blueviolet?style=for-the-badge" alt="Modalities">
</p>

---

**RAISE** is an open-source toolkit for automated hyper-parameter search and evaluation of Retrieval-Augmented Generation (RAG) pipelines. It provides a unified framework to systematically explore the configuration space of RAG systems — spanning chunking strategies, retrieval models, rerankers, pruners, and generators — and identify optimal configurations through a diverse suite of search algorithms.

**Key Features:**
- **14 search algorithms** covering Bayesian, evolutionary, bandit-based, local search, and reinforcement learning paradigms
- **Unified API & CLI** for seamless integration into research workflows
- **Text & multimodal pipelines** with pluggable components
- **7 built-in benchmarks** from widely-used QA datasets
- **Reproducible experiments** with layered YAML configuration and seed control

---

## Supported Algorithms (14)

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
<td rowspan="2"><b>Evolutionary</b></td>
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
