# 03 Project Structure

```text
raise/
├── src/raisex/
│   ├── api/
│   ├── core/
│   ├── cli/
│   ├── pipelines/
│   │   ├── text/
│   │   └── multimodal/
│   ├── search/
│   └── llmfactory/
├── configs/
├── data/datasets/
├── experiments/
└── docs/
```

| Module | Responsibility |
|---|---|
| `src/raisex/core` | Config loading, validation, evaluation orchestration, algorithm dispatch |
| `src/raisex/pipelines` | Text / multimodal pipelines |
| `src/raisex/search/algorithms` | Search algorithm implementations |
| `configs` | Search space and experiment configurations |
| `data/datasets` | Dataset resources |
| `experiments` | Large-scale comparative experiments and analysis |
