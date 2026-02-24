# 04 API Reference

## Import
```python
from raisex.api.public import (
    evaluate_rag,
    evaluate_rag_multimodal,
    check_config_valid,
    find_search_space,
    run_algorithms,
)
```

## Contract
| Function | Key Output Fields |
|---|---|
| `evaluate_rag` | `eval_report`, `outputs`, `chunking` |
| `evaluate_rag_multimodal` | `eval_report`, `outputs`, `chunking` |
| `check_config_valid` | `is_valid`, `errors` |
| `find_search_space` | `description`, `search_space`, `selection_template_text` |
| `run_algorithms` | `results` |
