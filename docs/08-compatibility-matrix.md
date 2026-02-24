# 08 Compatibility Matrix (Post-Cleanup)

This project currently supports only the new interface family.

## Supported Interfaces
| Interface Type | Supported Path |
|---|---|
| Python API | `raisex.api.public` |
| Evaluation CLI | `python -m raisex.cli.eval_cli ...` |
| Algorithm CLI | `python -m raisex.cli.algo_cli --algorithm ...` |
| Search space config | `configs/search_space/*` |
| Datasets | `data/datasets/*` |

## Removed Legacy Layer
- The legacy root-level entry layer has been fully retired.
- Duplicate legacy config copies have been cleaned up; only the versions under `configs/` are retained.
