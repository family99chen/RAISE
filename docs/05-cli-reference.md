# 05 CLI Reference

## Eval CLI
```bash
python -m raisex.cli.eval_cli <qa_json> <corpus_json> <config_yaml> [eval_mode]
python -m raisex.cli.eval_cli multimodal <qa_json> <corpus_json> <config_yaml> [eval_mode]
```

## Algo CLI
```bash
python -m raisex.cli.algo_cli --algorithm <name> --qa_json ... --corpus_json ... --config_yaml ...
python -m raisex.cli.algo_cli --algorithms randomalgo,greedy --qa_json ... --corpus_json ... --config_yaml ...
```

## Common Parameters
- `--algorithm` (single algorithm)
- `--algorithms` (comma-separated multiple algorithms)
- `--qa_json`
- `--corpus_json`
- `--config_yaml`
- `--eval_mode`
- `--score_weights`
- `--verbose` (include report_path / stdout / stderr in output)

## Default Output and Exit Codes
- Default output: a JSON array where each element has the structure `algorithm + metrics + status`.
- `--algorithm` uses the same array format (length 1).
- Partial failure policy: if any algorithm fails, the CLI exits with code `1`, but continues to output results for the remaining algorithms.
