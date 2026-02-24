# 09 Migration Guide

## Required Migration Steps
1. Switch API imports to `raisex.api.public`.
2. Switch CLI to `python -m raisex.cli.eval_cli` and `python -m raisex.cli.algo_cli`.
3. Switch data paths to `data/datasets/...`.
4. Switch config paths to `configs/...`.

## Verification
- New entry point commands execute successfully.
- Response structure fields remain consistent with pre-migration behavior.
