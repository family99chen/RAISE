# Changelog

All notable changes to RAISE are documented here. Version numbers follow the git tags (`v1.0`, `v1.1`, `v1.2.0`).

## [1.2.0] - 2026-09-12

Judge timeouts on large QA sets, environment-based model routing, and clearer eval reports.

### Fixed

- **LLMAAJ queue-wait timeouts.** v1.1 submitted every QA at once and started the 120s clock at submit time. Later items on n500+ died in the thread-pool queue without calling the judge. Timeouts are now measured only while a QA is in flight (`src/raisex/core/eval_pool.py`).
- **Judge HTTP hang.** Judge calls used the generator YAML timeout (180s × 3 retries). They now use `eval.judge_http_timeout_seconds` (default 45s).
- **Empty Qwen replies.** Chat responses with empty `content` fall back to `reasoning` / `reasoning_content`. Qwen thinking is off unless `QWEN_ENABLE_THINKING=1`.

### Changed

- Eval pool: at most 8 in-flight judge calls; per-item wall clock 90s.
- Generator / rewriter / pruner / judge endpoints load from `.env` (`QWEN_*`, `CITYU_LLM_*`). Copy `.env.example` and fill your OpenAI-compatible servers.
- Default search space is the full text space (~199,680 configs when rewriter/reranker/pruner can be off). See `configs/algorithms/fullspace.yaml`.
- Package version in `pyproject.toml` is now `1.2.0` (was still `0.1.0` on v1.1).

### Added

- Report fields that separate a real judge 0 from a failed call: `LLMAAJ_status`, `LLMAAJ_judged_n`, `LLMAAJ_timeout_n`, `pipeline_status`. A timeout-only batch scores `null`, not `0`.
- `.env.example` for local secrets (`.env` stays gitignored).
- Pilot / main experiment notes: `docs/plan-01-pilot-ablation.md`, `docs/plan-02-main-and-followup.md`.

### Removed

- `experiments/run_four_algorithms.py` (use `run_five_algorithms.py`).

## [1.1.0] - 2026-09

Tagged as `v1.1 eval better` on `main`. Parallel LLMAAJ eval and per-item timeout (the submit-time clock that 1.2.0 fixes).

## [1.0.0] - 2026-09

Initial public toolkit: 17 search algorithms, text and multimodal pipelines, CLI / Python API.
