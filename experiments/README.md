# Five-Algorithm Benchmark (raise)

This folder provides a reproducible benchmark pipeline for:

- `grpo`
- `greedy`
- `tpe`
- `mab_ts`
- `random`

It follows the agreed protocol:

- Budgets: `120,300,600`
- Seeds: `11,22,33,44,55`
- Unified objective: same `--score_weights`
- GRPO main run uses `--reward_mode composite`
- Greedy is run once per seed, then truncated to budget in analysis

## 1) Run experiments

```bash
python experiments/run_five_algorithms.py \
  --qa_json data/datasets/triviaqa/qa.json \
  --corpus_json data/datasets/triviaqa/corpus.json \
  --config_yaml configs/algorithms/default.yaml \
  --output_root outputs-experiments \
  --budgets 120,300,600 \
  --seeds 11,22,33,44,55 \
  --eval_mode both \
  --score_weights "llmaaj1.0,bertf12.0,rougel1.5,f11.5,bleu0.5,em1.0"
```

Optional GRPO ablation runs at max budget:

```bash
python experiments/run_five_algorithms.py \
  --qa_json data/datasets/triviaqa/qa.json \
  --corpus_json data/datasets/triviaqa/corpus.json \
  --config_yaml configs/algorithms/default.yaml \
  --output_root outputs-experiments \
  --run_ablations
```

## 2) Analyze results

```bash
python experiments/analyze_five_algorithms.py \
  --results_root outputs-experiments/results \
  --output_dir outputs-experiments/analysis \
  --budgets 120,300,600 \
  --seeds 11,22,33,44,55
```

Generated artifacts:

- `per_run.csv`
- `table1_objective.csv`
- `table2_metrics.csv`
- `significance.csv`
- `anytime_budget_120.png`, `anytime_budget_300.png`, `anytime_budget_600.png` (if `matplotlib` installed)
- `summary.json`

## Notes

- Statistical test uses Wilcoxon if `scipy` is available; otherwise it falls back to paired sign test.
- Objective is taken from trial `reward` if present (GRPO composite), otherwise trial `score`.
