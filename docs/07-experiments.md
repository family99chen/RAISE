# 07 Experiments

重投协议分两步，不要对调：

1. [规模消融，定 magic number](plan-01-pilot-ablation.md) — 题量/测试量/预算用 `10/50/100/500/1000`，五指标等权
2. [主实验与剩余消融](plan-02-main-and-followup.md) — 三个数锁定后再铺 7×13

Judge 密钥：复制 `.env.example` 为 `.env`，填 `CITYU_LLM_KEY`。

## Run
```bash
python experiments/run_five_algorithms.py \
  --qa_json data/datasets/triviaqa/qa.json \
  --corpus_json data/datasets/triviaqa/corpus.json \
  --config_yaml configs/algorithms/default.yaml \
  --output_root outputs-experiments
```

## Analyze
```bash
python experiments/analyze_five_algorithms.py \
  --results_root outputs-experiments/results \
  --output_dir outputs-experiments/analysis
```
