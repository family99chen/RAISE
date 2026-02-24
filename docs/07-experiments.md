# 07 Experiments

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
