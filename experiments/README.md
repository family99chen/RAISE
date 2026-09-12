# Experiment runners

Keep this folder small. The live entry points are:

- `prepare_benchmarks.py` — build `data/benchmarks/` (search 200 / held-out 500)
- `run_five_algorithms.py` / `analyze_five_algorithms.py` — main algorithm comparison
- `run_qa_size_ablation.py` — QA-size curve on one dataset
- `run_seed_stability_size_sweep.py` — seed variance vs QA size
- `run_paired_random_average_ablation.py` — search vs random-mean, paired by seed
- `run_text_benchmarks.py` / `run_text_benchmarks_generated.py` — rebuild-and-run helpers

## Corpus rule

- Everyday search/eval: each question keeps its own attached context. Corpus grows with QA count.
- QA-size sweeps: freeze corpus at the **largest** QA size. A 20-QA or 100-QA run still retrieves from the 200-QA collection.

## Example

```bash
python experiments/run_five_algorithms.py \
  --qa_json data/datasets/triviaqa/qa.json \
  --corpus_json data/datasets/triviaqa/corpus.json \
  --config_yaml configs/algorithms/default.yaml \
  --output_root outputs-experiments \
  --budgets 50 \
  --seeds 11,22,33,44,55 \
  --eval_mode both
```
