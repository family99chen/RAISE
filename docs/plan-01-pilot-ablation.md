# 第一步：规模消融（定 magic number）

只做一件事：在 **HotpotQA** 上，用三条**互不交叉**的一维扫描，定出后面主实验要用的三个数。

| 符号 | 含义 | 谁来定 |
|---|---|---|
| `n_search*` | 搜索时评多少道题 | A1 |
| `n_test*` | held-out 终评评多少道题 | A2 |
| `B*` | 每个 seed 试多少个配置 | A3 |

档位一律 **10 / 50 / 100 / 500 / 1000**。不要 50/100/200。

---

## 用哪个数据集

**只用 HotpotQA（官方 distractor，每题约 10 段维基，不是黄金支持句）。**

- 三个 magic number 都在这一套上定，不在 7 个环境上各扫一遍。
- 其他环境（Trivia / MSM / SQuAD / Qasper / ScienceQA / Multifield）这一步不跑。
- Multifield 官方只有 150 题，本来也扫不了 500/1000。

时间够，可以用 A1 已经搜出的 best config，在 SQuAD 或 MSM 的 held-out 上**只复核 A2**（不再搜）。那是附录，不是定 `n_search*` / `B*` 的依据。

---

## 库跟着题走

Hotpot 每题自带约 10 段上下文。**有多少题，就只用这些题自己的文**，不把小档硬检索大库。

先抽 1000 道 search 题当母池，再切前缀 `10 ⊂ 50 ⊂ 100 ⊂ 500 ⊂ 1000`。每一档 QA 和 corpus 成对：

```text
n=10   →  前 10 题  + 这 10 题的段落（约 100 段）
n=100  →  前 100 题 + 这 100 题的段落（约 1000 段）
n=1000 →  全部      + 1000 题的并集（约 10000 段）
```

小档库小，检索和建索引会快很多。冻一份 1000 题大库再评 10 题又慢，也说不清「稳了」到底是题多了还是库大了。主实验同样：`n_search*` 是 100，就用 100 题 + 100 题对应的 corpus。

held-out 一样：`n_test=50` 就用 held-out 前 50 题和它们自己的文。其他环境进主表时按各自的 `n*` 绑自己的 corpus。Multifield 用满官方 150。

---

## 三个数独立测，不是矩阵

不要 `5 档题 × 5 档预算 × 5 档测试`，更不要 13 个算法。那是 125 倍，跑不完。

每条扫描只动一个轴，另外两个冻死：

```text
           搜索题数 n_search     测试题数 n_test      预算 B
A1 定 n*        扫五档              不跑测试           冻 50
A2 定 test*     冻住（用 A1 的配置）    扫五档           不搜索
A3 定 B*        冻 100 或已定的 n*     不跑测试        一条轨迹收到 1000，切 Best@k
```

顺序：

1. **先 A1**。预算一律 50，只换「多少题 + 这些题自己的库」。
2. **再 A2**。不再搜，把 A1 的 best config 拿到 held-out 各档上评；每档用该档自己的 QA+corpus。
3. **A3 可以和 A2 并行**（A3 不看测试题数）。QA 和 corpus 冻在 100 题这一对（或 A1 已定的 `n_search*`），每个 `(算法, seed)` 只跑一条 1000 次的搜索，事后读 Best@10/50/100/500/1000。

算法只用三个族的代表：Random、TPE、CEM。种子 5 个。

生成器选定后冻结。不要用 DeepSeek Flash 当生成器，避免自己评自己。目标全程：

```text
rougel0.2,meteor0.2,f10.2,bleu0.2,llmaaj0.2
```

搜索空间默认是完整文本空间（`configs/algorithms/fullspace.yaml`，`plan01.yaml` / `default.yaml` 同内容）：全开约 **10.4 万**，加上 rewriter/reranker/pruner 可关约 **20.0 万**。后面实验没有特别说明都用这份，不要再用 216 点的冒烟档。`configforalgo_1k.yaml` / `configforalgo_10k.yaml` 只给点名的消融。

---

## 1. 先扩数据，再切前缀

磁盘上现在是 search 200 / held-out 500，不够 1000 档。

```bash
cd /home/cz/RAISE

# 可扩环境拉到 1000 / 1000。Multifield 仍是官方 150，没有 held-out。
# Qasper val 大约 1005 题，held-out 允许收到 ≥90% 即停，并在日志里 warning。
python experiments/prepare_benchmarks.py \
  --search_size 1000 \
  --heldout_size 1000 \
  --overwrite

# 切嵌套前缀；每一档写出自己的 qa + 对应 corpus
python experiments/slice_nested_sizes.py \
  --sizes 10,50,100,500,1000 \
  --splits search,heldout
```

之后成对使用，不要混档：

```text
QA:     data/benchmarks/hotpotqa/slices/search/n50/qa.json
Corpus: data/benchmarks/hotpotqa/slices/search/n50/corpus.json

QA:     data/benchmarks/hotpotqa/slices/heldout/n50/qa.json
Corpus: data/benchmarks/hotpotqa/slices/heldout/n50/corpus.json
```

---

## 2. Judge 环境

密钥放项目根目录 `.env`（已 gitignore）。从 `.env.example` 复制后填 `CITYU_LLM_KEY`：

```bash
# /home/cz/RAISE/.env
CITYU_LLM_KEY=你的key
CITYU_LLM_URL=https://llm.cs.cityu.edu.hk/v1
CITYU_LLM_MODEL=CS/DeepSeek-V4-Flash-FP8
```

代码会读这个文件；yaml 里不写 key。先打通再跑消融：

```bash
set -a && source /home/cz/RAISE/.env && set +a
curl https://llm.cs.cityu.edu.hk/v1/chat/completions \
  -H "Authorization: Bearer $CITYU_LLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "CS/DeepSeek-V4-Flash-FP8",
    "messages": [{"role": "user", "content": "你好，请用三句话介绍你自己。"}]
  }'
```

没 key 时 LLMAAJ 会直接报错，避免静默掉一个 0.2 权重、把平均算歪。

---

## 3. 三条扫描分别做什么

执行可以早停：A1 / A2 都先跑 10/50/100。100 已经满足第 5 节就补一档 500 作对照；100 仍吵必须上 500，必要时 1000。

### A1 只问：搜索要多少题

| 冻 | 动 |
|---|---|
| 数据集 = HotpotQA | `n_search ∈ {10,50,100,500,1000}`，每档用该档自己的 corpus |
| `B = 50` | |
| 3 算法 × 5 seed | |

格子：`3 × 5 × 5 × 50 = 3750` 次配评估。题级大约 `3 × 5 × 50 × (10+50+100+500+1000) = 1.245e6`。所以这一步不能铺 7 个环境。

看：每个 `(算法, n)` 的 5-seed 均值 / CV，以及三人谁高是否随 seed、随 n 对调。

### A2 只问：测试要多少题

不再搜索。把 A1 里每个 `(算法, seed)` 在 `n_search = 100`（若 A1 已锁定就用 `n_search*`）上的 best config 拿出来，在 held-out 各档前缀上评 `n_test ∈ {10,50,100,500,1000}`。**每一档用该档测试题自己的 corpus**，不要始终检索 held-out 1000 全库。

若想看「搜索题量会不会改变测试稳定性」，最多再加一档 `n_search = 500` 的配置，仍然不是矩阵：只是多评一组已经搜好的 config。

格子大约 `3 × 5 × (10+50+100+500+1000) ≈ 2.5e4` 题（一组 config 时）。必须做：审稿人问的是测试集是不是也太小。

看：三人排名从哪一档开始与 `n_test = 1000` 一致。

### A3 只问：要搜多少次

| 冻 | 动 |
|---|---|
| HotpotQA，QA+库冻在 n=100（或已定的 `n_search*`） | 一条轨迹收到 1000 次，事后切 Best@10/50/100/500/1000 |

每个 `(算法, seed)` **只开一条 job**。不要为 5 个 B 各搜一遍。

格子：`3 × 5 × 1000 × 100 = 1.5e6` 题级（n=100 时）。比 5 套独立预算便宜 5 倍。不要在 1000 题上再跑 1000 trial。

看：三人排名从哪一档开始与 Best@1000 一致。

---

## 5. 决策规则（写出 magic number 的依据）

三个数都取 **满足条件的最小档**。档位只允许落在 `{10,50,100,500,1000}`。

**`n_search*`**

1. 更吵的那个算法（Random / TPE 里 CV 更大的）在该档 `CV < 0.15`。
2. 该档上「谁高于 Random」在 5 个 seed 里至少 4 个一致，不再随 seed 对调。
3. 该档与下一档的胜者相同。若 100 过关、500 也过关，取 100。

**`n_test*`**

1. 三人排名与 `n_test = 1000` 相同。
2. 复合分相对 1000 档的绝对偏差 `< 0.02`，或 5-seed 标准误 `< 0.02`。
3. 同样取最小满足档。

**`B*`**

1. Best@B 的三人排名与 Best@1000 相同。
2. 更吵算法从 B 到下一档的增益 `< 0.01`。
3. 取最小满足档。经验预估仍是 50；规则说要 100 就写 100，不要为了好看压回去。

把判定表写进 `outputs-pilot-ablation/decision.json`，主实验文档只引用这三个数，不再改规则。

Hotpot 定完后，若还有时间：用同一套 best config 在 SQuAD 或 MSM 的 held-out 前缀上复核 `n_test*`（不再搜）。ScienceQA 是视觉任务，题量规则可以不同，但主表仍用同一组 magic number，并在文中写明「由文本试点选出，多模态未另扫」。

---

## 6. 费用和早停

| 扫描 | 量级 | 早停 |
|---|---|---|
| 扩数据 + 切前缀 | 一次，分钟到一小时（Trivia 最慢） | 无 |
| A1 | ~3750 次配置评估，~1.2e6 题 + 同等 judge | 先 10/50/100；100 仍吵再 500/1000 |
| A2 | ~5e4 题，不搜 | 必须五档都评，便宜 |
| A3 | ~1.5e6 题（n=100 时） | 用 anytime，不要 5 套预算 |

Judge 调用和生成次数同量级。评估缓存（`.eval_cache`）只在「同一配置被重复评」时救命，A1 不同 n 仍是不同 QA，缓存几乎帮不上。所以不要把 A1 铺到 7 个环境。

---

## 7. 建议命令骨架

先锁生成器（写入本次 manifest，之后所有命令用同一份 yaml）。然后：

```bash
cd /home/cz/RAISE
set -a && source .env && set +a

WEIGHTS="rougel0.2,meteor0.2,f10.2,bleu0.2,llmaaj0.2"
CFG=configs/algorithms/fullspace.yaml
OUT=outputs-pilot-ablation/hotpotqa

# A1 示例：search n=50，用这 50 题自己的库，预算 50
python -m raisex.search.algorithms.tpe \
  --qa_json data/benchmarks/hotpotqa/slices/search/n50/qa.json \
  --corpus_json data/benchmarks/hotpotqa/slices/search/n50/corpus.json \
  --config_yaml "$CFG" \
  --max_evals 50 \
  --seed 11 \
  --score_weights "$WEIGHTS" \
  --report_path "$OUT/A1/tpe/n50/seed_11/report.json"

# A3 示例：QA+库冻在 n=100，同一条轨迹收到 1000，事后切 Best@k
python -m raisex.search.algorithms.tpe \
  --qa_json data/benchmarks/hotpotqa/slices/search/n100/qa.json \
  --corpus_json data/benchmarks/hotpotqa/slices/search/n100/corpus.json \
  --config_yaml "$CFG" \
  --max_evals 1000 \
  --seed 11 \
  --score_weights "$WEIGHTS" \
  --report_path "$OUT/A3/tpe/n100/seed_11/report.json"
```

A2 是对已有 `best_config` 做 held-out 前缀评估，不要再开搜索。

旧脚本 `run_qa_size_ablation.py` / `run_seed_stability_size_sweep.py` 默认权重和档位已改成五等权 + `10,50,100,500,1000`，但它们仍指向旧 raw JSONL。**本轮以 `data/benchmarks/` + `slice_nested_sizes.py` 为准。**

---

## 8. 本步交付

1. `data/benchmarks/`：可扩环境 search/held-out 各约 1000，外加 slices。
2. `.env` 里可用的 `CITYU_LLM_KEY`，curl 烟测通过。
3. A1 / A2 / A3 的 report 与汇总表。
4. `decision.json`：`n_search*`、`n_test*`、`B*`，以及触发了第 5 节哪几条。

三个数进 [plan-02-main-and-followup.md](plan-02-main-and-followup.md)。未写出这三个数之前，不要开 7×13。
