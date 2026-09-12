# 第二步：主实验与剩余消融

本文件在 [plan-01-pilot-ablation.md](plan-01-pilot-ablation.md) 写出三个 magic number 之后才执行。旧投稿的 100 题 / 30 trial 数字不进主表，只可作附录对照。

先填试点结果，再开跑：

| 符号 | 取值 | 依据（decision.json） |
|---|---|---|
| `n_search*` | _待填_ | A1 |
| `n_test*` | _待填_ | A2 |
| `B*` | _待填_ | A3 |

协议与第一步相同，中途不换：

- 目标：`rougel0.2,meteor0.2,f10.2,bleu0.2,llmaaj0.2`（搜索和 held-out 同一套）
- Judge：CityU DeepSeek-V4-Flash-FP8（`.env` 的 `CITYU_LLM_KEY`）
- Corpus 跟 QA 绑定：`n_search*` 题就用 `slices/search/n{n_search*}/` 的 qa+corpus，`n_test*` 题就用 `slices/heldout/n{n_test*}/` 的 qa+corpus。不要指回 1000 题全库。Multifield 用满官方 150。
- 种子：`11, 22, 33, 44, 55`
- 生成器：与第一步同一套，已冻结
- 搜索空间：完整文本空间（`configs/algorithms/fullspace.yaml`，约 10.4 万全开 / 20.0 万含模块开关）。没有特别说明都用这份；缩小空间只用于点名消融

主结论只看 **held-out**。Search 分只用来画搜索行为和 anytime 曲线。

---

## E1 主基准（必须重跑）

| 轴 | 取值 |
|---|---|
| 环境 | 6 个可扩（Hotpot / Trivia / MSM / SQuAD / Qasper / ScienceQA）+ Multifield 150 |
| 算法 | 下面 13 个 |
| 种子 | 5 |
| 搜索 | 每个 `(环境, 算法, seed)`：`n_search*` 题，预算 `B*` |
| 终评 | 同一条 run 的 best config，在该环境 `n_test*` 题 held-out 上再评一次 |
| Multifield | 只有官方 150 题、没有 held-out；格子照跑，表上单独标「无 held-out」 |

**主表 13 个算法**

| 族 | 名字 | CLI |
|---|---|---|
| 无信息 | Random、Greedy | `randomalgo`、`greedy` |
| 局部 / 坐标 | Coordinate Descent、ILS | `coordinate_descent`、`iterative_local_search` |
| 模型 | TPE | `tpe` |
| 分布 | CEM、SA、Regularized Evolution | `cross_entropy`、`simulated_annealing`、`regularized_evolution` |
| 多臂 | TS、UCB | `mab_ts`、`mab_ucb` |
| RL | GRPO、Dr.GRPO、R++ | `grpo`、`doctor_grpo`、`reinforce_pp` |

不进主表：`upperbound` / `thupperbound`（上界，不是搜索算法）、`successive_halving`、`ppo`。时间够可当附录，不要为了凑数加算法。

格子（不含 Multifield）：`6 × 13 × 5 = 390` 条搜索轨迹，再加 390 次 held-out 终评。Multifield 再加 `13 × 5` 条搜索。

对外可以这样写：390+ 条轨迹、每条 `B*` 次配置评估、每次评估 `n_search*` 题；主数字是 held-out `n_test*` 上的 5-seed 均值±区间。

**主表怎么读**

- 加粗 = 该环境观测均值最高，不是「唯一赢家」。
- 每个环境标出分不开的 top-tier（区间重叠就同层）。
- 预期读法仍是：没有跨环境唯一赢家，环境和算法有交互。若某方法全面碾压，先查是不是生成器/指标把黑盒变简单了，不要改口。

**Search vs held-out（几乎不新算）**

同一批 best config 已经有 search@`B*` 和 held-out@`n_test*`。报：

- 每个环境的排名 Spearman / 是否换人
- 「search 第一」在 held-out 上掉到第几

这张表是主结论的一部分：search 分不能代替测试分。

---

## E2 预算曲线（代表方法，同一条轨迹）

第一步 A3 只在 Hotpot、3 个算法上定了 `B*`。主稿还需要让人看见「排名随预算变」。

- 环境：Hotpot + ScienceQA（一个文本、一个多模态）
- 方法：Random、TPE、CEM、GRPO（四个族各一个）
- 做法：把这 4×2 的主表 seed 轨迹 **直接收到 1000**（或至少 `max(B*, 200)`），从同一条 trace 切 Best@10/50/100/500/1000
- 不要另开一套「预算消融」网格

若第一步 A3 已经在 Hotpot 上把 Random/TPE/CEM 收到 1000，E2 只补 GRPO 和 ScienceQA。Greedy 候选耗尽就停，写明结转。

---

## E3 模块选项频率（不新搜）

390 个 `best_config` 做成模块选项频率图：rewriter 模板、chunk、retriever topk / BM25、rerank-k、pruner 模板。回答的是「搜索最后爱挑什么」，不是再做一次模块开关消融。

完整模块 knockout（关 rewriter / 关 reranker / 关 pruner 再搜一遍）排最后。时间不够就砍，用频率图顶上。

---

## E4 任务专用列（同一批预测，不改搜索目标）

搜索目标全程仍是五指标等权。附表多两列，用已经生成的答案重算：

| 环境 | 加一列 | 目的 |
|---|---|---|
| ScienceQA | 选择题准确率 | 词法平均会被长解释带跑 |
| SQuAD v2 | 拒答 EM / 有答 F1 | 空 `references` = 应拒答 |

不要为这两列重开搜索。若它们和五指标平均把第一名换人，写进讨论，不要事后改主目标。

---

## E5 指标 leave-one-out（不新跑）

每份 report 已经有 5 个分量。事后重加权：

- 去掉 LLMAAJ（只留四词法）
- 去掉某一个词法
- 只看 LLMAAJ

看主表排名动了多少。这是「五等权不容易被 judge 单杀」的直接证据。若去掉 LLMAAJ 后排名大乱，正文如实写，不要把 judge 权重再改回 0.5。

---

## E6 旧 100 题对照（可选，附录）

若审稿人要新旧协议并排：用 **同一生成器、同一五指标** 在旧 100 题 proxy 上抽 3 个算法各 5 seed、预算 `B*`，只放附录。不要把旧表的冠军名字抄进新主表。

---

## 主稿表格

| 表 | 内容 | 预期读法 |
|---|---|---|
| A | 6×13 held-out 均值±区间 + Multifield 150 | 没有唯一赢家；环境 × 算法交互 |
| B | 同一批配置的 search@`B*` vs held-out@`n_test*` | search 分不能当最终成绩 |
| C | 2 环境 × 4 方法的 Best@10…1000 | 浅预算会换人；`B*` 是规则选的 |
| D | Hotpot 上 QA 与库绑定的 search / test 题量曲线（第一步的图） | `n_search*` / `n_test*` 不是圆整数 |
| E | 五指标 leave-one-out 排名变化 | 主目标不是单靠 judge |
| F | best_config 模块频率 | 搜索在选什么，不是再搜一次 |

---

## 建议排期

| 阶段 | 工作 | 完成标准 |
|---|---|---|
| 已在第一步 | 扩 1000、切前缀、A1/A2/A3、写出三个数 | `decision.json` 有值 |
| 主表文本 4 环境 | Hotpot / MSM / SQuAD / Trivia，13×5，然后 held-out | 表 A 左四列 |
| 主表长文 + 视觉 | Qasper、ScienceQA、Multifield | 表 A 齐 |
| 几乎不新算 | 表 B / E / F，SQuAD 拒答，ScienceQA 准确率 | 附表齐 |
| 代表轨迹加长 | E2：Hotpot+ScienceQA × 4 方法收到 1000 | 表 C |
| 可砍 | 模块 knockout、旧 100 题附录 | 时间不够就没有 |

不要在主表跑到一半换模型、换权重、换 `n_*`。换了等于作废。

---

## 命令骨架

`n_search*` / `n_test*` / `B*` 换成第一步的数。QA 和 corpus 用同一档切片，不要混。

```bash
cd /home/cz/RAISE
set -a && source .env && set +a

N_SEARCH=n_search*
N_TEST=n_test*
BUDGET=B*
WEIGHTS="rougel0.2,meteor0.2,f10.2,bleu0.2,llmaaj0.2"
CFG=configs/algorithms/default.yaml

# 搜索（直接调算法模块，report 才会落盘）
python -m raisex.search.algorithms.tpe \
  --qa_json data/benchmarks/hotpotqa/slices/search/n${N_SEARCH}/qa.json \
  --corpus_json data/benchmarks/hotpotqa/slices/search/n${N_SEARCH}/corpus.json \
  --config_yaml "$CFG" \
  --max_evals "$BUDGET" \
  --seed 11 \
  --score_weights "$WEIGHTS" \
  --report_path outputs-main/hotpotqa/tpe/seed_11/report.json

# 终评：同一 best_config，换 held-out 前缀（具体入口按现有 eval CLI）
# QA:     data/benchmarks/hotpotqa/slices/heldout/n${N_TEST}/qa.json
# Corpus: data/benchmarks/hotpotqa/slices/heldout/n${N_TEST}/corpus.json
```

批量可用 `experiments/run_five_algorithms.py`，默认权重已是五等权。跑主表前把 `--datasets` 改成新 benchmarks 路径，并把 `--budgets` / `--seeds` 设成 `B*` 和 `11,22,33,44,55`。不要沿用该脚本里旧的数据集默认列表。

---

## 明确不做

- 不把旧 100 题结论写进新主表。
- 不在主实验里再扫题量或预算；那是第一步的事。
- 不把 LLMAAJ 权重改回 0.5，也不改回纯词法。
- 不用 Multifield 假装有 held-out。
- 不为「数字更好看」中途换生成器。
- 不上 35B 把格子逼小。
