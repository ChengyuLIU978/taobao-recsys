# CURRENT PROJECT STATUS — Taobao Recommender System

> 状态日期：2026-10-08  
> 项目目录：`taobao-recsys`  
> 当前阶段：**PROJECT FROZEN — interview/recruiting-ready baseline**  
> 下一里程碑：**先建立新的 temporal validation fold；未经新验证协议，不继续根据当前 test 优化 GenRec 或 Ranker**  
> 本文档依据：Notebook 中已执行 cell 与保存输出、已有数据文件、checkpoint、导出 embedding、mapping 元数据和实验结果文件。README、注释和文件名只作为辅助，不作为完成依据。

---

## 1. 一句话结论

项目已完成传统 Recall → Rank、Cold/Long-tail 诊断，以及一条完整的 TIGER-style collaborative Semantic-ID 生成式召回实验链路。完整 302,016-item V1 catalog 经 RQ-VAE 得到三层 codes，三层 utilization 均为 100%，suffix 后 final SID collision=0；Transformer + trie-constrained beam search 的 valid generation rate=100%。GenRec validation Recall@20=`0.063319`，为固定 union 新增 2 个 Head targets，oracle `0.384716→0.386426`；但一次性 test Recall@20/@50 均为 `0.042458`，相对 ItemCF 和 current union 的新增 target 都是 0，Tail 增量也是 0。因此该实验达到工程成功与非零单路召回，但当前证据不支持把它加入正式第四路 recall。它仍是 closed-catalog collaborative 方法，不能解决 `962/2512=38.30%` 的 truly train-cold targets。

最终项目封装已完成：README、技术报告、双语简历 bullet、44 问面试速查表、依赖与 Python 版本、Git 忽略规则、artifact manifest、机器可读 summary 和实验冻结声明均已落盘并验证。最终回归测试为 `28/28`，用时 `3.403s`；关键 JSON 可解析，README 内部链接与文档化命令语法检查全部通过。当前项目停止基于既有 TEST 的模型开发，可作为 GitHub/简历/面试展示基线。

### Final packaging status

| Deliverable | Status |
|---|---|
| `README.md` | DONE |
| `reports/FINAL_PROJECT_REPORT.md` | DONE |
| `reports/RESUME_BULLETS.md` | DONE |
| `reports/INTERVIEW_CHEATSHEET.md` | DONE — 44 questions |
| `requirements.txt` / `.python-version` | DONE — Python 3.13.9 |
| `.gitignore` | DONE — raw data and large binaries excluded |
| `artifacts/ARTIFACT_MANIFEST.md` | DONE |
| `artifacts/final/project_summary.json` | DONE / JSON validated |
| `EXPERIMENT_FREEZE.md` | DONE |
| Regression tests | DONE — 28/28 passed |

---

## 2. 当前真实进度总览

状态定义：

- **已完整完成并运行验证**：代码已实际执行，且有保存输出或产物可核对。
- **有代码但未确认运行**：存在实现，但没有可靠执行输出或产物证明。
- **只完成一部分**：关键环节已做，但仍缺少组成完整功能的步骤。
- **尚未开始**：没有发现有效实现或运行结果。

| Pipeline 环节 | 状态 | 当前实际完成内容 |
|---|---|---|
| 数据清洗与划分 | 已完整完成并运行验证 | 全量原始 CSV 被扫描；按用户做确定性约 1% 开发样本；生成 train / validation / test 与 purchase ground truth。 |
| Popularity | 已完整完成并运行验证 | 已构建推荐结果，并在统一 warm-start purchase evaluation set 上计算 Recall / HitRate / NDCG。 |
| ItemCF | 已完整完成并运行验证 | 已构建 ItemCF 召回，并在相同协议上完成 K=10/20/50 评估。 |
| Negative Sampling | 已完整完成并运行验证 | V0 使用 uniform random negatives；新实验采用每个正样本 4 个 uniform negatives；固定验证 negatives 会排除已知正样本。 |
| Dataset / DataLoader | 已完整完成并运行验证 | V0、Buy-only 与 Multi-behavior 训练都已实际跑完；支持固定 epoch step budget 和 weighted sampling。 |
| Two-Tower | 已完整完成并运行验证 | ID-only 双塔已训练；完成 PV-only V0、Buy-only V1、Multi-behavior V2。 |
| Training / Validation | 已完整完成并运行验证 | 三组实验各 3 epochs；V1/V2 使用固定 validation negatives 作 loss 诊断，并以 exact Recall@50 选最佳 checkpoint。 |
| Embedding 导出 | 已完整完成并运行验证 | 已导出 V0 user/item embeddings；shape、dtype、L2 norm 已检查。V1/V2 的 checkpoint 可恢复模型，实验评估时已生成 embedding。 |
| Top-K Retrieval | 已完整完成并运行验证 | 已实现 chunked exact inner-product Top-K；未一次性创建完整 `[9836, 302016]` score matrix；小样本与 dense exact 结果完全一致。 |
| Recall@K / HitRate@K / NDCG@K | 已完整完成并运行验证 | Popularity、ItemCF、V0、V1、V2 已在相同 warm-start purchase protocol 下计算 K=10/20/50 指标。 |
| FAISS | 只完成一部分 | 已规划 IndexFlatIP exact baseline；当前环境未安装 `faiss/faiss-cpu`，没有实际 FAISS 结果。未做 IVF/HNSW。 |
| Ranking | 已完整完成并运行验证 | 已用 validation candidates + purchase labels 训练 LightGBM LambdaRank；52 个特征，所有 aggregate 严格来自 train；完成 test 评估、importance 和一次轻量 ablation。 |
| Recall → Rank 完整 pipeline | 已完整完成并运行验证 | 已跑通 Recall → Candidate Union → train-only Features → Ranker → Final Top-K → Test Evaluation；test labels 只在打分完成后读取。 |
| Cold Start | 已完成分层评估 | 已按 train user/item 冷暖四象限、TT vocabulary 和 ItemCF catalog 完成 TEST 诊断；已计算 closed-catalog ceiling。这是评估，不是 Cold Start 解决方案。 |
| Long Tail | 已完整完成分层评估 | 已以 train-only item-count percentile 建立 Head/Torso/Tail，完成 target distribution、segment metrics/oracle、exposure/coverage、novelty、Gini、user-level exposure 和 bias amplification。 |
| Multi-stage Recall | 已完整完成并运行验证 | 已生成 validation/test 三路 candidate union；完成去重、source provenance、RRF、ItemCF-first、oracle、overlap、exclusive contribution、fallback 和 cold-target 统计。 |
| Controlled experiments | 已完整完成并运行验证 | 已建立共享 mapping、固定训练预算、统一 full-catalog purchase evaluation 的实验框架，并跑完 Buy-only、Multi-behavior 及一次 seed2027 Buy-only 复验。 |
| RQ-VAE | 已完整完成并运行验证 | 完整 302,016-item V1 embedding catalog；best epoch=15；validation reconstruction=0.00188850；三层 codebook utilization 均为 100%。 |
| Semantic IDs | 已完整完成并运行验证 | 生成 `(c0,c1,c2,suffix)`；raw tuple collision excess rate=17.6289%，deterministic suffix 后 302,016 个 final SID 全部唯一且 round-trip 成功。 |
| Generative Transformer | 已完整完成并运行验证 | 12,207 个严格因果 train purchase examples；behavior-aware history；best epoch=10，由 validation Recall@20/NDCG@20 选择。 |
| Generative Retrieval | 已完整完成并运行验证 | 完整 catalog trie、beam=50；validation/test valid generation rate=100%，无 invalid SID 或重复 item。 |
| Generative Incremental Recall | 已完整完成并运行验证 | Validation 新增 2 个 union-miss Head targets；test 新增 0，Tail 新增 0。保留为实验结果，不加入正式 candidate union。 |
| 实验问题记录 | 已完整完成并运行验证 | `PROJECT_ISSUES.md` 已记录实际问题、证据、根因、处理决定、结果和 trade-off。 |

---

## 3. 重要文件与目录

### 3.1 Notebooks

| 文件 | 作用 | 实际状态 |
|---|---|---|
| `notebooks/01_data_audit.ipynb` | 原始行为数据的初步结构、类型、行为与质量检查 | 已运行，但主要基于前 100,000 行，不代表完整数据统计。 |
| `notebooks/02_build_dev_dataset.ipynb` | 扫描完整原始数据；确定性抽取开发用户；按时间切分 train / validation / test；建立 purchase GT | 已运行并产出数据文件。 |
| `notebooks/03_build_recall_baselines.ipynb` | 构建 Popularity 与 ItemCF；计算 Recall / HitRate / NDCG | 已运行并有输出。 |
| `notebooks/04_build_two_tower.ipynb` | 构建 V0 ID-only Two-Tower；恢复严格 mapping；导出 embedding；执行 exact Top-K；与 baselines 公平比较 | 已运行；V0 checkpoint 未被覆盖。 |
| `notebooks/05_two_tower_controlled_experiments.ipynb` | 受控实验框架；行为统计；Buy-only V1；Multi-behavior V2；统一评估与集中度比较 | 已完整运行，8 个标记代码 cell 连续执行，无保存的错误输出。 |

### 3.2 Scripts

| 文件 | 作用 | 实际状态 |
|---|---|---|
| `scripts/06_repeat_buy_seed.py` | 独立重建 V1 数据、模型与评估协议；执行 seed2027 单次复验；保护历史产物；保存并验证结果 | 已通过 preflight 并完整运行；不依赖 Notebook 内存。 |
| `scripts/07_build_multistage_recall.py` | 从磁盘恢复三路召回，生成 validation/test candidates，融合、评估并保存 artifact | 已完整运行；候选用户来自 split interaction 文件，labels 只在候选落盘后用于评估。 |
| `tests/test_multistage_recall.py` | 验证去重、source flags、RRF、1-based ranks 与无 label 输入的 candidate-generation 接口 | 5 个测试全部通过。 |
| `scripts/08_train_ranker.py` | 构造 train-only ranking features；用 validation labels 训练 LambdaRank；在读取 test labels 前完成 test scoring；评估并保存端到端 artifact | 已完整运行；使用事务式 staging，不覆盖 recall/Two-Tower 历史产物。 |
| `scripts/run_end_to_end.py` | `--reuse-retrievers` 薄编排入口；复用完整 Recall/Ranking artifacts，缺失时才运行对应脚本，遇到部分目录则拒绝覆盖 | 已以 reuse-only 模式运行，成功重载模型和两份 ranking Parquet，未重训 retriever。 |
| `tests/test_ranking_artifacts.py` | 独立回读检查时间边界、label 访问契约、group 对齐、候选唯一性、模型重载和指标完整性 | 5 个测试全部通过。 |
| `scripts/09_analyze_cold_start_long_tail.py` | 从磁盘复用固定 TEST candidates/Top-50，构造 train-only popularity segments 并完成 cold/long-tail/exposure/novelty/oracle 诊断 | 已完整运行；没有加载训练模型、调参或修改历史 artifact。 |
| `tests/test_cold_long_tail_analysis.py` | 验证 cold 四象限、Head/Torso/Tail partition、cold 不进 Tail、train-only popularity、segment denominator 和 exposure 守恒 | 7 个测试全部通过；加上旧回归测试共 17/17 通过。 |
| `scripts/10_export_v1_item_embeddings.py` | Strict-load V1 checkpoint，按持久化 mapping 导出 normalized item-tower embeddings | 已运行；shape=`(302016,64)`，20 个 mapping 对位与 hash 检查通过。 |
| `scripts/11_train_rqvae.py` | Deterministic train/validation item split、MiniBatchKMeans 初始化、三层 residual quantization | 已完整训练 15 epochs 并保存 best checkpoint/history/metrics。 |
| `scripts/12_build_semantic_ids.py` | 全 catalog quantization、collision suffix、双向 lookup、prefix similarity 与 segment code diagnostics | 已运行；final SID collision=0。 |
| `scripts/13_train_generative_retriever.py` | 构建严格因果 behavior-aware purchase sequences，训练小型 encoder-decoder Transformer，仅用 validation 选 checkpoint | 已运行；best epoch=10，test 在 selection 阶段未读取。 |
| `scripts/14_evaluate_generative_recall.py` | 冻结后执行 test label-free generation、一次性 test evaluation、incremental oracle/exposure/failure analysis | 已完成；test 配置没有用于回头调参。 |
| `scripts/run_generative_pipeline.py` | 生成式全流程入口；检测到 completed summary 后拒绝重训或重复 test evaluation | 已运行完成态保护检查。 |
| `tests/test_rqvae.py` / `test_semantic_ids.py` / `test_generative_retrieval.py` | STE、codebook gradient、SID 唯一性、trie、Transformer、beam、leakage 与 checkpoint reload | 新增 11 项；连同传统主线回归共 28/28 通过。 |

### 3.3 文档

| 文件 | 作用 |
|---|---|
| `PROJECT_ISSUES.md` | 项目真实问题清单；当前记录 28 类问题，包含传统主线、cold/long-tail 与生成式实验的真实成功、失败和边界。 |
| `CURRENT_PROJECT_STATUS.md` | 本文档；用于快速判断当前进度、已有证据、风险和下一步。 |

### 3.4 Two-Tower 核心产物

目录：`artifacts/two_tower/`

| 产物 | 说明 |
|---|---|
| `two_tower_best.pt` | 原始 PV-only V0 checkpoint；3 epochs；审计和后续实验中未覆盖。 |
| `user_mapping.csv` | 训练词表中 user ID → embedding index 映射。 |
| `item_mapping.csv` | 训练词表中 item ID → embedding index 映射。 |
| `mapping_metadata.json` | mapping 重建规则、数量、训练文件哈希等元数据。 |
| `user_embeddings.npy` | V0 user embeddings，shape `(9836, 64)`。 |
| `item_embeddings.npy` | V0 item embeddings，shape `(302016, 64)`。 |

Mapping 已从训练代码的实际规则恢复并验证：

1. 只使用 train 中的 `pv` 行。
2. 选择 `user_id, item_id`。
3. 按首次出现顺序去重。
4. 使用 pandas `unique()` 的稳定顺序建立 index。
5. 恢复得到 `users = 9,836`、`items = 302,016`，与 checkpoint 完全一致。

训练文件 SHA-256：

```text
c8a702804ada2e1da7e7e448db71aabc86afa0e1336ac7147bcd40e8c5ad9148
```

原始 V0 checkpoint SHA-256：

```text
D3CD9C94C0033318C94436F0575A49AD8048F48B4FFF40251601482E12E7E78A
```

### 3.5 Controlled experiment 产物

目录：`artifacts/two_tower/experiments/`

| 产物 | 内容 |
|---|---|
| `behavior_statistics.csv` | 每种行为的数量、用户/物品覆盖、未来 purchase overlap、lift 与权重。 |
| `controlled_metrics.csv` | V0、V1、V2 在 K=10/20/50 的统一指标。 |
| `controlled_concentration.csv` | 各模型的 Top-1 与 Top-50 推荐集中度。 |
| `controlled_coverage.csv` | 统一 warm-start 评估覆盖信息。 |
| `TT_V0_PV_UNIFORM/` | V0 的实验记录、指标和集中度；引用原 checkpoint，不覆盖它。 |
| `TT_V1_BUY_UNIFORM/` | Buy-only checkpoint、config、history、metrics、集中度与样例推荐。 |
| `TT_V2_MULTI_UNIFORM/` | Multi-behavior checkpoint、config、history、metrics、集中度与样例推荐。 |
| `TT_V1_BUY_UNIFORM_SEED2027/` | Buy-only alternate-seed 的 `best.pt`、config、完整 history、K=10/20/50 metrics、集中度与验证摘要。 |

V1/V2 及 seed2027 replication checkpoint 均保存对应 config、mapping reference、最佳 epoch 和检索指标，并已用 strict state loading 验证可读取。

### 3.6 Multi-stage Recall 产物

目录：`artifacts/multistage_recall/`

| 产物 | 内容 |
|---|---|
| `validation_candidates.parquet` | 9,719 个 validation users 的 3,304,863 条去重候选。 |
| `test_candidates.parquet` | 9,688 个 test users 的 3,294,775 条去重候选。 |
| `validation_metrics.json` | Validation source、oracle、RRF 与 ItemCF-first 指标。 |
| `test_metrics.json` | Test 最终评估；没有用于调整 quota、RRF 或规则。 |
| `source_overlap.json` | Validation/test 三组 pairwise candidate overlap。 |
| `source_contribution.json` | Ground-truth hit 的 exclusive source combination 与 TT 独立贡献。 |
| `candidate_statistics.json` | 候选规模、fallback、构建统计和所有 validation checks。 |
| `config.json` | 固定 quotas、RRF=60、路径、mapping、协议和 no-leakage 声明。 |
| `sample_candidates.csv` | 5 个随机 warm users 的人工检查样例。 |
| `execution_summary.json` | Parquet 回读、baseline reproduction 与历史 artifact 哈希验证。 |

### 3.7 Ranking 与端到端产物

目录：`artifacts/ranking/`

| 产物 | 内容 |
|---|---|
| `validation_ranking_dataset.parquet` | 3,304,863 条 validation candidates；含 52 个模型特征、validation label 和 training-fit score。 |
| `test_ranking_dataset.parquet` | 3,294,775 条 test candidates；在 ranker 打分完成后才附加离线 label。 |
| `ranker.pkl` / `ranker.txt` | 完整 52-feature LightGBM LambdaRank 模型；已回读并验证预测一致。 |
| `ranker_recall_only.pkl` / `.txt` | 预先声明的 recall-only 轻量 ablation 模型。 |
| `feature_manifest.json` | 52 个特征的 dtype、source、train-only 标志、说明和 missing-value policy。 |
| `config.json` | 后端、固定参数、label 访问顺序、类别规则、评估协议和输入哈希。 |
| `test_metrics.csv` | Popularity、Two-Tower、ItemCF、RRF、ItemCF-first 和 Ranker 的 test warm @10/@20/@50 指标。 |
| `test_recommendations.parquet` | 全部 test users 的 Ranker Top-50，含 score、source ranks、source count 和离线 label。 |
| `feature_importance.csv` | Gain importance、split count、importance fraction 与排名。 |
| `ablation_metrics.csv` | Recall-only 与全特征 ranker 在 test warm @20/@50 的固定对比。 |
| `summary.json` | 训练统计、validation fit 诊断、test 指标、oracle gap、leakage checks 和 bottleneck 结论。 |
| `sanity_sample.csv` | 固定 seed 抽取 5 个 warm test users 的 Top-10 人工检查样例。 |

### 3.8 Cold Start / Long Tail 分析产物

目录：`artifacts/analysis/`

| 产物 | 内容 |
|---|---|
| `cold_start_metrics.csv` / `cold_start_summary.json` | 四象限 bucket 统计、四模型指标、train/TT/ItemCF catalog cold 统计和 cold-user fallback 结论。 |
| `item_popularity_segments.parquet` | 317,721 个 train-known items 的 train behavior counts、rank、percentile 和 Head/Torso/Tail segment。 |
| `test_purchase_segments.parquet` | 2,512 个 TEST purchase pairs 的 cold flags、bucket、segment 及 source/oracle/ranker hit flags。 |
| `long_tail_target_distribution.csv` / `long_tail_metrics.csv` | Head/Torso/Tail/Cold demand 分布与按 segment 的四模型指标。 |
| `segment_oracle_metrics.csv` | Head/Torso/Tail union oracle、Ranker@50 和 segment oracle gap。 |
| `exposure_metrics.csv` / `novelty_metrics.csv` | ItemCF/Two-Tower/Ranker Top-50 exposure、coverage、popularity percentile、Gini 和 novelty。 |
| `user_exposure_distribution.csv` | 每个 TEST user 的 Head/Torso/Tail Top-50 exposure fraction。 |
| `popularity_bias_amplification.csv` | train-known TEST target demand share 与 recommendation exposure share 之差。 |
| `two_tower_segment_contribution.csv` | TT Top-100 相对 ItemCF Top-200 的 segment-level exclusive target hits。 |
| `closed_catalog_ceiling.json` / `analysis_summary.json` | closed-catalog ceiling、all-GT 比例、全部检查与分析结论。 |

### 3.9 Experimental Generative Retrieval 产物

目录：`artifacts/generative/`

| 产物 | 内容 |
|---|---|
| `embedding_manifest.json` / `v1_item_embeddings.npy` / `item_ids.npy` | V1 checkpoint/mapping hash、strict-load 证明与 302,016×64 normalized embeddings。 |
| `rqvae_config.json` / `rqvae_history.csv` / `rqvae_best.pt` / `rqvae_metrics.json` | 固定 RQ-VAE config、15 epochs 历史、best checkpoint、reconstruction 与三层 codebook diagnostics。 |
| `semantic_ids.parquet` / `semantic_id_lookup.npz` / `semantic_id_stats.json` | raw codes、deterministic suffix、双向映射、collision、segment utilization 与 prefix-similarity diagnostic。 |
| `token_manifest.json` | PAD/BOS/EOS/SEP、behavior、三层 code 与 suffix 的互不重叠 token ranges。 |
| `train_sequences.npz` / `validation_histories.npz` / `test_histories.npz` | purchase-aligned causal sequences 与 train-only history；test histories 在 checkpoint freeze 后构建。 |
| `generator_config.json` / `generator_history.csv` / `generator_best.pt` | 固定 Transformer config、10 epochs CE/retrieval history 与 validation-selected checkpoint。 |
| `validation_generative_metrics.json` / `test_generative_metrics.json` | 单路 Recall/HitRate/NDCG、validity、segment、incremental oracle 和 exposure。 |
| `validation_generated_recommendations.parquet` / `test_generated_recommendations.parquet` | beam=50 合法 Semantic-ID 推荐，含 beam score、raw SID、segment 与 evaluation-only `is_gt`。 |
| `incremental_recall_analysis.csv` / `segment_incremental_recall.csv` | validation/test 相对 ItemCF 与 current union 的增量命中和 Head/Torso/Tail oracle。 |
| `failure_analysis.csv` | 固定 seed 的 ItemCF-miss 案例，包括 GenRec hit/miss、history、Top-10 与 SID prefix。 |
| `generative_summary.json` / `final_summary.json` / `pre_test_freeze.json` | 完整机器可读结论、artifact paths、test-once gate 与 28/28 测试状态。 |

---

## 4. 数据处理与评估集合

### 4.1 数据规模

- 原始行为 CSV：约 3.67 GB。
- 原始行数：100,150,807。
- 开发集抽样规则：`user_id % 100 == 0`，约 1% 用户，确定性可复现。
- 开发集总行数：1,001,832。
- Train：724,885 行。
- Validation：138,406 行。
- Test：138,541 行。
- Validation purchase GT：2,540 个 interaction，1,760 个用户。
- Test purchase GT：2,512 个 interaction。

### 4.2 Warm-start validation purchase evaluation

Two-Tower 当前是纯 ID 模型，因此正式横向比较只对 user 与 target item 均在 train-PV vocabulary 中的 validation purchases 评估。

| 统计项 | 数值 |
|---|---:|
| GT users | 1,760 |
| GT interactions | 2,540 |
| Unseen users | 16 |
| Unseen target interactions | 1,032 |
| Unique unseen target items | 1,023 |
| 实际 warm evaluation users | 1,170 |
| 实际 warm evaluation interactions | 1,496 |
| User coverage | 66.48% |
| Interaction coverage | 58.90% |

这表示当前公开的 Two-Tower 指标只描述 warm-start 子集，不代表全体 validation purchase 表现。约 41.10% 的 validation purchase interactions 因目标 item 未在 train-PV vocabulary 中而无法由当前 ID-only 模型检索。

---

## 5. Baseline 与模型结果

所有下列指标都在同一个 warm-start purchase evaluation set 上重新计算。三列依次为 Recall、HitRate、NDCG。

| Model | K | Recall | HitRate | NDCG |
|---|---:|---:|---:|---:|
| Popularity | 10 | 0.002991 | 0.003419 | 0.000992 |
| Popularity | 20 | 0.004701 | 0.005128 | 0.001405 |
| Popularity | 50 | 0.007123 | 0.010256 | 0.002016 |
| ItemCF | 10 | 0.108468 | 0.129060 | 0.051385 |
| ItemCF | 20 | 0.189158 | 0.217094 | 0.072717 |
| ItemCF | 50 | **0.371696** | **0.405983** | **0.110981** |
| TT_V0_PV_UNIFORM | 10 | 0.002564 | 0.002564 | 0.000920 |
| TT_V0_PV_UNIFORM | 20 | 0.002991 | 0.003419 | 0.001066 |
| TT_V0_PV_UNIFORM | 50 | 0.003846 | 0.004274 | 0.001221 |
| TT_V1_BUY_UNIFORM | 10 | 0.004274 | 0.005128 | 0.001903 |
| TT_V1_BUY_UNIFORM | 20 | 0.004274 | 0.005128 | 0.001903 |
| TT_V1_BUY_UNIFORM | 50 | **0.011966** | **0.012821** | **0.003453** |
| TT_V1_BUY_UNIFORM_SEED2027 | 10 | 0.000855 | 0.000855 | 0.000304 |
| TT_V1_BUY_UNIFORM_SEED2027 | 20 | 0.002564 | 0.003419 | 0.000746 |
| TT_V1_BUY_UNIFORM_SEED2027 | 50 | **0.008405** | **0.010256** | **0.001927** |
| TT_V2_MULTI_UNIFORM | 10 | 0.000078 | 0.000855 | 0.000054 |
| TT_V2_MULTI_UNIFORM | 20 | 0.003069 | 0.005128 | 0.000858 |
| TT_V2_MULTI_UNIFORM | 50 | 0.007058 | 0.010256 | 0.001677 |

结论：

- 当前最佳 Two-Tower 是 **Buy-only V1**。
- V1 的 Recall@50 比 V0 提升 `+0.008120`，约为 V0 的 `3.11×`。
- seed2027 Buy-only 的 Recall@50 比 V0 提升 `+0.004558`，约为 V0 的 `2.19×`；方向性提升得到一次 alternate-seed replication 支持。
- seed2027 比 seed42 V1 低 `0.003561`，相对下降约 `29.76%`，说明数值仍有不可忽略的 seed variance。
- V2 的 Recall@50 比 V0 提升 `+0.003212`，约为 V0 的 `1.84×`。
- V1 仍远弱于 ItemCF：`0.011966` 对 `0.371696`。
- 两个 Buy-only seeds 都高于 V0，进一步支持“PV training objective 与 purchase retrieval objective 不匹配”的假设；两次运行仍不足以证明严格稳定或单因素因果关系。

---

## 6. Two-Tower V0：PV-only baseline

### 6.1 模型与训练

- 类型：ID-only Two-Tower。
- 用户词表：9,836。
- 物品词表：302,016。
- Embedding 输出维度：64。
- Epochs：3。
- 原 checkpoint：`artifacts/two_tower/two_tower_best.pt`。

训练历史：

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.698493 | 0.692557 |
| 2 | 0.690275 | 0.682466 |
| 3 | 0.679218 | 0.660640 |

### 6.2 Embedding 导出

- User embeddings：`(9836, 64)`，`float32`。
- Item embeddings：`(302016, 64)`，`float32`。
- 两侧 embedding 均经过 L2 normalization。
- 因此 inner product 等价于 cosine similarity。

### 6.3 Exact retrieval 实现

- User batch size：64。
- Item chunk size：50,000。
- 每个 chunk 保留局部 Top-50。
- 对局部结果滚动合并为全局 Top-50。
- 不构造完整 `[9836, 302016]` score matrix。
- 8 个用户的小样本 chunked Top-10 与 dense exact Top-10 完全一致。
- 全量 warm evaluation 的 exact Top-50 已实际执行完成。

### 6.4 V0 推荐塌缩

- 1,170 个 warm users 的 Top-1 只有 2 个不同 item。
- item `3845720` 成为 91.97% 用户的 Top-1。
- 所有用户 Top-50 的并集只有 301 个 item，占 302,016 候选物品的极小部分。

---

## 7. Controlled experiment framework

### 7.1 共享不变量

为了尽量只观察 positive objective 的影响，V1/V2 使用：

- 相同 user/item mapping：9,836 / 302,016。
- 相同模型结构：ID-only towers、embedding dim 64、hidden dim 128、output dim 64、temperature 0.1。
- 相同 optimizer：Adam，learning rate `1e-3`。
- 相同 batch size：1,024。
- 相同负采样：每个正样本 4 个 uniform random negatives。
- 相同 epochs：3。
- 相同基础 random seed：42。
- 相同每 epoch positive draw budget：521,287，即 510 batches。
- 相同 exact full-catalog warm purchase retrieval evaluation。
- 不过滤用户历史物品，与现有 Popularity / ItemCF 评估口径保持一致。

### 7.2 Deterministic validation negatives

- Validation positive interaction 数：1,496。
- 固定 negatives shape：`(1496, 4)`。
- 固定 seed：2026。
- 每个用户的 negatives 排除 train 中已知 mapped interactions。
- 同时排除该用户 validation GT items。
- Sampled validation loss 只作为诊断项，**不作为最终模型优劣标准**。
- V1/V2 最佳 checkpoint 按 exact Recall@50 选择；相同时用 NDCG@50 打破平局。

---

## 8. 行为统计与 Multi-behavior 权重

| Behavior | Rows | Unique pairs | Users | Items | Future purchase overlap | Overlap rate | Lift vs PV | Sampling weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pv | 649,470 | 521,287 | 9,836 | 302,016 | 457 | 0.000877 | 1.000 | 1.000 |
| fav | 20,367 | 20,283 | 3,417 | 18,943 | 69 | 0.003402 | 3.880 | 1.970 |
| cart | 40,241 | 39,304 | 6,715 | 34,888 | 372 | 0.009465 | 10.796 | 3.286 |
| buy | 14,807 | 14,212 | 5,696 | 13,145 | 72 | 0.005066 | 5.779 | 2.404 |

Multi-behavior 实验的规则：

1. 将同一 mapped user-item pair 合并。
2. 如果一对出现多种行为，取最强行为的权重。
3. 权重使用 `sqrt(lift)`，避免原始 lift 过大。
4. 使用带 replacement 的 `WeightedRandomSampler`。
5. 每个 epoch 固定抽取 521,287 个 positive samples。
6. 本轮没有做权重搜索，避免将受控实验变成大规模调参。

---

## 9. Experiment 1：TT_V1_BUY_UNIFORM

### 9.1 数据覆盖

- Mapped buy positive pairs：11,844。
- Users：5,141。
- Items：10,819。
- 1,170 个 warm evaluation users 中，712 个在训练期有 buy。
- 为保持和 V0 相同的 step budget，每个 buy pair 每 epoch 平均被重复抽到约 44.01 次。

### 9.2 训练与选择结果

| Epoch | Train loss | Fixed sampled val loss | Exact Recall@50 | Exact NDCG@50 | 训练时间 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.398223 | 1.362624 | 0.003419 | 0.000764 | ~61.2s |
| 2 | 0.098541 | 2.001213 | 0.008974 | 0.002751 | ~61.0s |
| 3 | 0.040999 | 2.208330 | **0.011966** | **0.003453** | ~60.3s |

最佳 checkpoint：Epoch 3。

### 9.3 推荐集中度

- Unique Top-1 items：393。
- Dominant Top-1 item：`2560262`。
- Dominant Top-1 share：5.38%。
- Top-50 union：3,934 items。
- Top-50 catalog coverage：1.30%。

与 V0 相比，Buy-only 不仅提升 purchase Recall@50，也明显缓解了推荐塌缩。

### 9.4 Alternate-seed replication：TT_V1_BUY_UNIFORM_SEED2027

除随机种子 `42 → 2027` 和独立输出目录外，训练数据、mapping、11,844 个 mapped buy pairs、模型、optimizer、learning rate、batch size、4 个 uniform negatives、521,287 positive draws/epoch、3 epochs、固定 validation negatives、exact retrieval 和 checkpoint selection rule 均与原 V1 一致。

逐 epoch 结果：

| Epoch | Train loss | Fixed sampled val loss | Recall@50 | HitRate@50 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.405950 | 1.309375 | 0.000427 | 0.000855 | 0.000098 |
| 2 | 0.096238 | 1.983929 | 0.005556 | 0.006838 | 0.001329 |
| 3 | 0.041490 | 2.142018 | **0.008405** | **0.010256** | **0.001927** |

最佳 checkpoint：Epoch 3。

集中度：

- Unique Top-1 items：424。
- Dominant Top-1 item：`2560262`。
- Dominant Top-1 share：2.05%。
- Top-50 union：4,151 items。
- Top-50 catalog coverage：1.37%。

正式结论：**Buy-only improvement replicated directionally under an alternate random seed.** 该结果没有出现 V0 式的明显 collapse，但 Recall@50 相对 seed42 下降约 29.76%，仍存在数值 seed sensitivity 风险。

---

## 10. Experiment 2：TT_V2_MULTI_UNIFORM

### 10.1 数据覆盖

- Mapped positive pairs：541,958。
- Users：9,836。
- Items：302,016。
- 基于 `sqrt(lift)` 的行为权重进行 weighted sampling。

### 10.2 训练与选择结果

| Epoch | Train loss | Fixed sampled val loss | Exact Recall@50 | Exact NDCG@50 |
|---:|---:|---:|---:|---:|
| 1 | 0.695109 | 0.687192 | 0.000078 | 0.000076 |
| 2 | 0.678471 | 0.662867 | **0.007058** | **0.001677** |
| 3 | 0.652971 | 0.634544 | 0.005918 | 0.001569 |

最佳 checkpoint：Epoch 2。Epoch 3 虽然 sampled validation loss 更好，但真实 full-catalog Recall@50 下降，因此没有被选中。

### 10.3 推荐集中度

- Unique Top-1 items：11。
- Dominant Top-1 item：`3845720`。
- Dominant Top-1 share：81.20%。
- Top-50 union：688 items。
- Top-50 catalog coverage：0.23%。

第一版 Multi-behavior 虽优于 V0 的 Recall@50，但仍然明显受 PV 数据量主导，并保留严重的推荐塌缩。

---

## 11. Multi-stage Recall + Candidate Fusion

### 11.1 固定配置与候选 schema

- ItemCF Top-200，保持原 cosine-style 共现算法和最近 50 个 unique history items。
- `TT_V1_BUY_UNIFORM` seed42 Top-100，strict checkpoint restore，chunked exact full-catalog retrieval。
- Popularity Top-50，使用 train 全行为 interaction count。
- RRF 固定为 `sum(1 / (60 + source_rank))`，没有调参。
- Rank 全部从 1 开始；未出现的 source rank/score 统一为 null。
- 每行保留三个 source flag、rank、score、`recall_source_count` 和 `rrf_score`。
- Historical-item filtering 保持关闭，以维持与历史评估协议一致。

### 11.2 Validation source performance

统一评估人口仍为 1,170 warm users / 1,496 warm purchase interactions。

| Source | Quota | Recall | HitRate | Hit interactions |
|---|---:|---:|---:|---:|
| Popularity | 50 | 0.007123 | 0.010256 | 12 |
| ItemCF | 200 | 0.377329 | 0.411966 | 561 |
| Two-Tower seed42 | 100 | 0.020065 | 0.023077 | 27 |
| Candidate union | ≤350 | **0.384716** | **0.420513** | **573** |

Union 相对 ItemCF@200 增加 12 个 warm target hits，mean Recall 增量约 `+0.007387`。

### 11.3 Fusion ranking baselines

| Method | Recall@10 | Recall@20 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| ItemCF | 0.108468 | 0.189158 | 0.371696 | 0.110981 |
| RRF | 0.046610 | 0.078169 | 0.179821 | 0.052453 |
| ItemCF-first | 0.108468 | 0.189158 | 0.371696 | 0.110981 |

RRF 明显破坏强 ItemCF 排序；ItemCF-first 能保护 ItemCF Top-50，但补充候选排在 ItemCF Top-200 之后，因此 Top-50 不会获得增量。当前 union 的排序 headroom 很小，不能仅靠 Ranker 解决较低的 oracle ceiling。

### 11.4 Source overlap 与独立贡献

Validation 全用户 candidate-pair overlap：

- ItemCF ∩ Two-Tower：3,561 pairs，Jaccard `0.001253`。
- ItemCF ∩ Popularity：3,208 pairs，Jaccard `0.001358`。
- Two-Tower ∩ Popularity：20,336 pairs，Jaccard `0.014203`。

Validation warm GT exclusive contribution：

- Only ItemCF：538 interactions。
- Only Two-Tower：6 interactions。
- Only Popularity：3 interactions。
- ItemCF + TT：17 interactions。
- ItemCF + Popularity：5 interactions。
- TT + Popularity、但无 ItemCF：3 interactions。
- All three：1 interaction。
- Not recalled：923 interactions。

`two_tower_unique_gt_hits = 9`，即 TT 找回 9 个 ItemCF 未找回的 target，占 warm interactions 的 `0.6016%`；其中 3 个也被 Popularity 找回。Popularity 完全独立于 ItemCF 和 TT 的 hits 为 3。

### 11.5 Test、candidate size、fallback 与 cold limitation

- Test warm population：1,116 users / 1,438 interactions。
- Test warm union oracle Recall / HitRate：`0.255548 / 0.289427`。
- Test all-GT union oracle Recall：`0.168837`。
- Validation 平均/中位/p95/max candidates per user：`340.04 / 347 / 349 / 350`。
- Test 平均 candidates per user：`340.09`。
- Popularity-only fallback users：validation 2，test 3。
- Validation 从未出现在 train catalog 的 targets：894/2,540，`35.20%`。
- Test 从未出现在 train catalog 的 targets：962/2,512，`38.30%`。
- Validation/test candidates 都在读取 label 文件前完成并保存；test labels 只用于最终评估。

### 11.6 Validation 结果

两个 Parquet 均已回读；所有 user-item 唯一；source flag、source count、RRF、1-based ranks、rank/score 对齐和 350 上限全部通过。5 个随机用户样例已保存。历史 Two-Tower mappings 与 V0/V1/V2/seed2027 artifacts 的运行前后 SHA-256 完全一致。

---

## 12. Ranking + Recall → Rank End-to-End

### 12.1 数据边界与模型

- Backend：`lightgbm.LGBMRanker`，objective=`lambdarank`，seed=42。
- 固定参数：200 trees、learning rate 0.05、31 leaves；无 grid search，未根据 test 调参。
- Ranker 训练周期：validation candidates + validation purchase labels。
- 全量 validation candidates 为 3,304,863 行，其中 659 个 positive rows。
- LambdaRank 只保留至少含 1 个 positive candidate 的 query：549 users / 188,875 rows / 659 positives；每个 query 保留全部 negatives。
- `sum(group)=188,875`，与训练行数完全一致。
- 最终泛化评估周期：test。Test GT 在特征构造、训练和 test score 预测全部完成后才解析。
- 正式评估人口与先前一致：1,116 warm users / 1,438 purchase interactions，不过滤历史 items。

### 12.2 特征与泄漏检查

共 52 个模型特征：15 个 recall/source 特征，37 个基于 train 的 user/item/user-item/user-category 统计特征。原始 `user_id/item_id` 只作为键，不作为连续数值特征。Missing source rank 固定填 999，missing source score 填 0，unseen pair recency 填 -1 days。

Category 使用 train 期 item 最后一次出现的 category 并编码；发现 25 个 train items 有多 category，已用这一固定规则处理，没有使用 validation/test 信息修补。

运行内检查和独立 artifact tests 均通过：

1. aggregate max timestamp `1512143999 <= train end`;
2. test GT 不进入训练 dataframe，且在 test score 产生后才解析；
3. validation labels 只作为 label，aggregate features 全部来自 `train.csv`；
4. LambdaRank group sizes 与 rows 对齐；
5. validation/test 同 user 候选无重复；
6. 历史 baseline 指标全部精确复现，保护输入哈希未改变；
7. `ranker.pkl` 已独立重载并成功预测。

### 12.3 Test End-to-End 结果

| Method | Recall@10 | Recall@20 | Recall@50 | HitRate@50 | NDCG@50 |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.001792 | 0.003136 | 0.009670 | 0.012545 | 0.002553 |
| Two-Tower | 0.001344 | 0.001344 | 0.005376 | 0.006272 | 0.001312 |
| ItemCF | 0.080257 | 0.124298 | **0.241398** | **0.273297** | 0.075433 |
| RRF | 0.031138 | 0.054734 | 0.119191 | 0.140681 | 0.032805 |
| ItemCF-first | 0.080257 | 0.124298 | **0.241398** | **0.273297** | 0.075433 |
| Ranker | **0.166771** | **0.202315** | 0.238792 | 0.270609 | **0.124189** |

- Ranker 相对 ItemCF：Recall@10 `+0.086514`，Recall@20 `+0.078017`，Recall@50 `-0.002606` (`-1.08%` relative)。
- Ranker NDCG@50 相对 ItemCF 增加 `+0.048756`，说明相关候选明显向前移动，但少量 target 被推出 Top-50。
- Test warm candidate oracle Recall=`0.255548`，Ranker Recall@50=`0.238792`，oracle gap=`0.016756`。
- All-GT Ranker Recall@50=`0.159307`，all-GT oracle=`0.168837`，受 cold targets 限制明显。

### 12.4 Ablation-lite 与 feature importance

| Variant | Recall@20 | Recall@50 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|
| Recall-only | 0.155914 | **0.243302** | 0.061298 | 0.079950 |
| Recall + train aggregates | **0.202315** | 0.238792 | **0.116536** | **0.124189** |

全特征模型在 Top-20 和排序质量上明显更好，但 Recall@50 略低；这一轮没有因此删特征或调参。Top gain features 是 `seen_in_train` (58.09%)、`category_id` (13.52%)、`user_item_interaction_count` (4.35%) 和 pair recency (3.43%)。所有 Two-Tower 相关特征合计 gain fraction 仅 `0.0332%`；它们的可见边际作用很小，但未做 TT-isolation ablation，因此不声称严格因果贡献为零。

### 12.5 当前瓶颈结论

Ranker 已经非常接近现有候选集的 Top-50 oracle，而 oracle 本身只有 `0.255548`；因此 **retrieval remains the main bottleneck**。同时 38.30% test purchase targets 不在 train catalog，所以 **catalog coverage / cold-item retrieval is a major bottleneck**。Ranking 仍有明确改进空间，尤其是在保持 Top-10/20 和 NDCG 提升的同时避免 Top-50 小幅下降，但它不是当前主瓶颈。

---

## 13. Cold Start + Long Tail Segmentation Analysis

### 13.1 Catalog 边界与 Cold Start

- Full train catalog：317,721 items；Two-Tower train-PV vocabulary：302,016 items；ItemCF last-50 matrix catalog：222,183 items。
- TEST GT：2,512 interactions / 1,756 users，所有 purchase users 都在 train 出现，因此没有可评估的 train-cold user。
- Train-cold targets：962 interactions / 956 unique items，占 `38.30%`。
- TT-vocab-cold targets：1,059 interactions / 1,050 unique items，占 `42.16%`。
- ItemCF-matrix-cold targets：1,118 interactions / 1,111 unique items，占 `44.51%`。

| Bucket | Interactions | Fraction | Ranker Recall@50 | Interpretation |
|---|---:|---:|---:|---|
| A: Warm user + Warm item | 1,550 | 61.70% | 0.249677 | train-known target under closed catalog |
| B: Cold user + Warm item | 0 | 0.00% | N/A | 无样本 |
| C: Warm user + Cold item | 962 | 38.30% | 0 | structurally unretrievable |
| D: Cold user + Cold item | 0 | 0.00% | N/A | 无样本；若出现则 structurally unretrievable |

Closed-catalog interaction Recall ceiling=`0.617038`。Ranker all-GT interaction Recall@50=`0.154061`（387 hits），只达 ceiling 的 `24.97%`；union oracle all-GT interaction Recall=`0.164411`（413 hits），达 ceiling 的 `26.65%`。因此在 962 个 structural cold misses 之外，仍有 1,137 个 train-known targets 没有进入候选集，warm retrieval failure 在绝对数量上更大。

Popularity fallback 无法做真正 cold-user 效果估计，因为 TEST purchase GT 中 train-cold users=0。Multi-stage 中的 3 个 popularity-only fallback users 不是 train-cold users。

### 13.2 Head / Torso / Tail 定义与表现

仅用 train interaction count，按 `count DESC, item_id ASC` 排序，然后直接按 item row index 分割：Head 63,544 items，Torso 95,316，Tail 158,861。Popularity percentile 定义为 `0=most popular, 1=least popular`。Tail items 全部只在 train 出现 1 次；cold item 不会进入 Tail。

TEST target distribution：Head 920（36.62%）、Torso 337（13.42%）、Tail 293（11.66%）、Cold 962（38.30%）。

| Segment | Targets | ItemCF R@50 | TT R@50 | Ranker R@50 | Oracle Recall | Ranker NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| Head | 920 | 0.270652 | 0.006522 | 0.271739 | 0.290217 | 0.147105 |
| Torso | 337 | 0.258160 | 0.002967 | 0.246291 | 0.261128 | 0.114719 |
| Tail | 293 | 0.194539 | 0.000000 | 0.184300 | 0.197952 | 0.093452 |

Tail 是最难 segment，且 Tail oracle 本身最低。Ranker 距 Tail oracle 只有 `0.013652`，所以 Tail 主要受 retrieval 限制，不是主要受 ranking 限制。Ranker 相对 ItemCF 的 Recall@10 增益主要来自 Head（`+0.151087`）；Torso `+0.029674`，Tail `-0.003413`。

### 13.3 Recommendation exposure、coverage 与 novelty

曝光人口为 `test.csv` 的全部 9,688 users。

| Model | Head % | Torso % | Tail % | Catalog coverage | Tail coverage | Mean novelty | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|
| ItemCF | 38.06% | 26.24% | 35.70% | 69.46% | 60.36% | 18.1503 | 0.5318 |
| Two-Tower | 84.99% | 14.65% | 0.36% | 1.94% | 0.02% | 16.4527 | 0.9966 |
| Ranker | 58.71% | 19.96% | 21.33% | 62.32% | 50.65% | 17.0023 | 0.6954 |

Train-known TEST demand share 为 Head/Torso/Tail=`59.35%/21.74%/18.90%`。因此 ItemCF 并不是 exposure 意义上的明显 Head-biased；它相对 demand 少曝光 Head `-21.30pp`，多曝光 Tail `+16.80pp`。Two-Tower 才是明显 Head amplification：Head `+25.63pp`、Tail `-18.54pp`。Ranker 相对 ItemCF 更 Head-heavy，但相对 demand 的 Head 差只有 `-0.64pp`，Tail 为 `+2.43pp`；数据不支持“Ranker 进一步放大了相对 demand 的 popularity bias”。

### 13.4 Two-Tower Tail contribution 与阶段结论

Two-Tower Top-100 相对 ItemCF Top-200 只新增 5 个 TEST target hits，全部在 Head；Torso/Tail exclusive hits 都是 0。Two-Tower Tail Top-50 Recall 也是 0，Tail exposure 只有 0.36%。因此本次数据明确不支持“Two-Tower 虽然 overall 弱，但对 Tail 有特殊价值”。

Cold Start 与 Long Tail 是两个不同问题：Cold item 从未出现在 train，当前 closed-catalog ID-based 系统结构上无法召回；Tail item 已出现在 train，仍可由 ItemCF/TT/Ranker 召回。本轮完成了分层评估，没有“解决 Cold Start”。

---

## 14. TIGER-style Collaborative Semantic-ID Generative Retrieval

### 14.1 实验边界与固定协议

这是现有 Taobao Multi-stage Recommender 的 **experimental complementary recall source**，不是 exact TIGER reproduction。原 TIGER 使用 content-derived semantic representations；本项目没有丰富 item 文本，因此输入是 `TT_V1_BUY_UNIFORM` 的 purchase-aligned collaborative item-tower output。Active catalog 是完整 V1 vocabulary（302,016 items），没有根据 validation/test targets 缩小 catalog。

RQ-VAE 与 generator 只使用 train 数据训练；RQ-VAE checkpoint 按 deterministic 90/10 item split 的 validation reconstruction 选择；generator 按原 validation purchase warm population的 Recall@20、NDCG@20 tie-break 选择。Validation/test history 都只来自 `train.csv`，与既有 ItemCF candidate generation 的 cutoff 一致。所有配置与 validation diagnostics 在 test 前冻结；test 没有用于调 codebook、网络、beam、history、quota 或 epoch。

### 14.2 RQ-VAE 与 Semantic IDs

| Metric | Value |
|---|---:|
| RQ-VAE best epoch | 15 |
| Validation reconstruction loss | 0.00188850 |
| Codebook 0 utilization / perplexity | 100% / 108.7562 |
| Codebook 1 utilization / perplexity | 100% / 114.9056 |
| Codebook 2 utilization / perplexity | 100% / 113.1856 |
| Unique raw tuples | 248,774 |
| Raw tuple collision excess rate | 17.6289% |
| Maximum raw collision group | 11 |
| Maximum suffix | 10 |
| Final SID collision | 0 |

没有 codebook collapse。共享 SID prefix 的平均 embedding cosine 从 exact-prefix 0 的 `0.6908` 上升到 prefix 3 的 `0.9114`，说明 codes 保留部分 collaborative structure；这只是 diagnostic，不证明内容语义。

### 14.3 Sequence、Transformer 与合法生成

Train 中 14,807 个 purchase events 有 12,470 个 target 位于 active catalog；去除 263 个空 active-history examples 后得到 12,207 个严格因果训练样本。发现同一用户存在同 timestamp 事件后，构建器按 timestamp batch 处理，保证 `history_timestamp < target_timestamp`，不把同秒早一行当作历史。

Transformer 配置为 d_model=128、4 heads、2-layer encoder/decoder、FFN=256、10 epochs。CPU 实现仍保留 `<BEHAVIOR> c0 c1 c2 suffix <SEP>` 六个 token，但在 encoder 前做 event-wise embedding pooling，将 attention 长度从 180 降到 30。Best epoch=10。完整 catalog trie + beam=50 在 validation/test 的 valid generation rate 均为 100%，invalid path 与 duplicate item 都是 0。

### 14.4 Validation 与一次性 Test 结果

| Split | R@5 | R@10 | R@20 | R@50 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.059729 | 0.063319 | 0.063319 | 0.065456 | 0.050897 | 0.051360 |
| Test | 0.042279 | 0.042458 | 0.042458 | 0.042458 | 0.036875 | 0.036875 |

| Split | Current union hits | Gen hits | Exclusive vs ItemCF | Exclusive vs union | Union oracle before | After | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 573 | 96 | 2 | 2 | 0.384716 | 0.386426 | +0.001709 |
| Test | 366 | 64 | 0 | 0 | 0.255548 | 0.255548 | 0 |

Validation 的 2 个增量 target 都属于 Head；test 没有复现增量。两边 Tail GenRec Recall@50 与 exclusive hits 都为 0。GenRec test 单路明显高于历史 Two-Tower，但低于 ItemCF，且没有提高正式 test union oracle。

### 14.5 Exposure、失败模式与正式决定

GenRec test Top-50 exposure：Head/Torso/Tail=`78.08%/19.32%/2.60%`，unique items=14,137，full train catalog coverage=4.45%，Gini=`0.988493`，mean novelty=`17.0052`。它比 Two-Tower（Head 84.99%、Tail 0.36%、Gini 0.996580）略健康，但仍严重集中于 Head。

Failure analysis 对 ItemCF misses 保留 22 个样例：2 个 validation GenRec recoveries 和 20 个两者都 miss 的样例。主要失败模式是生成结果复制用户近期/热门协同模式，但目标 SID 经常没有共享首层 prefix；Tail targets 没有命中。

最终决定：工程链路达到 LEVEL A，单路非零 Recall 达到 LEVEL B；validation 有极小 LEVEL C 信号，但 test 没有增量且 Tail=0。因此保留完整失败/部分成功结果，**不把当前 GenRec 加入正式第四路 recall，不重训 Ranker，不根据 test 回头调参**。

它仍是 V1 collaborative embedding 驱动的 closed-catalog 方法：1,059 个 test targets 不在 active V1 catalog，其中 962 个是真正 train-cold。没有 content/metadata representation 时不能称为 Cold Start solution。

---

## 15. 已确认的重要问题与风险

### 15.1 训练目标与业务目标不一致

V0 主要学习 PV，但最终评估 purchase retrieval。受控实验中 Buy-only 明显优于 PV-only，支持 objective mismatch 假设。

限制：V0 是历史 checkpoint，最初按 sampled validation loss 训练/选择；V1/V2 按 exact Recall@50 选择。旧 epoch checkpoints 不存在，因此无法把 V0 完全重新选择为 retrieval-aware baseline。

### 15.2 Sampled validation loss 与真实 retrieval 指标不一致

- V1：sampled val loss 从 `1.362624` 恶化到 `2.208330`，但 Recall@50 从 `0.003419` 提升到 `0.011966`。
- V2：Epoch 3 sampled val loss 继续改善，但 Recall@50 从 `0.007058` 降到 `0.005918`。

因此 sampled loss 不适合作为最终 full-catalog purchase retrieval 的模型选择标准。

### 15.3 Buy-only 数据稀疏与重复抽样

Buy-only 只有 11,844 个 mapped pairs。为了保持相同步数，单个 pair 每 epoch 平均重复约 44 次，可能导致过拟合或 seed 敏感。这是下一步首先需要验证的风险。

### 15.4 ID-only warm-start 覆盖有限

Validation purchase interactions 只有 58.90% 可进入当前 warm evaluation。模型无法表示新用户或不在 train-PV vocabulary 中的新物品。

### 15.5 模型能力明显弱于 ItemCF

即使最佳 V1 的 validation Recall@50 提升到 0.011966，也远低于 ItemCF 的 0.371696。Ranking 已经建成，但 Two-Tower 特征在 ranker 中的 gain importance 合计仅 0.0332%，与它的低独立命中贡献一致；它仍不能成为主召回源。

### 15.6 Multi-behavior 权重仍被 PV 主导

尽管 fav/cart/buy 的单样本权重更高，PV 数量优势仍很大。第一版 weighting 没有解决塌缩，也不是最终 multi-behavior 方案。

### 15.7 Alternate-seed 方向复现，但仍有数值波动

Buy-only seed42 与 seed2027 的 Recall@50 分别为 0.011966 和 0.008405，两者均高于 V0 的 0.003846，方向一致；但 seed2027 相对 seed42 下降约 29.76%。因此可以说方向获得一次复现支持，不能声称模型已经严格稳定。V2 仍只有单 seed。

---

## 16. 发现过的流程与实现问题

以下问题已在审计或开发中发现；这里记录现状，不表示都已从历史 Notebook 输出中抹除：

1. **Mapping 不可随意重建**：checkpoint 只保存 embedding table 大小，若 ID → index 顺序不一致，模型会静默返回错误结果。现已按原训练规则恢复、验证并持久化 mapping。
2. **Notebook 状态依赖尚未完全消除**：早期数据与召回仍以 Notebook 为主；seed 复验、Multi-stage Recall 和 Ranking 已有可独立运行脚本。
3. **旧验证 loss 存在随机 negative 不稳定性**：V0 的 historical validation loss 不能与新 deterministic protocol 完全等价；新 Two-Tower 实验已使用固定 negatives。
4. **完整 score matrix 有内存风险**：`9836 × 302016` 不一次性构造；当前使用 chunked exact Top-K。
5. **FAISS 仍未实现**：当前只有 chunked brute-force exact baseline，没有 IndexFlatIP 一致性或耗时比较。
6. **Windows Unicode 模型路径**：LightGBM 原生 writer 无法直接写入带中文的项目路径；脚本通过 ASCII temp 保存 native text 后再用 Python I/O 复制，并额外保存可直接加载的 `ranker.pkl`。
7. **Warm 与 all-GT 指标必须分开**：排序正式横向比较使用 1,116 个 test warm users；Ranker all-GT Recall@50 只有 0.159307。
8. **V0/V1/V2 selection protocol 不完全对称**：历史 V0 缺少中间 checkpoints，该限制仍需披露。

更完整的问题、证据与处理决定见 `PROJECT_ISSUES.md`。

---

## 17. 当前最后一个成功运行的步骤

当前最后一个成功运行的阶段是 **TIGER-style Collaborative Semantic-ID Generative Retrieval** 全闭环：

```text
V1 embedding export → RQ-VAE → Semantic IDs → causal sequences
→ Transformer → trie-constrained beam search
→ validation selection → frozen one-time test evaluation
→ segment/incremental/exposure/failure analysis
```

关键结果：完整 V1 catalog 302,016 items；RQ-VAE 三层 utilization 100%；final SID collision=0；generator best epoch=10；validation/test valid generation=100%；validation current-union exclusive hits=2（均 Head），test exclusive hits=0，Tail exclusive hits=0。

最后完整回归：

```text
Ran 28 tests in 3.395s
OK
```

`scripts/run_generative_pipeline.py` 已以完成态执行保护检查，确认存在 completed `final_summary.json` 时不会重训或重复 test evaluation。传统 Two-Tower、multi-stage recall、ranking 与 cold/long-tail artifacts 的保护哈希保持不变。

### 17.1 Generative 阶段前的历史最后步骤

当前最后一个成功运行的阶段曾是 **Cold Start + Long Tail Segmentation Analysis**：

```text
python scripts/09_analyze_cold_start_long_tail.py
```

脚本在不训练、不调参、不修改模型产物的前提下，基于固定 test purchase GT、固定 multi-stage candidates 和固定 ranker Top-50 完成了：

- cold user / cold item / strict-warm bucket 审计；
- train-only Head / Torso / Tail 确定性分段；
- Popularity / ItemCF / Two-Tower / Ranker 分层离线指标；
- candidate oracle、closed-catalog ceiling、exposure、coverage、novelty 与 Gini；
- Two-Tower 相对 ItemCF 的 segment-level incremental hit 审计。

最后成功验证：

```text
python -m unittest tests.test_cold_long_tail_analysis tests.test_multistage_recall tests.test_ranking_artifacts -v
Ran 17 tests
OK
```

关键输出：

- test GT：2,512 interactions；其中 train-cold item target 962（38.2962%）；
- closed-catalog interaction Recall@50 ceiling：0.617038；
- Ranker full-test interaction Recall@50：0.154061；
- Ranker Head / Torso / Tail Recall@50：0.271739 / 0.246291 / 0.184300；
- Two-Tower Tail hits：0，Tail exclusive hits：0；
- test GT 中 train-cold user：0，因此当前数据不能评估真正的 cold-user fallback 效果。

### 17.2 Cold/long-tail 分析前的历史快照

当前最后完整完成的功能步骤是：

> 通过 `scripts/08_train_ranker.py` 用 validation candidates + validation purchase labels 训练固定参数 LightGBM LambdaRank，严格使用 train-only aggregates，对 3,294,775 条 test candidates 打分，然后才读取 test GT 完成端到端评估。

核心输出：test warm Ranker Recall@10/@20/@50=`0.166771/0.202315/0.238792`，HitRate@50=`0.270609`，NDCG@50=`0.124189`；candidate oracle=`0.255548`，oracle gap=`0.016756`。两个 ranker 、两份 ranking Parquet、Top-50、metrics、importance、manifest、config 和 summary 均已保存。`tests/test_ranking_artifacts.py` 5/5 通过，所有保护输入哈希保持不变。

---

## 18. 接下来最合理的开发顺序

1. **先建立新的 temporal validation/ranker-selection fold**：当前 validation 已用于训练传统 ranker，也用于 GenRec checkpoint selection；当前 test 已做一次最终诊断。没有新时间折叠前，不应继续调整生成式架构、beam、codebook 或 fusion。
2. **把当前 GenRec 保留为实验 baseline，而不是正式第四路 source**：validation 的 2 个 Head 增量未在 test 复现，Tail 增量为 0；不应据此重训旧 Ranker。
3. **若继续生成式研究，目标应是新的 warm/Tail incremental coverage**：必须先在新 validation fold 预注册配置，再报告 ItemCF/union misses；不能只优化单路 Recall 或 diversity。
4. **若目标转向 truly cold item**：需要测试时可用的 content/metadata encoder。Collaborative Semantic ID 无法表示 train 中从未出现的 962 个 test targets。
5. **保留传统主线为当前正式系统**：ItemCF primary recall + fixed union + LambdaRank 仍是正式 baseline；RRF 和本次 GenRec 的失败/部分成功结果都保留。

### 18.1 Generative 阶段前的历史计划

1. **优先改善 warm retrieval coverage**：固定现有 ranker 与 test protocol，预注册新的互补 recall source；必须报告全局及 Head/Torso/Tail 的 incremental hits 与 oracle。当前 candidate union 对 1,550 个 train-known target 只命中 413 个，仍遗漏 1,137 个，这是最大绝对损失来源。
2. **再设计真正的 cold-item representation**：若有测试时可用的 item metadata/content，可建立非纯 ID 的 item encoder；严格区分 `Cold` 与 `Tail`，并审计特征可用时间，避免 leakage。
3. **Generative Recommendation 可以进入研究阶段，但不能直接宣称解决 Cold Start**：更合理的目标是作为新的 warm/long-tail recall source，先验证它是否带来 candidate-union 增量命中，尤其是 Tail；仍须保持 test diagnostic-only。
4. **若未来继续优化训练或 Ranking**：先建立新的 validation/time split 或滚动验证，不允许根据当前 test 分层结果反复选模型、调 quota 或调参数。

### 18.2 Cold/long-tail 分析前的更早历史计划

### Step 1 — Cold Start / Catalog coverage 分层评估

先将 test purchase 按 target 是否在 train catalog、是否在 TT vocabulary 以及 user 是否 warm 分桶，确认 38.30% train-cold targets 的具体分布。这一步只做分析，不立即训练更复杂模型。

### Step 2 — Long Tail 与推荐集中度分层

按 train item popularity 将 target 和 recommendations 分为 head / torso / tail，报告分桶 Recall/NDCG、catalog coverage、novelty 和集中度。保留当前 Ranker 作为固定 baseline。

### Step 3 — 预注册新召回源以提高 oracle

- ItemCF 继续作为主召回源。
- 新召回必须用 validation 预注册配置，并同时报告 incremental oracle hits 和 all-GT/cold coverage。
- 不根据 test 改 quota、fusion 权重或 ranker 参数。
- Two-Tower 保留为辅助源，不追加 post-hoc seeds。

### Step 4 — 再研究 Top-50 trade-off

仅在有新 validation/test 周期或预注册协议时，再研究如何保留当前 @10/@20/NDCG 提升同时恢复 Recall@50。不应用本次 test 结果回头调参。

### Step 5 — 可选 FAISS IndexFlatIP 工程一致性

若后续需要检索性能 benchmark，只先用 `IndexFlatIP` 与当前 chunked exact Top-K 做结果一致性和耗时比较，暂不进入 IVF/HNSW。

---

## 19. 现在不应该做的事

- 不根据 GenRec test 的 0 增量回头改 beam、architecture、history、codebook 或 active catalog；
- 不反复运行 test evaluation；`pre_test_freeze.json` 已标记 completed-once；
- 不把 validation 的 2 个 Head hits 包装成已验证的系统提升；
- 不把当前 GenRec 加入旧 Ranker 并重训，因为没有独立 ranker-selection split；
- 不把 collaborative SID 称为 content-derived semantic ID、exact TIGER 或 Cold Start solution；
- 不删除 test 0 增量、Tail 0 命中和 Head concentration 结果。

### 19.1 Generative 阶段前的历史约束

- 不根据本次 test bucket / segment 结果回头调参、选 checkpoint 或调 fusion quota；
- 不把 `Cold` 与 `Tail` 混为一谈；
- 不宣称现有系统已经解决 Cold Start；
- 不把没有 metadata/content 表示的纯 ID 模型包装成 cold-item 方法；
- 不把 Two-Tower 描述为长尾补充：本次审计中其 Tail hit 与 exclusive Tail hit 都为 0；
- 不因 RRF 失败而删除或覆盖历史结果；
- 不直接进入更复杂的 ranker，当前各 segment 的主要瓶颈仍是 retrieval coverage。

### 19.2 Cold/long-tail 分析前的更早历史约束

- 不要覆盖现有 V0/V1/V2、seed2027、Multi-stage Recall 或 Ranking artifacts。
- 不要把一次 alternate-seed 方向复现表述为严格稳定性证明。
- 不要根据本次 test labels 调整 ranker 参数、feature set、quota、RRF 或 source 规则。
- 不要用 sampled validation loss 代替 full-catalog exact retrieval 指标。
- 不要一次性构造完整 `[9836, 302016]` score matrix。
- 不要将 RRF 当作有效最终排序；它在 validation/test 都显著弱于 ItemCF。
- 不要把 warm-start 指标误报为全流量表现。
- 不要直接跳到 DIN/DIEN/DeepFM/DCN/neural ranker、生成式推荐或大规模超参搜索。

---

## 20. 可复制的简版状态

以下为当前权威快照，可直接复制到新对话：

```text
CURRENT PROJECT STATUS — taobao-recsys (2026-10-08)

- 传统主线已完成：Data → Popularity/ItemCF/Two-Tower → Candidate Union → LightGBM LambdaRank。
- Test warm Ranker Recall@10/@20/@50 = 0.166771/0.202315/0.238792，NDCG@50=0.124189；current union oracle=0.255548。
- Cold/Long-tail 诊断已完成但 Cold Start 未解决：962/2512 test targets 为 train-cold；Tail 是最难的 train-known segment。
- TIGER-style collaborative Semantic-ID GenRec 已完整运行，不是 exact TIGER，也不是 cold-start method。
- Active catalog=完整 V1 vocabulary 302,016 items；V1 normalized embedding dim=64。
- RQ-VAE best epoch=15，validation reconstruction=0.00188850；三层 utilization=100%/100%/100%，无 collapse。
- Raw SID tuple unique=248,774，collision excess rate=17.6289%；item_id-ordered suffix 后 final collision=0。
- Generator 使用 12,207 个严格因果 purchase examples，behavior-aware histories，best epoch=10，beam=50，invalid generation=0。
- Validation GenRec Recall@20/@50=0.063319/0.065456；相对 current union 新增2个Head target，oracle +0.001709。
- 一次性 Test GenRec Recall@20/@50=0.042458/0.042458；相对 ItemCF/union 新增均为0，Tail hit与exclusive hit均为0。
- Test GenRec exposure Head/Torso/Tail=78.08%/19.32%/2.60%，coverage=4.45%，Gini=0.988493；仍严重偏 Head。
- 决定：工程闭环与非零单路召回成功，但不作为正式第四路 recall，不重训旧 Ranker，不根据 test 调参。
- 所有生成式、cold/long-tail、recall、ranking tests 共28/28通过；旧模型和旧 artifacts 未被覆盖。
- 下一步必须先建立新的 temporal validation fold；若要解决 truly cold item，需要 content/metadata representation。
```

### 20.1 Generative 阶段前的历史简版状态

以下为生成式实验前的历史快照：

```text
CURRENT PROJECT STATUS

- 项目：taobao-recsys；Notebook-driven，并已有可独立重跑的实验/分析脚本。
- 已完成：数据清洗与时间切分、purchase GT、Popularity、ItemCF、negative sampling、Dataset/DataLoader、Two-Tower V0/V1/V2、seed=2027 复验、embedding 导出、exact Top-K、warm-start Recall/HitRate/NDCG、Multi-stage Recall/Candidate Fusion、LightGBM LambdaRank、Cold Start/Long Tail segmentation analysis。
- 当前最佳最终排序：fixed-candidate LightGBM LambdaRank；test warm Recall@10/@20/@50 = 0.166771 / 0.202315 / 0.238792，NDCG@50 = 0.124189。
- 固定 candidate warm oracle Recall@50 = 0.255548；Ranker 已接近该候选集上限，继续改 ranker 的空间小于改善 retrieval coverage 的空间。
- test purchase GT 共 2,512 interactions；962 个 target item 不在完整 train catalog（38.2962%）；没有 train-cold user。
- closed-catalog interaction Recall@50 ceiling = 0.617038；Ranker full-test interaction Recall@50 = 0.154061，candidate-union oracle = 0.164411。
- Ranker Head/Torso/Tail Recall@50 = 0.271739 / 0.246291 / 0.184300；Tail 最难，且主要受 retrieval 限制。
- Two-Tower Top-50 曝光高度集中：Head 84.9870%、Tail 0.3601%、Gini 0.996580；其 Tail target hits 和 exclusive Tail hits 都为 0。
- ItemCF 相对 train-known demand 并非 Head-biased，而是 Tail over-exposure；Ranker exposure 与 demand 大体校准。
- Cold Start 尚未解决：现有模型均为 closed-catalog，无法命中从未出现在 train 的 item；本轮只是诊断与分层评估。
- RRF 的失败实验必须保留，不能覆盖历史事实。
- 最新独立分析脚本：scripts/09_analyze_cold_start_long_tail.py；产物位于 artifacts/analysis/。
- 最新测试：cold/long-tail 7 项 + recall 5 项 + ranking 5 项，共 17/17 通过。
- 下一步：优先研究新的互补 recall source；Generative Recommendation 可以作为 warm/long-tail retrieval research 进入下一阶段，但必须预注册、报告 segment incremental oracle，且不得使用当前 test 反复调参。
```

### 20.2 Cold/long-tail 分析前的更早历史简版状态

```text
CURRENT PROJECT STATUS — taobao-recsys (2026-10-07)

Ranking 与 Recall → Rank 端到端 baseline 已完整运行并验证。

已完成：
- 数据清洗、时间划分、validation/test purchase GT
- Popularity、ItemCF、Two-Tower V0/V1/V2 与 Buy-only seed2027 复验
- mapping/embedding 导出、chunked exact full-catalog Top-K、warm Recall/HitRate/NDCG
- ItemCF Top-200 + seed42 V1 TT Top-100 + Popularity Top-50 candidate union
- validation/test candidates、oracle、source overlap/contribution、RRF 和 ItemCF-first
- 52-feature LightGBM LambdaRank；validation labels 训练，test labels 仅最终评估
- Recall-only vs all-feature ablation、feature importance、5 个 test-user sanity samples
- 5/5 持久化 artifact/leakage/group/model-load tests

Test warm evaluation: users=1,116, interactions=1,438
Method          Recall@10  Recall@20  Recall@50  HitRate@50  NDCG@50
Popularity      0.001792   0.003136   0.009670   0.012545    0.002553
Two-Tower       0.001344   0.001344   0.005376   0.006272    0.001312
ItemCF          0.080257   0.124298   0.241398   0.273297    0.075433
RRF             0.031138   0.054734   0.119191   0.140681    0.032805
ItemCF-first    0.080257   0.124298   0.241398   0.273297    0.075433
Ranker          0.166771   0.202315   0.238792   0.270609    0.124189

Candidate oracle warm Recall=0.255548
Ranker Recall@50=0.238792
Oracle gap=0.016756
Ranker vs ItemCF Recall@50 delta=-0.002606 (-1.08%)
Cold target fraction absent from train=38.30%
Two-Tower feature gain fraction=0.0332% (descriptive, not causal)

结论：
- Ranker 在 Recall@10/@20 和 NDCG@50 明显超过 ItemCF，Recall@50 略低。
- Recall-only ablation Recall@50=0.243302，但全特征模型的 @20 和 NDCG 更好。
- 现有 Ranker 已接近 union oracle；主要瓶颈是 retrieval coverage 和 cold-item catalog coverage，不是单纯 ranking。
- RRF 失败结果保留，不作为最终推荐。

尚未完成：
- Cold Start 方案与分层评估
- Long Tail 分层与优化
- 能明显提高 candidate oracle 的新召回源
- FAISS IndexFlatIP 可选工程 benchmark

NEXT RECOMMENDED STEP:
先做 Cold Start / Long Tail 分层评估，再预注册新召回源以提高 oracle。不使用本次 test 结果回头调参。
```

