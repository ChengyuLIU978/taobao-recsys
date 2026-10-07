# Taobao Multi-Stage Recommender System with Generative Retrieval Extension

## Executive Summary

本项目在 Taobao UserBehavior 的 100,150,807 条行为上搭建了一条完整、可审计的推荐系统链路。开发阶段使用确定性的约 1% user-level sample，共 1,001,832 条行为；系统按时间切分数据，完成 Popularity、ItemCF、ID-only Two-Tower、多路候选集、52-feature LightGBM LambdaRank、一次性 test 评估，以及 Cold Start/Long Tail 诊断。

正式系统是 `Popularity + ItemCF + Two-Tower → Candidate Union → LambdaRank`。在统一 test-warm purchase 协议上，Ranker 的 Recall@10/20/50 为 `0.166771 / 0.202315 / 0.238792`，NDCG@50 为 `0.124189`。ItemCF 的 Recall@50 为 `0.241398`，略高于 Ranker；Ranker 的优势主要在 early ordering 与 NDCG。候选 oracle Recall=`0.255548`，与 Ranker@50 只差 `0.016756`，说明后续主要瓶颈是 retrieval coverage。

项目还完成了 TIGER-style collaborative Semantic-ID 生成式召回实验。V1 item embeddings 经 RQ-VAE 量化，三层 codebook utilization 均为 100%。Raw `(c0,c1,c2)` tuple collision 仍可测量，collision excess rate=`17.6289%`；按 item ID 确定性排序的 suffix 使最终 `(c0,c1,c2,suffix)` Semantic ID 映射一一对应，因此 final SID collision count=0。Transformer 与 trie-constrained beam search 实现 100% valid generation。该模块 validation 为 current union 新增 2 个 Head targets，但 test 新增 0，因此不进入正式候选 union，只保留为完整且诚实的实验基线。

## Problem Definition

目标是根据用户在训练时间窗内的浏览、收藏、加购和购买行为，为未来时间窗的购买行为生成 Top-K item recommendation。项目同时研究四个层次的问题：

1. 如何构建无 future leakage 的时序数据与评估协议；
2. 多种 retrieval 模型能否提供互补候选；
3. Learning-to-Rank 能否把候选中的正样本提前；
4. collaborative Semantic IDs 与生成式检索能否提供传统召回之外的增量覆盖。

核心离线指标为 user-level macro Recall@K、HitRate@K 与 NDCG@K；候选阶段还报告 oracle recall、source overlap、exclusive target hits、catalog coverage 与 segment exposure。

## Dataset

- 原始数据：Taobao UserBehavior，100,150,807 interactions，约 3.67 GB。
- 确定性开发集：`user_id % 100 == 0`，1,001,832 interactions。
- Train：724,885。
- Validation：138,406。
- Test：138,541。
- Validation purchase GT：2,540 interactions / 1,760 users。
- Test purchase GT：2,512 interactions / 1,756 users。

user-level sampling 保留一个用户的完整历史，避免 row-level sampling 将序列、共现与 recency 信号切碎；模运算使样本可重复构造。

## Data Audit

项目检查了 schema、行为类型、空值、重复、timestamp 边界、用户和 item 覆盖、purchase target 数量与 mapping vocabulary。全量数据构建由 `02_build_dev_dataset.ipynb` 执行；`01_data_audit.ipynb` 的部分初步统计只基于前 100,000 行，因此不被误当作全量结论。

Two-Tower mapping 恢复时还验证了构建规则：从 train `pv` 行取 `(user_id,item_id)`，保持首次出现顺序并使用 pandas stable `unique()`。最终 vocabulary 是 9,836 users 与 302,016 items，与 checkpoint tensor shape 完全一致。

## Temporal Split

| Split | Timestamp |
|---|---|
| Train | 1511539200–1512143999 |
| Validation | 1512144000–1512230399 |
| Test | 1512230400–1512316799 |

不用 random split 的原因是推荐数据存在强时间方向：未来购买不能参与历史序列，未来交互不能进入 popularity、user/item aggregate 或 similarity。Ranking 的所有 52 个 aggregate feature 都只使用 train。生成式序列按 timestamp batch 处理同秒事件，保证 `history_timestamp < target_timestamp`。

## Evaluation Protocol

传统主表使用相同的 strict warm purchase population；K 取 10、20、50。Warm 表用于公平比较 closed-catalog retrievers；另以全 2,512 个 test purchase interactions 诊断 train-cold ceiling。Test labels 不参与 candidate generation、feature construction、model fitting 或 checkpoint selection，只在最终评分结果形成后读取。

Recall 是按用户对 GT coverage 求平均，HitRate 衡量用户是否至少命中一个 target，NDCG 同时考虑命中位置。Candidate oracle 忽略候选内部排序，回答“正样本是否进入候选集”，因此可区分 retrieval 和 ranking 两类瓶颈。

## Popularity

Popularity 是非个性化 fallback。Test-warm Recall@10/20/50=`0.001792/0.003136/0.009670`，NDCG@50=`0.002553`。它的覆盖稳定，但个性化能力极弱；主要价值是补齐无历史或候选不足用户，而不是作为强主召回。

## ItemCF

ItemCF 基于 train 行为共现，是最强单一 recall source。Test-warm Recall@10/20/50=`0.080257/0.124298/0.241398`，NDCG@50=`0.075433`。在这个短时间窗、行为重复强且 item identity 直接可用的数据上，局部共现信号明显强于 ID-only neural embedding。

## Two-Tower

Two-Tower 使用 ID embedding、uniform negatives 与点积检索。Exact retrieval 采用 user batch 与 item chunk，避免一次性构造 `[9836,302016]` 完整 score matrix；小样本结果与 dense exact 完全一致。

正式多路 pipeline 使用的 Two-Tower 在 test-warm 上 Recall@10/20/50=`0.001344/0.001344/0.005376`。它明显弱于 ItemCF，并且 exposure 极端偏向 Head。

## Controlled Experiments

| Model | Positive objective | Recall@50 |
|---|---|---:|
| TT_V0_PV_UNIFORM | PV-only | 0.003846 |
| TT_V1_BUY_UNIFORM | Buy-only | 0.011966 |
| TT_V2_MULTI_UNIFORM | Weighted multi-behavior | 0.007058 |

V0 的训练目标是高频 PV，而最终目标是未来 purchase，存在 objective mismatch。保持 mapping、capacity、optimizer、negative count、positive draw budget、epoch 与 exact evaluation 不变后，Buy-only 达到 V0 的约 3.11 倍。Multi-behavior 比 V0 好但不如 Buy-only，说明额外行为并不自动转化为 purchase-aligned signal。

## Seed Replication

预注册的唯一 alternate seed 是 2027。它复用 V1 的全部配置，只改变随机种子与独立输出目录。Recall@50=`0.008405`，仍明显高于 V0，但比 seed42 的 `0.011966` 低约 29.76%。正确结论是：

> The Buy-only improvement replicated directionally under a preregistered alternate seed.

一次 replication 支持方向，但不能声称严格稳定。

## Multi-stage Recall

固定 quota 为 ItemCF Top-200、Two-Tower Top-100、Popularity Top-50，去重后每个用户最多 350 candidates。候选表保留 source flag、source rank、source score、source count 与 RRF score。Validation 与 test candidates 均先以 label-free 接口生成并落盘，再做离线评估。

Two-Tower 与 ItemCF 相比只有 5 个 test exclusive target hits，且全部为 Head；Tail exclusive hits=0。它提供少量互补，但现有证据不支持把它称为 Tail recall source。

## Candidate Fusion

项目比较了 source-level ranking、union oracle、RRF 与 ItemCF-first。融合阶段的关键原则不是“候选越多越好”，而是保留 provenance 并让后续 ranker 学习不同 source 的可信度。低 overlap 只说明集合不同，不代表不同部分包含更多正样本。

## RRF Failure

Validation ItemCF Recall@50=`0.371696`，naive RRF 只有 `0.179821`。RRF 假设不同 source 的 rank 有类似含义，但这里 ItemCF 明显强于 Popularity 与 Two-Tower；把弱源高 rank 与强源高 rank 近似等价会稀释 ItemCF。

这个失败说明经典算法仍需满足数据与 calibration 假设。最终系统保留 RRF 特征供 ranker 学习，但不直接把 naive RRF 当最终排序。

## Ranking

Ranker 使用 LightGBM LambdaRank，共 52 features：

- recall/source 特征；
- user train-only 行为与 recency；
- item train-only popularity、purchase 与 category；
- user-item count、behavior 与 recency；
- user-category affinity。

Raw `user_id` / `item_id` 不作为连续数值特征。Validation candidates 中只保留至少有一个正样本的 549 queries 进行训练，共 188,875 rows / 659 positives。保存的 model 已 reload 验证预测一致。

| Method | R@10 | R@20 | R@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| ItemCF | 0.080257 | 0.124298 | 0.241398 | 0.075433 |
| Ranker | 0.166771 | 0.202315 | 0.238792 | 0.124189 |

Ranker 把 early recall 与 NDCG 大幅提高，但 R@50 略低于 ItemCF。不能表述为所有指标全面领先。

## Candidate Oracle

Test-warm candidate oracle=`0.255548`，Ranker@50=`0.238792`，gap=`0.016756`。即使拥有理想排序器，固定候选集也只能达到 oracle。继续堆更复杂的 ranking 模型只能争取这 1.68 个百分点；增加真正互补的 target coverage 更重要。

## Cold Start

2,512 个 test purchase interactions 中，962 个 target 从未在 train 出现，占 38.30%。对于依赖 train item ID 的 ItemCF、Two-Tower、Ranker 和 collaborative SID，closed-catalog ceiling=`0.617038`。Ranker 全 test interaction Recall@50=`0.154061`；historical macro Recall@50=`0.159307`。

当前工作只是诊断，不是解决方案。未来需要在预测时可获得的 item metadata、图像或文本 encoder，并重新做 temporal leakage audit。

## Long Tail

Head/Torso/Tail 仅按 train interaction frequency 划分：前 20%、随后 30%、剩余 50%。Cold item 单列，不被错误归入 Tail。

| Segment | Ranker R@50 | Oracle R@50 | Gap |
|---|---:|---:|---:|
| Head | 0.271739 | 0.290217 | 0.018478 |
| Torso | 0.246291 | 0.261128 | 0.014837 |
| Tail | 0.184300 | 0.197952 | 0.013652 |

Tail 的 ranker-oracle gap 并不特别大，但 oracle 本身最低，因此主问题是 retrieval coverage。Exposure 进一步显示 Two-Tower Head=`84.99%`、Tail=`0.36%`，Tail Recall@50 与 exclusive hits 都为 0。

## Generative Recommendation

实验分支流程是：V1 collaborative item embedding → RQ-VAE → unique Semantic ID → behavior-aware causal sequence → Transformer encoder-decoder → trie-constrained beam search → standalone/incremental evaluation。

它是 TIGER-style extension，不是 exact TIGER。没有 content representation，所有 SID 都来自 train-known collaborative item。因此它不能支持 true cold start。

## RQ-VAE

输入为 `(302016,64)` L2-normalized V1 item embeddings。RQ-VAE 使用 3 个 residual codebooks，每层 128 codes。Best epoch=15，validation reconstruction=`0.00188850`。三层 utilization 均为 100%，perplexity=`108.7562/114.9056/113.1856`，没有明显 codebook collapse。

这些是 tokenizer health signals，而不是 retrieval success 的替代指标。

## Semantic IDs

Raw `(c0,c1,c2)` 只有 248,774 个 unique tuples；对 302,016 items 而言 collision excess=53,242，rate=`17.6289%`，最大 collision group=11。若直接用 raw tuple，一个生成路径无法唯一映射 item。

修复方法是在每个 tuple group 内按 `item_id ASC` 排序并分配 suffix。最终 `(c0,c1,c2,suffix)` 的 max suffix=10，collision=0，item→SID→item round-trip 全部通过。Trade-off 是 suffix 只保证可逆，不提高前三层 semantic capacity。

## Transformer

训练集有 12,207 purchase-aligned strict-causal examples。历史包含 `<PV>/<FAV>/<CART>/<BUY>`、三层 code、suffix 和 `<SEP>`。Transformer 配置：d_model=128、4 heads、2 encoder layers、2 decoder layers。Best epoch=10，仅由 validation Recall@20、NDCG@20 tie-break 选择。

同 timestamp 事件没有可观测的真实先后顺序，因此按 timestamp batching：先为该秒 targets 使用严格更早历史构造样本，再统一更新 history。这样牺牲同秒潜在信息，换取明确因果边界。

## Constrained Retrieval

Prefix trie 包含完整 302,016-item SID catalog。Beam size=50。Validation/test valid generation rate 均为 100%，invalid/incomplete paths=0，duplicate item=0。

Trie 只约束输出合法性。Test Top-50 只覆盖 14,137 unique items，catalog coverage=4.45%，Gini=`0.988493`，说明合法输出仍可能高度集中。

## Incremental Recall

| Split | Gen R@20 | Gen R@50 | Union before | Union + Gen | Exclusive hits |
|---|---:|---:|---:|---:|---:|
| Validation | 0.063319 | 0.065456 | 0.384716 | 0.386426 | 2 |
| Test | 0.042458 | 0.042458 | 0.255548 | 0.255548 | 0 |

Validation 两个 exclusive targets 都是 Head。Test 相对 ItemCF=0、相对 current union=0、Tail=0。Test exposure 为 Head 78.08%、Torso 19.32%、Tail 2.60%。因此 GenRec 工程上完成，但没有达到“可加入第四路 recall”的系统级标准。

## Failure Analysis

1. **PV-only Two-Tower:** objective 与 purchase evaluation 不一致。
2. **Multi-behavior:** 简单 weighting 没有超过 Buy-only，并且 exposure 仍集中。
3. **RRF:** source strength/rank calibration 不一致，融合反而破坏强 ItemCF。
4. **Two-Tower Tail:** Tail Recall 和 exclusive hits 都为 0，假设不成立。
5. **GenRec:** quantizer health、100% valid generation 与系统增量是三个不同层次；validation 的 2 个新增 target 未在 test 复现。
6. **Closed catalog:** collaborative models 对 38.30% train-cold test targets 结构上无能为力。

失败结果全部保留，因为它们定义了下一轮研究应解决的真实瓶颈，也展示了不以 test-driven tuning 掩盖负结果的实验纪律。

## Limitations

- 主实验使用确定性 1% 用户样本，没有报告 full 100M training 的 scaling curve。
- 数据缺乏丰富 item metadata，无法完成 content-based cold start。
- Two-Tower 只使用 ID representation，表达能力与泛化边界受限。
- Ranker 只在固定候选集上工作，无法找回 absent candidates。
- GenRec 训练 purchase examples 只有 12,207，且目标分布高度偏 Head。
- 当前 test 已被最终诊断使用，不能继续用来 model selection。
- FAISS/ANN 没有实际运行；现有报告只含 chunked exact retrieval。

## Future Work

1. 建立新的 temporal validation fold 或 rolling evaluation。
2. 优先寻找 warm complementary retrieval，并以 incremental oracle 为进入 union 的门槛。
3. 使用 metadata/content encoder 处理 true cold items。
4. 在新 validation protocol 上研究 Tail-aware generative training。
5. 改进 negative sampling 与 exposure/popularity debiasing。
6. 将 exact baseline 与 FAISS/ANN 的 recall、latency、memory 做工程 benchmark。

不应继续根据当前 test 调参以获得更高分。

## Interview Talking Points

- 数据与评估比模型名字更重要：user-level sampling、temporal split、label-access contract。
- 需要同时看 ranking metric 与 candidate oracle，才能定位系统瓶颈。
- ItemCF 胜过 Two-Tower 不是反常，而是数据稀疏度、目标设计和 inductive bias 的结果。
- “不同候选”不等于“互补 target”；应报告 exclusive target hits。
- 失败实验可以形成清晰的诊断链：hypothesis → evidence → decision → trade-off。
- 生成合法 Semantic ID 只是工程正确性；互补 recall 才是系统价值。

## Interview Story 1 — Mapping Recovery

**Situation:** 已有 V0 checkpoint，但训练时没有单独持久化 user/item mapping。  
**Problem:** embedding row 与真实 ID 无法可靠对应；直接猜 mapping 会让 retrieval 结果全部失真。  
**Diagnosis:** 回读训练 Notebook 的实际 cell 和 checkpoint tensor shape，定位 vocabulary 的构建语句与 stable order。  
**Fix:** 从 train `pv` rows 重放 `(user_id,item_id)` 首次出现顺序，得到 users=9,836、items=302,016；用 checkpoint shape、训练文件 hash 与 strict load 共同验证，并持久化 CSV/metadata。  
**Result:** 成功恢复 V0 embedding、导出矩阵并运行 exact retrieval，后续所有 controlled experiments 共用清晰 mapping contract。  
**Trade-off:** 这是确定性恢复，但比训练时原生保存 mapping 更脆弱；因此后续 checkpoint 必须绑定 mapping reference 和 config。

## Interview Story 2 — Objective Mismatch

**Situation:** PV-only Two-Tower 的 sampled training/validation loss 正常，但 purchase Recall 很差。  
**Problem:** 优化高频浏览点击不等于优化未来购买，loss 又只在 sampled negatives 上计算。  
**Diagnosis:** 固定 architecture、budget、negatives 与 exact full-catalog protocol，只改变 positive definition。  
**Fix:** 构建 Buy-only V1，并预注册一次 seed2027 replication。  
**Result:** seed42 Recall@50=`0.011966`，约为 V0 `0.003846` 的 3.11×；seed2027=`0.008405`，方向复现但有 29.76% relative drop。  
**Trade-off:** purchase positives 更稀疏，variance 更高；结论限于方向支持，不能声称严格稳定。

## Interview Story 3 — Candidate Oracle

**Situation:** LambdaRank 显著提升了 Recall@10 和 NDCG，但考虑继续增加更复杂 ranking 模型。  
**Problem:** 不清楚损失来自排序错误还是候选根本缺失。  
**Diagnosis:** 计算忽略内部顺序的 candidate oracle。  
**Fix:** 对同一 warm population 比较 oracle `0.255548` 与 Ranker@50 `0.238792`。  
**Result:** gap 只有 `0.016756`；Head/Torso/Tail 的 segment gap 也较小，真正上限来自 retrieval coverage。  
**Trade-off:** 复杂 ranker 仍可能改善 early ordering，但无法找回 absent target；资源应先投入互补召回。

## Interview Story 4 — RRF Failure

**Situation:** 三路 recall overlap 低，经典 RRF 看似适合做无监督融合。  
**Problem:** RRF validation Recall@50 从 ItemCF `0.371696` 降到 `0.179821`。  
**Diagnosis:** ItemCF 远强于其他 source，但 RRF 只看 rank，弱源高位被赋予过高影响。  
**Fix:** 不把 RRF 当最终规则；保留 source provenance/rank/score，由 LambdaRank 学习条件化权重。  
**Result:** Ranker 提升 test early recall 与 NDCG，同时诚实保留 ItemCF R@50 略高的事实。  
**Trade-off:** supervised fusion 需要标签与 leakage-safe feature pipeline；无监督 RRF 更简单但假设不成立。

## Interview Story 5 — GenRec Failure

**Situation:** 希望用 collaborative Semantic IDs 增加传统 union misses，尤其 Tail。  
**Problem:** 需要区分 tokenizer、generation 和 system value 三个层次。  
**Diagnosis:** RQ-VAE reconstruction=`0.00188850`、三层 utilization=100%；trie generation valid=100%，但单独计算 incremental oracle 和 segment hits。  
**Fix:** 保留 17.63% raw tuple collision excess 作为量化诊断，并用按 item ID 确定性排序的 suffix 使 final SID mapping 一一对应、final collision=0；再用 trie 保证合法输出。模型只按 validation 选择，并在冻结后执行一次 test。  
**Result:** validation 新增 2 个 Head targets，test 新增 0、Tail 新增 0；不加入正式 union。  
**Trade-off:** 工程链路可复现且可作为研究 baseline，但继续在当前 test 上调 beam/codebook/weight 会构成 test-driven optimization。这个案例说明 representation quality、generation validity 与 complementary recall 是三个不同问题。
