# Project Issues

## ISSUE-001 — Two-Tower checkpoint did not contain ID mappings

- **Problem:** `two_tower_best.pt` stored embedding weights indexed by continuous IDs, but did not store the original `user_id/item_id` correspondence.
- **Evidence / How we found it:** The checkpoint contains `model_state_dict`, vocabulary sizes and model dimensions only. No mapping key or sidecar mapping existed.
- **Root cause / hypothesis:** Training created mappings in notebook memory and saved only model state and scalar configuration.
- **Fix / Decision:** Reconstructed mappings using the exact training logic and row order: filter PV, select `user_id/item_id`, drop duplicate pairs keeping first occurrence, then use pandas `unique()` order. Validated recovered sizes against both the documented values and checkpoint metadata before retrieval. Saved CSV mappings plus the train CSV SHA-256 and reconstruction method.
- **Result:** Recovered and checkpoint vocabularies both equal 9,836 users and 302,016 items; strict state-dict loading succeeds. Mappings are persisted under `artifacts/two_tower/`.
- **Trade-off:** Recovery still relies on the current training data bytes and the original pandas ordering semantics. Future checkpoints should store mappings or immutable mapping references at training time.

## ISSUE-002 — Training objective and purchase retrieval evaluation are misaligned

- **Problem:** The Two-Tower was trained and validated on PV pairs, while the recommendation target is validation-day purchases.
- **Evidence / How we found it:** `validation_ground_truth.csv` was loaded by the training notebook but not used after display. The saved validation loss is sampled PV-vs-random-negative BCE, not purchase Recall/NDCG. On the fair warm purchase set, Two-Tower Recall@50 is 0.003846 versus ItemCF 0.371696.
- **Root cause / hypothesis:** The first neural baseline optimized click similarity without a retrieval-aware purchase validation loop or multi-behavior objective.
- **Fix / Decision:** Kept V0 unchanged, then ran two separately versioned controlled experiments with the same shared vocabulary/model capacity/training-step budget: buy-only positives and behavior-weighted positives. Model selection for the new experiments uses exact warm purchase Recall@50.
- **Result:** V0/V1/V2 Recall@50 is 0.003846/0.011966/0.007058. Both purchase-aligned objectives improve over V0, so the experiment supports the mismatch hypothesis. It does not prove single-factor causality because V0's historical model-selection rule cannot be retroactively changed and V1 requires heavy positive resampling.
- **Trade-off:** The current V0 checkpoint remains a reproducible baseline, but is not suitable as the production retrieval model. A confirmatory seed is required before adopting V1.

## ISSUE-003 — Retrieved candidates are strongly popularity-collapsed

- **Problem:** Different users receive nearly identical head-item recommendations.
- **Evidence / How we found it:** Across 1,170 warm evaluation users there are only 2 unique Top-1 items; item 3,845,720 is Top-1 for 91.97% of users. The union of all user Top-50 lists contains only 301 of 302,016 candidate items.
- **Root cause / hypothesis:** Likely contributors are uniform random negatives, an ID-only architecture, the PV-to-purchase mismatch, and only three training epochs. These are hypotheses, not yet isolated causal findings.
- **Fix / Decision:** Do not conceal the result or proceed to Ranking. Tested separately versioned buy-only and weighted multi-behavior objectives under the same candidate catalog and exact evaluation protocol.
- **Result:** Buy-only reduces dominant Top-1 share from 91.97% to 5.38% and increases unique Top-50 exposure from 301 to 3,934 items. Weighted multi-behavior remains concentrated: 81.20% dominant Top-1 share and only 688 unique Top-50 items.
- **Trade-off:** More informative or harder negatives increase training cost and require careful false-negative handling.

## ISSUE-004 — ID-only PV vocabulary excludes a large part of purchase ground truth

- **Problem:** The current model cannot represent unseen users or purchase targets absent from the train-PV vocabulary.
- **Evidence / How we found it:** Validation GT contains 1,760 users and 2,540 interactions. There are 16 unseen users and 1,032 unseen-target interactions (1,023 unique unseen target items). The fully warm evaluation set contains 1,170 users (66.48% coverage) and 1,496 interactions (58.90% coverage).
- **Root cause / hypothesis:** Both towers use ID embeddings only, and the item vocabulary is built exclusively from training PV pairs.
- **Fix / Decision:** Evaluate all three baselines on the identical fully warm purchase target set and report coverage separately. Do not treat cold targets as retrievable by this model.
- **Result:** Warm-start metrics are fair and reproducible, with coverage limitations explicit.
- **Trade-off:** Warm-only metrics describe model quality only where embeddings exist and must not be presented as all-traffic performance.

## ISSUE-005 — Sampled validation loss is stochastic and can contain false negatives

- **Problem:** Validation negatives are resampled during every dataset access, can repeat within a sample, and exclude only train-PV positives plus the current positive item.
- **Evidence / How we found it:** `TwoTowerDataset.sample_negative` uses an unbounded random loop; validation reuses the training positive dictionary. Another validation positive or a train cart/fav/buy item can therefore be labelled negative.
- **Root cause / hypothesis:** The same simple online uniform sampler was reused for training and validation.
- **Fix / Decision:** V1/V2 use one fixed validation sample/seed. Each fixed negative set excludes all mapped train interactions and all validation purchase positives. Exact warm purchase retrieval remains the model-selection and final-comparison protocol.
- **Result:** Resolved for the new controlled experiments. V0's historical sampled validation loss remains unchanged and is retained only as baseline evidence.
- **Trade-off:** Broader positive exclusion and fixed validation candidates require more memory or preprocessing.

## ISSUE-006 — Notebook paths depend on the launch directory

- **Problem:** Earlier notebook cells use `Path('..')` or `../data/...`, which fail if the working directory is the project root instead of `notebooks/`.
- **Evidence / How we found it:** The saved training output assumes the notebook-directory working directory, while the baseline notebook already contains a separate cwd workaround.
- **Root cause / hypothesis:** Paths were authored interactively without one shared project-root rule.
- **Fix / Decision:** The Retrieval V2 section resolves the project root from either the project root or `notebooks/`. Earlier successful cells were left unchanged to preserve their history.
- **Result:** New retrieval cells no longer depend on one launch location.
- **Trade-off:** The old data-building and training cells remain cwd-sensitive until a later notebook-cleanup pass.

## ISSUE-007 — FAISS consistency check is unavailable in the current environment

- **Problem:** A FAISS `IndexFlatIP` comparison cannot run with the current environment.
- **Evidence / How we found it:** The `faiss` module and `faiss-cpu` distribution are absent.
- **Root cause / hypothesis:** FAISS was never added as a project dependency.
- **Fix / Decision:** Did not mutate the environment during this round. Kept chunked exact inner-product retrieval as the correctness baseline and did not implement IVF/HNSW.
- **Result:** Exact Top-K completed; no FAISS timing or result-consistency claim is made.
- **Trade-off:** There is no alternative-index benchmark yet, but no optional dependency was introduced merely for a redundant exact-search implementation.

## ISSUE-008 — Buy-only positives are sparse and require heavy resampling

- **Problem:** A buy-only objective has far fewer mapped training pairs than the PV baseline.
- **Evidence / How we found it:** Only 11,844 mapped buy pairs are available, covering 5,141 users and 10,819 items. Only 712 of 1,170 warm evaluation users have a training-period buy. Matching V0's 521,287 samples per epoch repeats each buy pair about 44 times on average.
- **Root cause / hypothesis:** Purchases are intrinsically sparse and 2,368 train buy pairs fall outside the fixed PV-derived vocabulary.
- **Fix / Decision:** Used uniform sampling with replacement to keep the optimizer-step budget controlled and reported the resampling factor explicitly. No architecture expansion or parameter sweep was introduced.
- **Result:** Best epoch 3 reaches Recall@50 0.011966, HitRate@50 0.012821 and NDCG@50 0.003453, but training loss falls to 0.040999 while the fixed sampled diagnostic loss rises to 2.208330.
- **Trade-off:** The controlled step budget improves comparability but creates overfitting risk and leaves users without historical buys effectively untrained beyond shared tower parameters and initialization.

## ISSUE-009 — Sampled loss and full-catalog retrieval metrics disagree

- **Problem:** Sampled BCE loss is not a reliable model-selection proxy for exact purchase retrieval.
- **Evidence / How we found it:** For V1, fixed sampled loss worsens from 1.362624 to 2.208330 while Recall@50 improves from 0.003419 to 0.011966. For V2, sampled loss improves again at epoch 3, but Recall@50 declines from 0.007058 to 0.005918.
- **Root cause / hypothesis:** Sampled BCE measures separation from four fixed random negatives, whereas full-catalog Recall measures ranking among 302,016 candidates.
- **Fix / Decision:** Select V1/V2 checkpoints by exact warm purchase Recall@50, using NDCG@50 as the tie-breaker. Retain sampled loss only as a diagnostic.
- **Result:** V1 selects epoch 3; V2 correctly retains epoch 2 instead of overwriting it with the lower-loss epoch 3 model.
- **Trade-off:** Exact model selection adds a full embedding/export and catalog retrieval pass per epoch, but on this dev dataset each pass is inexpensive relative to training.

## ISSUE-010 — The first multi-behavior weighting still favors popularity collapse

- **Problem:** The initial explainable weighting improves Recall over V0 but retains severe recommendation concentration.
- **Evidence / How we found it:** Data-derived square-root-lift weights are PV 1.0000, fav 1.9699, cart 3.2857 and buy 2.4039. The best V2 Recall@50 is 0.007058, but item 3,845,720 remains Top-1 for 81.20% of users and all Top-50 lists expose only 688 items.
- **Root cause / hypothesis:** PV pairs remain overwhelmingly numerous, so modest per-pair weights may not sufficiently change the effective objective. This is a hypothesis; no weight sweep was performed.
- **Fix / Decision:** Preserved the result and stopped after the single predeclared weighting strategy. Do not tune multiple weights post hoc in this round.
- **Result:** V2 beats V0 but loses to Buy-only on Recall, NDCG and recommendation diversity.
- **Trade-off:** The no-sweep decision keeps the comparison interpretable but does not identify an optimal multi-behavior mix.

## ISSUE-011 — Buy-only improves directionally across two seeds but retains numerical seed variance

- **Problem:** The original Buy-only result used only seed 42, so its improvement over V0 could have been caused by one favorable initialization or sampling sequence.
- **Evidence / How we found it:** Ran exactly one preregistered replication, `TT_V1_BUY_UNIFORM_SEED2027`, using the same train data SHA, 11,844 mapped buy pairs, shared mappings, model architecture, optimizer, learning rate, batch size, four uniform negatives, 521,287 positive draws per epoch, three epochs, fixed validation negatives and exact warm purchase checkpoint-selection protocol. Seed2027 selected epoch 3 with Recall@50 0.008405, HitRate@50 0.010256 and NDCG@50 0.001927. V0 Recall@50 is 0.003846 and seed42 V1 Recall@50 is 0.011966.
- **Root cause / hypothesis:** Buy-only is a better-aligned positive objective for purchase retrieval, while sparse positives, repeated sampling and random initialization still create material run-to-run variance.
- **Fix / Decision:** Preserve both runs without post-hoc reruns or tuning. Record the formal conclusion: `Buy-only improvement replicated directionally under an alternate random seed.` Do not interpret two seeds as a strict stability proof.
- **Result:** Seed2027 remains 2.185x above V0, but is 0.003561 lower than seed42, a 29.76% relative decrease. It does not show the V0 collapse pattern: dominant Top-1 share is 2.05%, there are 424 unique Top-1 items, and Top-50 catalog coverage is 1.37%. The original V0/V1/V2 protected artifacts were unchanged by SHA-256 verification, and the new checkpoint strict-loads successfully.
- **Trade-off:** Directional reproducibility is sufficient to retain Buy-only as an auxiliary recall candidate and begin Multi-stage Recall evaluation, but its absolute metric variance and large gap to ItemCF mean it should not yet be treated as a stable primary retriever or justification for entering Ranking.

## ISSUE-012 — Multi-stage union has low source overlap but only a small oracle gain

- **Problem:** Combining three low-overlap recall sources creates a large candidate table but adds few relevant purchase targets beyond the strong ItemCF source.
- **Evidence / How we found it:** Validation contains 3,304,863 deduplicated candidate pairs for 9,719 users, averaging 340.04 candidates per user. Pairwise candidate intersections are ItemCF/Two-Tower 3,561, ItemCF/Popularity 3,208 and Two-Tower/Popularity 20,336, with Jaccard scores 0.001253, 0.001358 and 0.014203. On 1,496 warm validation targets, ItemCF@200 hits 561 and the full union hits 573. Two-Tower retrieves 9 targets missed by ItemCF; only 6 are Two-Tower-only and 3 are also retrieved by Popularity. Popularity alone retrieves 3 targets missed by both other sources.
- **Root cause / hypothesis:** Two-Tower and Popularity provide diverse candidates, but most of that diversity is irrelevant to the purchase target. The current Two-Tower remains weak, and fixed popularity candidates have little personalized incremental value.
- **Fix / Decision:** Preserve all source provenance and report union oracle before training a ranker. Keep ItemCF as the primary source, Two-Tower as an auxiliary source and Popularity as fallback. Do not tune quotas or RRF on test labels.
- **Result:** Validation warm oracle Recall improves from ItemCF@200 0.377329 to union 0.384716, a mean-Recall gain of 0.007387. Test warm union oracle Recall is 0.255548; all-GT oracle Recall is 0.168837. The candidate union is valid and reproducible, but its incremental target coverage is modest.
- **Trade-off:** Large low-overlap candidate sets increase downstream storage and ranking cost without guaranteeing relevant coverage. A new recall source should be judged by incremental oracle hits, not raw candidate diversity.

## ISSUE-013 — Naive RRF cannot combine a dominant strong source with two weak sources

- **Problem:** Equal-formula Reciprocal Rank Fusion over-promotes high-ranked items from weak sources and substantially degrades ItemCF's strong ordering.
- **Evidence / How we found it:** On validation warm GT, ItemCF Recall@10/20/50 is 0.108468/0.189158/0.371696 with NDCG@50 0.110981. Fixed RRF with constant 60 produces 0.046610/0.078169/0.179821 and NDCG@50 0.052453. ItemCF-first exactly preserves ItemCF's Top-50 metrics because all ItemCF Top-200 candidates are placed before supplemental candidates.
- **Root cause / hypothesis:** Source ranks are not equally calibrated: ItemCF is much stronger than the other sources, while unweighted RRF treats a high TT or Popularity rank as comparable evidence.
- **Fix / Decision:** Keep RRF only as a transparent failed baseline. Use ItemCF-first when ordering must not damage the strong source, but do not claim it improves Top-50. Do not tune RRF weights or its constant in this stage.
- **Result:** ItemCF-first protects the baseline but adds no Top-50 gain; RRF is not suitable as the current production ordering.
- **Trade-off:** Learning a ranker could calibrate heterogeneous sources, but the union oracle ceiling is only 0.384716, so retrieval coverage should improve before ranking work is prioritized.

## ISSUE-014 — Cold targets and candidate oracle show retrieval is the current bottleneck

- **Problem:** A large share of purchase targets cannot be retrieved from the train catalog, and most warm targets are still absent from the three-source union.
- **Evidence / How we found it:** Validation has 894/2,540 cold-target interactions absent from train (35.20%); test has 962/2,512 (38.30%). Even on the restricted warm set, validation union misses 923/1,496 targets and test union misses 1,072/1,438. Validation warm oracle Recall is 0.384716 and test warm oracle Recall is 0.255548. Popularity-only fallback covers the users lacking both personalized sources, but this affects only 2 validation and 3 test users.
- **Root cause / hypothesis:** The current catalog and ID-only models cannot represent unseen target items, while the available warm recall sources do not cover enough purchase intent.
- **Fix / Decision:** Report warm and all-GT oracle separately, preserve `unretrievable_cold_target_interactions`, and use the fixed existing union for one leakage-safe ranking baseline. Test labels remain evaluation-only and are parsed only after test scores are produced.
- **Result:** The completed ranker reaches test warm Recall@50 0.238792 against a union oracle of 0.255548, leaving only 0.016756 mean-Recall headroom inside the existing candidates. All-GT Ranker Recall@50 is 0.159307 against all-GT oracle 0.168837. This confirms retrieval/candidate generation and catalog coverage remain the main bottlenecks; a ranker cannot recover targets absent from the union.
- **Trade-off:** Adding content or metadata recall could improve cold/long-tail coverage but expands the project beyond the current ID-only scope and must be evaluated as a separately controlled stage. Ranking improvements can still change early precision/order, but cannot raise the candidate oracle.

## ISSUE-015 — Ranker improves early ordering and NDCG but slightly loses Recall@50

- **Problem:** The full-feature LambdaRank baseline does not dominate the strong ItemCF ordering at every cutoff.
- **Evidence / How we found it:** On the untouched test warm set, ItemCF Recall@10/20/50 is 0.080257/0.124298/0.241398 and NDCG@50 is 0.075433. The fixed full-feature ranker reaches 0.166771/0.202315/0.238792 and NDCG@50 0.124189. Thus it more than doubles Recall@10 and materially improves Recall@20 and NDCG, but Recall@50 is 0.002606 lower (-1.08% relative). The recall-only ablation reaches Recall@50 0.243302 but is weaker at Recall@20 (0.155914) and NDCG@50 (0.079950).
- **Root cause / hypothesis:** Train-only user-item and category features strongly prioritize a smaller set of historically plausible candidates near the top. This improves head ordering but can push a few candidate positives below rank 50. `seen_in_train` alone contributes 58.09% of gain importance; this is descriptive evidence, not a causal proof.
- **Fix / Decision:** Preserve the full-feature model as the preregistered primary baseline and preserve the recall-only model as an ablation. Do not use the test result to change features, tree count, loss, or tie-breaking. Any attempt to recover Recall@50 while retaining head gains requires a new validation protocol or future temporal fold.
- **Result:** Ranking is useful for Top-10/20 and NDCG, but the statement "Ranker beats ItemCF" is only true at those early-order metrics, not at Recall@50 or HitRate@50. The full ranker is close to the candidate oracle, so its Top-50 loss is secondary to the much larger retrieval ceiling.
- **Trade-off:** Optimizing deeper-list recall may weaken early ranking quality. A single scalar objective or cutoff cannot be selected retrospectively from this test set.

## ISSUE-016 — Validation labels train the ranker, so no independent ranker-selection split remains

- **Problem:** The requested temporal protocol uses validation candidates and validation purchase labels for ranker training, leaving test as the only independent future period and no separate ranker early-stopping or hyperparameter-selection set.
- **Evidence / How we found it:** The full validation candidate table has 3,304,863 rows and 659 positive candidate rows. LambdaRank training uses all candidates for the 549 queries containing at least one positive, or 188,875 rows. Validation training-fit Recall@50 is 0.384638, essentially the 0.384716 validation oracle, so it is explicitly not a generalization estimate.
- **Root cause / hypothesis:** The project has one train period for aggregate features/retrievers, one validation period assigned to ranker labels, and one protected test period. Creating an extra tuning split after seeing test would violate the current protocol.
- **Fix / Decision:** Use one fixed moderate LightGBM configuration (200 trees, learning rate 0.05, 31 leaves, seed 42), no early stopping, no grid search and no test-driven threshold or feature selection. Treat the recall-only test ablation as descriptive analysis only, not model selection.
- **Result:** The first end-to-end baseline has a valid one-shot test estimate and reproducible artifacts, but it does not support tuning claims or selection among many ranker configurations.
- **Trade-off:** A future ranking iteration should create an earlier temporal fold or rolling validation scheme before touching the protected test again; doing so requires rebuilding the time-sliced training protocol rather than retrofitting this result.

## ISSUE-017 — Test purchase 中的 cold item 造成 closed-catalog 结构性上限

- **Problem**
  - 现有 Popularity、ItemCF、Two-Tower 与 Ranker 都只能推荐训练期已知 item，无法命中 test 中首次出现的 item。
- **Evidence / How we found it**
  - test purchase ground truth 共 2,512 个 interaction，其中 962 个（38.2962%）target item 不在完整 train catalog。
  - 因此 full-test、Top-50、closed-catalog 的 interaction-level 理论上限只有 `1550 / 2512 = 0.617038`。
  - Ranker 在全量 GT 上命中 387 个 interaction：interaction-level Recall@50=`0.154061`；历史项目口径的 macro Recall@50=`0.159307`。
  - 固定 candidate union 在全量 GT 上命中 413 个 interaction，interaction-level oracle=`0.164411`。
- **Root cause / hypothesis**
  - 这是 catalog representation 问题，不是排序器或分数融合问题。协同过滤与纯 ID embedding 对从未出现在训练期的 item 没有表示。
- **Fix / Decision**
  - 本轮只量化 cold-item bucket、closed-catalog ceiling 和可达上限，不伪称已解决 Cold Start。
  - 后续若要解决，需要引入测试时可用的 item metadata/content 表示，并单独设计 cold-item evaluation。
- **Result**
  - Ranker 达到 closed-catalog interaction ceiling 的 24.9677%；candidate union 达到 26.6452%。
  - train-known 但 candidate union 未命中的 target 有 1,137 个，多于 cold-item 的 962 个，因此当前最大绝对损失仍来自 warm retrieval。
- **Trade-off**
  - 引入内容特征可扩大可推荐 catalog，但需要新的数据依赖、特征时点审计和独立验证协议。

## ISSUE-018 — Tail target 的召回能力显著弱于 Head

- **Problem**
  - 训练期已知 item 中，Tail purchase target 的可召回性明显较低。
- **Evidence / How we found it**
  - 训练 catalog 按 train interaction count 降序、`item_id` 升序确定性分段：Head 前 20%，Torso 接续 30%，Tail 后 50%。
  - test GT 分布为 Head 920、Torso 337、Tail 293、Cold 962。
  - Ranker Recall@50：Head=`0.271739`、Torso=`0.246291`、Tail=`0.184300`。
  - 固定 candidate union oracle：Head=`0.290217`、Torso=`0.261128`、Tail=`0.197952`。
- **Root cause / hypothesis**
  - Tail item 在 train 中通常只有一次交互，共现与 ID embedding 监督都很稀疏；这与 cold item 完全未出现是两类不同问题。
- **Fix / Decision**
  - 本轮不调参、不重训；按 segment 固定报告模型指标、oracle 和 gap。
- **Result**
  - Tail 是三个 train-known segment 中最难的部分，而且主要瓶颈位于 retrieval，而不是现有 ranker 的重排能力。
- **Trade-off**
  - 强行提高 Tail 曝光可能损伤相关性；后续新召回源必须用 segment oracle 与全局质量共同验证。

## ISSUE-019 — 不同召回源存在显著但方向不同的曝光偏置

- **Problem**
  - 只看整体 Recall 会掩盖推荐曝光是否过度集中在少数热门 item。
- **Evidence / How we found it**
  - 在 train-known test demand 中，Head/Torso/Tail 占比为 59.3548% / 21.7419% / 18.9032%。
  - ItemCF Top-50 曝光为 38.0587% / 26.2367% / 35.7046%，Gini=`0.531849`。
  - Two-Tower Top-50 曝光为 84.9870% / 14.6529% / 0.3601%，Gini=`0.996580`。
  - Ranker Top-50 曝光为 58.7106% / 19.9562% / 21.3332%，Gini=`0.695398`。
- **Root cause / hypothesis**
  - Two-Tower 的纯 ID、uniform negatives 与当前目标使输出集中在很小的热门集合；ItemCF 的局部共现候选更分散；Ranker 又对候选进行了偏向高相关性的重排。
- **Fix / Decision**
  - 同时保存 catalog coverage、popularity percentile、novelty、Gini 和相对 demand 的 amplification，不用单一“热门偏置”标签概括所有模型。
- **Result**
  - Two-Tower 明显放大 Head 曝光（+25.632pp）并压低 Tail（-18.543pp）。
  - ItemCF 相对 demand 并非 Head-biased，而是 Tail over-exposure（+16.801pp）。
  - Ranker 比 ItemCF 更 head-heavy，但相对实际 demand 基本校准：Head -0.644pp、Tail +2.430pp。
- **Trade-off**
  - Gini、coverage 和 segment mix 描述的是不同现象，不能互相替代；低 Gini 也不自动代表更高相关性。

## ISSUE-020 — Candidate oracle 的 segment gap 说明后续应优先改善 retrieval

- **Problem**
  - 需要区分各 segment 的损失主要来自候选未召回，还是候选已有但排序错误。
- **Evidence / How we found it**
  - Candidate oracle Recall@50 为 Head=`0.290217`、Torso=`0.261128`、Tail=`0.197952`。
  - Ranker 与 oracle 的差距分别为 0.018478、0.014837、0.013652。
- **Root cause / hypothesis**
  - 当前固定候选集本身已经限定了绝大多数可达上限；ranker 只留下较小的 segment 内 gap。
- **Fix / Decision**
  - 下一阶段先验证新的、互补的 recall source，再考虑继续扩大 ranker 复杂度。
- **Result**
  - Head、Torso、Tail 都以 retrieval coverage 为主要瓶颈，Tail 的 oracle 绝对值最低。
- **Trade-off**
  - 新召回源只有在提供增量 target hits、且不过度恶化候选规模与线上成本时才值得保留。

## ISSUE-021 — Two-Tower 没有提供 Tail 增量命中

- **Problem**
  - 需要确认 Two-Tower 在融合中是否真的补充了长尾 target，而不是只增加热门候选。
- **Evidence / How we found it**
  - 使用固定 quota 的 Two-Tower Top-100 与 ItemCF Top-200 比较：Head/Torso/Tail target hits 为 11/1/0；Two-Tower exclusive hits 为 5/0/0。
  - 全量 2,512 个 target 中，Two-Tower 共命中 12 个，仅 5 个是 ItemCF 未命中的增量命中，且全部位于 Head。
- **Root cause / hypothesis**
  - 当前 Two-Tower 的曝光高度集中，Tail exposure 只有 0.3601%，不足以形成长尾补充。
- **Fix / Decision**
  - 保留 Two-Tower 的真实融合结果，但不再把它描述为 Tail recall source。
- **Result**
  - 现有证据不支持 Two-Tower 改善 Tail；它的少量增量贡献全部来自 Head。
- **Trade-off**
  - 删除 Two-Tower 可能损失少量 Head 增量命中；保留则增加计算与候选去重成本，需要由后续新召回实验统一比较。


## ISSUE-022 — 三层 Semantic-ID raw tuple 存在不可忽略的 collision

- **Problem**
  - RQ-VAE 的 `(c0,c1,c2)` 不是 item-level 唯一 ID，直接作为生成目标会让一个 token path 对应多个 item。
- **Evidence / How we found it**
  - 302,016 个 active items 只产生 248,774 个 unique raw tuples；collision excess items=53,242，collision excess rate=`17.6289%`。
  - 共有 41,615 个 collision groups，最大 group size=11，collision-group mean size=2.2794。
- **Root cause / hypothesis**
  - 3×128 的离散空间虽大，但 RQ-VAE 优化目标是 reconstruction，不保证 item-level injectivity；相似 collaborative embeddings 会共享 codes。
- **Fix / Decision**
  - 对每个 raw tuple group 按 `item_id ASC` 确定性排序，分配 suffix `0,1,...`，最终 SID 为 `(c0,c1,c2,suffix)`。
- **Result**
  - suffix max=10；302,016 个 final SIDs 全部唯一，item→SID→item round-trip 通过，final collision=0。
- **Trade-off**
  - suffix 解决可逆性，不增加前三层表示能力；同 tuple item 的区分仍主要由末位 token 承担。

## ISSUE-023 — RQ-VAE 没有 collapse，但 reconstruction 与 retrieval 价值不能画等号

- **Problem**
  - Codebook collapse 会让 Semantic-ID tokenizer 失效；反过来，高 utilization 也不保证下游 retrieval 有增量价值。
- **Evidence / How we found it**
  - Best epoch=15，validation reconstruction loss=`0.00188850`。
  - Level 0/1/2 utilization 均为 128/128=100%；perplexity=`108.7562/114.9056/113.1856`；most-used share 均低于 2.1%。
  - Head/Torso/Tail 各自也都使用全部 128 codes。
- **Root cause / hypothesis**
  - Train-only MiniBatchKMeans sequential initialization和 residual VQ loss 避免了明显 collapse，但 tokenizer 仍只重构 V1 collaborative geometry。
- **Fix / Decision**
  - RQ-VAE checkpoint 只按 validation reconstruction loss 选择，utilization 仅作 tie-break/diagnostic；最终 retrieval 价值单独用 constrained Recall 和 incremental oracle 判断。
- **Result**
  - 没有 codebook collapse；但 test union oracle 增量仍为 0，证明“量化健康”不等于“召回互补”。
- **Trade-off**
  - 为追求下游指标直接调 tokenizer 会引入 validation/test 选择风险；当前固定结果保留不调参。

## ISSUE-024 — Constrained generation 合法，但合法输出仍可能高度集中

- **Problem**
  - 自回归模型可能生成无效 code path、重复 item，或者只在很小的热门集合内生成合法路径。
- **Evidence / How we found it**
  - 完整 302,016-item prefix trie、beam=50；validation/test valid generation rate 均为 100%。
  - Invalid/incomplete paths=0，duplicate items removed=0。
  - 但 test Top-50 只覆盖 14,137 unique items，full train catalog coverage=4.45%，Gini=`0.988493`。
- **Root cause / hypothesis**
  - Trie 只保证结构合法；purchase supervision 和 collaborative code 分布仍使概率质量集中于 Head paths。
- **Fix / Decision**
  - 保留 trie hard constraint，并把 validity 与 exposure/coverage 分开报告，不把 100% valid 误写成高质量 retrieval。
- **Result**
  - 工程合法性完全通过；推荐多样性与 Tail coverage 仍不足。
- **Trade-off**
  - 无约束生成可能增加表面 diversity，但会产生不可映射 SID；本项目不采用。

## ISSUE-025 — Generative incremental coverage 没有在 Test 复现

- **Problem**
  - 新 recall source 的核心价值是找回 ItemCF/current-union misses，而不是只获得非零单路 Recall。
- **Evidence / How we found it**
  - Validation：GenRec 命中 96 interactions；相对 ItemCF 和 current union 都新增 2 个 target；union oracle `0.384716→0.386426`，delta=`+0.001709`。
  - Test：GenRec 命中 64 interactions；相对 ItemCF/current union 新增均为 0；union oracle 保持 `0.255548`。
- **Root cause / hypothesis**
  - GenRec 学到的 target 大部分已被强 ItemCF/union 覆盖；validation 的两个 Head 增量过小，可能是时间段偶然性，未表现出稳健互补性。
- **Fix / Decision**
  - 不删除 validation 正信号，也不忽略 test 0 增量；当前不加入正式第四路 recall，不重训 Ranker。
- **Result**
  - 达到工程成功和非零单路 Recall，但没有达到 test 系统成功标准。
- **Trade-off**
  - 继续根据当前 test 调模型会造成 test-driven optimization；任何后续版本必须先建立新的 temporal validation fold。

## ISSUE-026 — Generative retrieval 没有改善 Tail

- **Problem**
  - 实验研究问题之一是 collaborative Semantic ID 是否能补充 Two-Tower/ItemCF 的 Tail misses。
- **Evidence / How we found it**
  - Validation Tail Recall@20/@50=`0/0`，Tail exclusive hits=0。
  - Test Tail Recall@20/@50=`0/0`，Tail exclusive hits=0。
  - Test exposure Tail share=2.60%，虽然高于 Two-Tower 的 0.36%，仍远低于 Head 78.08%。
- **Root cause / hypothesis**
  - Tail item 的 purchase supervision 稀疏；collaborative embedding 与 autoregressive target frequency 都偏向 Head。RQ-VAE 能量化 Tail，不代表 generator 能学习其条件概率。
- **Fix / Decision**
  - 如实保留 Tail 失败结果，不做 post-hoc reweighting、beam 调整或 test-driven oversampling。
- **Result**
  - 当前 GenRec 不能作为 Tail recall source。
- **Trade-off**
  - Tail reweighting 可能提高覆盖但损伤整体相关性，必须在新的 validation fold 预注册后研究。

## ISSUE-027 — Collaborative Semantic ID 仍受 closed-catalog 限制

- **Problem**
  - 容易把“生成 token”误解成能生成训练期从未出现的新 item。
- **Evidence / How we found it**
  - Active catalog 完全来自 V1 item mapping；test 中 1,059 个 targets 不在 active V1 catalog，其中 962 个是真正不在完整 train catalog 的 cold targets。
  - 没有 active SID 的 target 在生成式评估中结构上不可召回。
- **Root cause / hypothesis**
  - Semantic IDs 来自 train-trained collaborative item embeddings；无历史 item 没有 embedding，也没有合法 trie path。
- **Fix / Decision**
  - README/status 明确称为 TIGER-style collaborative extension，不称 exact TIGER 或 Cold Start solution；cold items 从未注入 semantic catalog。
- **Result**
  - GenRec 只研究 warm/long-tail representation 与 retrieval，Cold Start 仍未解决。
- **Trade-off**
  - 真正 cold-item 支持需要 content/metadata encoder、上线时可用特征与新的 leakage audit。

## ISSUE-028 — 同 timestamp 事件必须按批处理以保证严格因果历史

- **Problem**
  - 最初按 `(user_id,timestamp)` 排序后逐行构造 purchase history，会把同一秒较早出现的行误当成目标购买之前的历史。
- **Evidence / How we found it**
  - `history_timestamp < target_timestamp` 断言在首次序列构建时失败，训练在开始前被主动终止。
- **Root cause / hypothesis**
  - 数据 timestamp 只有秒级分辨率；稳定行顺序不代表同秒事件的真实先后关系。
- **Fix / Decision**
  - 对每个用户线性扫描 timestamp batch：先用严格更早历史构造该秒所有 purchase examples，再统一把该秒合法事件加入 history。
- **Result**
  - 12,207 个 final train examples 全部满足严格 `<`；对应 leakage test 通过。
- **Trade-off**
  - 同秒中真实存在但不可观测的先后信号被保守丢弃，换取清晰可审计的因果边界。

## ISSUE-029 — ID-only user tower does not model recent behavior explicitly

- **Problem**
  - 历史 Two-Tower 的 user representation 仅由 user ID 决定，无法显式表达用户在当前 cutoff 前的近期行为。
- **Evidence / How we found it**
  - 历史 ID-only Two-Tower 在 protected test 上明显弱于 ItemCF；V3 内部 EVAL 的现代 objective ID baseline Recall@50 仍仅为 `0.002791`。
- **Root cause / hypothesis**
  - user-ID embedding 只能在训练过程中累积静态身份信号，不能根据 query cutoff 前的 PV/FAV/CART/BUY 序列动态更新意图。
- **Fix / Decision**
  - 建立 behavior-aware last-50 history tower：共享 item embedding + behavior embedding，masked mean pooling + MLP；与 ID baseline 保持训练样本、objective、batch order 和评估完全一致。
- **Result**
  - 内部 EVAL 上 V3B Recall@50=`0.016547`，相对 V3A 提升 `+0.013756`；HitRate@50 提升 `+0.015550`，NDCG@50 提升 `+0.004959`。
- **Trade-off**
  - 要在请求时构建并编码最近历史；简单 mean pooling 丢失顺序和事件间隔信息，且不解决 cold-item 问题。

## ISSUE-030 — V3 Sequence-Aware internal temporal experiment

- **Problem**
  - 需要在不继续使用原 validation/test 的前提下，隔离现代 retrieval objective 与 user representation 的影响。
- **Evidence / How we found it**
  - 固定 internal fold：V3_TRAIN=`2017-11-25..29`，SELECT=`2017-11-30`，EVAL=`2017-12-01`；两个模型共用 10,254 个 strict-history examples 和 248,244-item train-only catalog。
  - EVAL：V3A R@50=`0.002791`，V3B=`0.016547`，Internal ItemCF=`0.284179`。
  - V3A/V3B dominant Top-1 均为 `0.002392`；Top-50 coverage 分别为 `0.142525/0.141582`。
- **Root cause / hypothesis**
  - 显式历史能修复部分 ID-only 信息瓶颈，但短窗口局部 item-item 共现仍是更强 inductive bias；mean pooling 也不保证更健康的 catalog exposure。
- **Fix / Decision**
  - 在单次 pre-eval freeze 后执行内部 EVAL，按预注册规则将结果归类为 MIXED；不调参、不重切日期、不运行原 protected test。
- **Result**
  - Explicit history 提高了相关性指标，但 concentration 没有同时改善，且 V3B 与 ItemCF 的 R@50 gap 仍为 `-0.267632`。
- **Trade-off**
  - 该结果只支持内部 temporal protocol 下的 V3A/V3B 因果方向对比，不是新 protected-test 结果，也不能与历史 V1 直接归因。
