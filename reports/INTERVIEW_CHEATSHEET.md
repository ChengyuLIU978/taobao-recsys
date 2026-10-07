# Interview Cheatsheet

## 项目一句话

我在 Taobao 1.0015 亿条行为上构建了一个 leakage-safe 的多阶段推荐系统，从时序数据、三路召回、候选融合到 LambdaRank，并完整验证了 Cold/Long-tail 和 Semantic-ID 生成式召回的成功、失败与系统边界。

## 2 分钟版本

数据是 Taobao UserBehavior，原始 100,150,807 条。我用 `user_id % 100 == 0` 做确定性 user-level 1% 开发样本，保留完整用户历史，再按时间划分 train/validation/test，避免 future leakage。

召回层实现 Popularity、ItemCF 和 ID-only Two-Tower。ItemCF 是最强 source；Two-Tower 的关键实验发现是 PV objective 与 purchase evaluation 不一致，Buy-only V1 的 Recall@50 从 V0 的 `0.003846` 提到 `0.011966`，alternate seed 为 `0.008405`，方向复现但并不严格稳定。候选用 ItemCF200 + TT100 + Popularity50，保留 source provenance。Naive RRF 因 source strength/calibration 不一致严重下降。

排序层用 52 个 train-only features 训练 LightGBM LambdaRank。Test-warm Recall@10=`0.166771`、NDCG@50=`0.124189`；ItemCF R@50=`0.241398` 仍略高于 Ranker `0.238792`。Candidate oracle=`0.255548`，说明瓶颈已主要在 retrieval。

生成式扩展把 302,016 个 V1 item embeddings 用 RQ-VAE 转成 Semantic IDs，再用 behavior-aware Transformer 和 trie-constrained beam search 生成候选。量化和合法性都成功，但 validation 的 2 个增量 target 没有在 test 复现，所以不加入正式 union。这个项目最重要的能力不是堆模型，而是用统一 protocol 判断什么真正有效。

## 5 分钟版本

### 1. 数据与协议

原始数据约 3.67 GB、1.0015 亿行。确定性 user sample 有 1,001,832 行；train/validation/test 是 724,885/138,406/138,541。全部 split 按时间，ranking aggregate 只来自 train。Test labels 在 test scoring 后才读取。

### 2. Retrieval

Popularity 做 fallback；ItemCF 捕捉短窗口共现；Two-Tower 做 ID embedding retrieval。Exact Top-K 用 chunked inner product。V0 的 PV-only objective 与购买目标错位；受控实验显示 Buy-only 更合理，但 ItemCF 仍远强于 Two-Tower。

### 3. Fusion 与 Ranking

每个用户最多 350 candidates，所有来源、rank 和 score 都保留。RRF validation R@50=`0.179821`，远低于 ItemCF `0.371696`，说明异构 source rank 不可直接等权。LambdaRank 用 52 个 train-only features，显著提升前排质量。Oracle gap 只有 `0.016756`，说明继续堆 ranker 回报有限。

### 4. Cold/Long-tail

Test 2,512 个购买中有 962 个 train-cold，占 38.30%，closed-catalog ceiling=`0.617038`。Tail Ranker R@50=`0.184300`，oracle=`0.197952`，主要问题在召回。Two-Tower Tail R@50=0、Tail exclusive=0。

### 5. GenRec

V1 `(302016,64)` embeddings 经 3×128 RQ-VAE，reconstruction=`0.00188850`，utilization 全 100%。Raw tuple 有 17.6289% collision excess，用 deterministic suffix 实现 item↔SID 1:1。Transformer 用 12,207 个 strict causal purchase examples，trie beam 保证 100% valid output。Validation union gain `+0.001709`，test gain `0`，所以保留 baseline、不进入主系统。

### 6. 结论

我能解释每层的输入、输出、指标、泄漏边界和 trade-off；也能说明为什么负结果不是失败的项目，而是把优化方向从 ranking、naive fusion 和表面健康指标转向真正的 complementary retrieval 与 cold content signal。

## 架构流程

```text
100M events
  → deterministic user sample
  → temporal split
  → Popularity / ItemCF / Two-Tower
  → source-aware union (≤350)
  → 52 train-only features
  → LambdaRank
  → protected test evaluation
  → cold / head-torso-tail / exposure diagnostics

V1 item embeddings
  → RQ-VAE
  → (c0,c1,c2,suffix)
  → behavior-aware causal history
  → Transformer
  → SID trie + beam search
  → incremental oracle analysis
```

## 必背指标

| 项目 | 数值 |
|---|---:|
| Raw interactions | 100,150,807 |
| Dev interactions | 1,001,832 |
| Ranker Recall@10 / @20 / @50 | 0.166771 / 0.202315 / 0.238792 |
| Ranker NDCG@50 | 0.124189 |
| ItemCF Recall@50 | 0.241398 |
| Candidate oracle | 0.255548 |
| Oracle gap | 0.016756 |
| Train-cold targets | 962 / 2,512 = 38.30% |
| Closed-catalog ceiling | 0.617038 |
| V0 / V1 / seed2027 / V2 R@50 | 0.003846 / 0.011966 / 0.008405 / 0.007058 |
| Gen validation/test R@50 | 0.065456 / 0.042458 |
| Gen validation/test incremental hits | 2 / 0 |

## 最重要的失败实验

- PV-only Two-Tower：训练目标与 purchase evaluation 不一致。
- Multi-behavior weighting：没有超过 Buy-only，且集中度问题仍在。
- RRF：把弱源 rank 与强 ItemCF rank 近似等价，破坏强源。
- Two-Tower 长尾假设：Tail Recall 与 exclusive hits 都为 0。
- GenRec：量化健康、输出合法，但 test incremental recall=0。

## 高频追问与答案

### 数据与评估

**1. 为什么用 user-level sample，而不是随机抽行？**  
抽完整用户能保留序列、共现、recency 和购买前历史；随机抽行会把这些信号切碎。模运算还可确定性复现。

**2. 为什么只用约 1%？**  
在本地 CPU/内存预算下先验证完整工程与实验协议。结论是开发 baseline，不声称等同 full-scale production result。

**3. 为什么 temporal split？**  
推荐是预测未来。Random split 会把未来行为泄漏到 aggregate、similarity、sequence 和 target 之前的特征里。

**4. Validation 和 test 分别做什么？**  
Validation 用于 objective/model selection 和 ranker fit；test 只做最终一次评估，不能回头调参。

**5. Warm evaluation 会不会掩盖 cold 问题？**  
会，所以主表用 warm 做公平的 closed-catalog 比较，同时单独报告全 GT、train-cold fraction 和 ceiling。

**6. Recall、HitRate、NDCG 有什么区别？**  
Recall 衡量每个用户 target 覆盖，HitRate 只看是否至少命中，NDCG 还奖励更靠前的命中。

### Two-Tower

**7. 为什么选择双塔？**  
User/item 可离线编码，在线用向量检索扩展到大 catalog；同时能与 exact/ANN retrieval 对接。

**8. 为什么最初用 BCE？**  
正负 pair 的 pointwise logistic objective 实现简单、适合 uniform sampled negatives；但 sampled loss 不能替代 full-catalog metric。

**9. 为什么需要 negative sampling？**  
完整 user×item 空间太大，绝大多数是未观测 pair；采样使训练可行。

**10. 为什么 Buy-only 比 PV-only 好？**  
最终评价目标是购买。PV 数量大但意图弱，产生 objective mismatch；Buy-only 与目标更一致。

**11. seed2027 说明模型稳定吗？**  
只说明方向得到一次预注册 alternate-seed 支持。它比 seed42 低 29.76%，不能声称严格稳定。

**12. 为什么 ItemCF 比 Two-Tower 强？**  
短时间窗中的局部共现对未来购买很直接；ID-only 双塔缺少内容特征，正样本少，uniform negatives 又较容易。

**13. 为什么 sampled validation loss 与 recall 不一致？**  
Loss 只比较少量 sampled negatives；full-catalog Top-K 需要在 302K items 中竞争，优化对象不同。

**14. 为什么不用一次性完整 score matrix？**  
`9836×302016` 会占用大量内存；按 user batch/item chunk 合并局部 Top-K 可得到相同 exact 结果。

### Candidate 与 Fusion

**15. 为什么 ItemCF200、TT100、Popularity50？**  
这是 validation 阶段冻结的有限计算预算，给最强 ItemCF 最大 quota，同时保留互补源与 fallback；test 后不再调整。

**16. 为什么 candidate union 上限是 350？**  
三个 quota 之和为 350；去重只会减少，用固定上限控制 feature 与 ranking 成本。

**17. 为什么保存 source provenance？**  
Ranker 需要知道候选来自哪个 source、rank/score 和多源支持程度，才能学习条件化信任。

**18. 为什么 RRF 失败？**  
异构 source 的 rank 不同质，ItemCF 远强于其他 source；RRF 过度抬高弱源高位，稀释强源。

**19. Candidate overlap 低为什么不一定好？**  
集合不同不代表新增部分包含 GT。应看 exclusive target hits 与 union oracle delta。

**20. 为什么不直接只用 ItemCF？**  
Two-Tower 有少量 Head exclusive hits，Popularity 可 fallback；统一 candidate table 也为监督式融合提供实验基础。

### Ranking

**21. 为什么 LambdaRank？**  
目标是 query 内 item 排序，LambdaRank 直接优化排序相关 surrogate，适合稠密表格特征且 CPU 高效。

**22. 为什么不用 DIN？**  
当前瓶颈首先是 candidate coverage；DIN 需要更多序列工程与计算，也不能找回候选外 target。先建立强、可解释的 LTR baseline 更合理。

**23. 为什么在 validation 上训练 Ranker？**  
Train 用于构建无泄漏的历史 aggregate，validation candidates+labels 提供下一时间窗监督；test 保持最终 holdout。

**24. 为什么 raw user_id/item_id 不做 numeric feature？**  
数值大小没有序关系，会产生伪距离与过拟合；身份信号应通过统计、embedding 或 categorical treatment 表达。

**25. Ranker 是否全面超过 ItemCF？**  
不是。Ranker 显著提高 R@10、R@20、NDCG，但 R@50=`0.238792` 略低于 ItemCF `0.241398`。

**26. 为什么 test 不能继续调？**  
反复根据 test 选择模型会把 test 变成 validation，最终分数不再是无偏泛化估计。

**27. Candidate oracle 有什么用？**  
它把“是否召回”与“如何排序”分开。Oracle-ranker gap 小意味着 retrieval 是主要瓶颈。

### Cold Start 与 Long Tail

**28. 为什么 cold item 无法召回？**  
ItemCF、ID embedding 和 collaborative SID 都要求 item 在 train 有历史；train-unseen item 没有 similarity、embedding 或合法 SID path。

**29. Cold 和 Tail 有什么区别？**  
Tail 是 train-known 但低频；Cold 是 train 未出现。把 Cold 塞进 Tail 会夸大长尾问题并混淆可达上限。

**30. 为什么 Tail 表现差？**  
低频 item 的共现、购买监督与生成 target 都少。Segment oracle 本身低，说明先缺候选，而不是只缺排序。

**31. Two-Tower 是否帮助长尾？**  
当前没有：Tail R@50=0，Tail exclusive hits=0，Tail exposure 仅 0.36%。

### Generative Retrieval

**32. 什么是 RQ-VAE？**  
Residual-Quantized VAE/autoencoder 用多层 codebook 逐步量化残差，把连续 embedding 表示成离散 code sequence。

**33. 什么是 residual quantization？**  
第一层编码原向量，下一层编码前一层未解释的 residual；多层 code 共同近似原 embedding。

**34. 什么是 STE？**  
Straight-Through Estimator 在 forward 使用离散最近邻 code，在 backward 近似把梯度传回 encoder，使不可导量化可训练。

**35. Codebook utilization 100% 说明什么？**  
说明每个 code 都被用到、没有明显 collapse；不说明 code 对推荐一定有用。

**36. 为什么需要 suffix？**  
Raw 三层 tuple 对 item 不是 injective。Suffix 在 collision group 内提供确定性唯一标识，使 item↔SID 可逆。

**37. Suffix 会提升 semantic quality 吗？**  
不会。它解决 identity collision，区分能力主要集中在最后 token，不改善前三层 quantization geometry。

**38. 为什么需要 prefix trie？**  
无约束 autoregressive decoding 会产生 catalog 中不存在的 code path。Trie 在每一步只允许合法前缀的下一个 token。

**39. 100% valid generation 是否代表成功？**  
只代表工程合法性。Test catalog coverage 4.45%、Gini 0.988493，且 incremental hits=0，相关性与互补性仍失败。

**40. 为什么 generator 在 test 没增量？**  
它命中的 targets 基本已被强 ItemCF/union 覆盖；purchase supervision 少且偏 Head，collaborative codes 没创造新的冷启动信息。

**41. GenRec 是 exact TIGER 吗？**  
不是。它是 TIGER-style collaborative extension；没有 rich content-derived semantics，也不解决 cold items。

**42. 为什么同 timestamp 要 batch？**  
秒级 timestamp 无法确定同秒事件先后。逐行会伪造因果顺序；batch 后只使用严格更早时间的历史。

**43. 为什么不把 GenRec 加入 union 再让 Ranker决定？**  
进入 union 应先证明 incremental oracle value。Test 新增 target=0，只会增加候选与计算成本，没有上限收益。

**44. 下一步最应该做什么？**  
先建立新的 temporal validation fold，再研究 complementary warm retrieval；之后做 content-based cold item、Tail-aware training、debiasing 和 ANN benchmark。

### Mapping Recovery

**45. 为什么已有 Two-Tower checkpoint 不能直接用于真实 user/item retrieval？如何恢复 mapping？**  
Checkpoint 只保存 embedding table weights、tensor dimensions 和模型参数，没有保存 raw `user_id` / `item_id` 到 embedding index 的映射。问题是在 strict load 后准备把 embedding row 映射回真实 ID 时发现的：tensor shape 只能说明有多少行，不能说明第 `i` 行属于哪个原始 ID。随便对 ID 排序或重新生成 mapping 会造成 silent error，因为 shape 仍匹配、模型仍可运行，但每一行 embedding 会被解释成错误的 user/item。恢复时严格重放原训练规则：只使用 train 中的 `pv` 行，选择 `user_id, item_id`，按首次出现保留地去重，并使用 pandas `unique()` 的 stable order 建立 index。恢复结果为 users=`9,836`、items=`302,016`；再用 checkpoint dimensions、strict checkpoint load，以及 train file SHA-256 `c8a702804ada2e1da7e7e448db71aabc86afa0e1336ac7147bcd40e8c5ad9148` 共同验证。后续 checkpoint 应同时保存 immutable mapping，或保存不可变 mapping reference、hash 和构建规则。

**46. 如果 mapping 顺序错了，为什么模型可能仍然能运行但结果完全错误？**  
Embedding table 的行数和维度没有变化，所以 `state_dict` 可以正常加载，matrix multiplication 和 Top-K 也不会报错；错误只发生在“row index 代表哪个真实 ID”的语义层。用户向量会查询到错误用户、item score 会被翻译成错误 item ID，因此这是很难通过 shape check 发现的 silent semantic corruption。

**47. 生产系统里应该如何避免 mapping recovery 问题？**  
把 vocabulary/mapping 视为模型 artifact 的一部分：训练时原子化保存 immutable mapping、schema version、source-data hash、mapping hash、unknown-ID policy 和 model config；部署时先校验 hash 与 cardinality，再 strict-load checkpoint。模型 registry 应把 checkpoint 与 mapping 作为同一版本单元，禁止单独替换其中一个。

## 结束语

如果面试官只记住一个结论：这个项目不是“某个模型拿到一个分数”，而是一套从 leakage control、retrieval、ranking、oracle diagnosis 到负结果管理的完整推荐系统实验方法。
