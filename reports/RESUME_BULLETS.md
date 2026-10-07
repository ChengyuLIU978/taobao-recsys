# Resume Bullets

## 中文

- 基于 Taobao UserBehavior 1.001 亿条行为构建端到端多阶段推荐系统，完成时序数据治理、Popularity/ItemCF/Two-Tower 召回、候选融合与 52 特征 LightGBM LambdaRank，并以一次性 test protocol 审计 future leakage。
- 将 test-warm Recall@10 从 ItemCF 的 `0.0803` 提升至 Ranker 的 `0.1668`，NDCG@50 从 `0.0754` 提升至 `0.1242`；结合 `0.2555` candidate oracle 定位 retrieval coverage 为下一阶段主要瓶颈。
- 在 30.2 万 item catalog 上实现 RQ-VAE Semantic-ID、behavior-aware Transformer 与 trie-constrained beam search；实现 100% 合法生成并解决 17.63% raw SID collision excess，同时如实验证其 test incremental recall 为 0、未纳入正式召回。

## English

- Built an end-to-end multi-stage recommender over 100.15M Taobao behavior events, including leakage-safe temporal data processing, Popularity/ItemCF/Two-Tower retrieval, candidate union, and a 52-feature LightGBM LambdaRank pipeline.
- Improved test-warm Recall@10 from `0.0803` (ItemCF) to `0.1668` and NDCG@50 from `0.0754` to `0.1242`; used a `0.2555` candidate oracle to identify retrieval coverage—not ranker capacity—as the main remaining bottleneck.
- Implemented RQ-VAE Semantic IDs and trie-constrained Transformer retrieval over a 302K-item catalog, achieving 100% valid generation and resolving 17.63% raw SID collision excess; retained the model as an experimental baseline after it produced zero incremental test hits.
