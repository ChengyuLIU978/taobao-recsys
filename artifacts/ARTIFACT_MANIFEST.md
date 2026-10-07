# Artifact Manifest

All reported experiment outputs are frozen as of 2026-10-08. Sizes below are the local file sizes observed during final packaging. Binary checkpoints, arrays, Parquet datasets and pickles are intentionally excluded by `.gitignore`; their small JSON/CSV configs, metrics and manifests remain suitable for Git.

| Artifact | Stage | Purpose | Local size | Reproducible from | Frozen |
|---|---|---|---:|---|:---:|
| `data/raw/UserBehavior.csv` | Source data | Raw Taobao UserBehavior events | 3,502.223 MiB | External dataset; not redistributed | Yes |
| `data/interim/UserBehavior_dev.csv` | Data pipeline | Deterministic `user_id % 100 == 0` sample | 35.989 MiB | `notebooks/02_build_dev_dataset.ipynb` | Yes |
| `data/processed/dev/train.csv` | Data pipeline | Chronological train interactions | 26.039 MiB | `notebooks/02_build_dev_dataset.ipynb` | Yes |
| `artifacts/two_tower/user_mapping.csv` | Two-Tower | User ID → embedding index | 0.120 MiB | `notebooks/04_build_two_tower.ipynb` | Yes |
| `artifacts/two_tower/item_mapping.csv` | Two-Tower | Item ID → embedding index | 4.441 MiB | `notebooks/04_build_two_tower.ipynb` | Yes |
| `artifacts/two_tower/two_tower_best.pt` | V0 PV | Historical PV-only checkpoint | 76.267 MiB | `notebooks/04_build_two_tower.ipynb` | Yes |
| `artifacts/two_tower/experiments/TT_V1_BUY_UNIFORM/checkpoint_best.pt` | V1 Buy | Seed-42 purchase-positive checkpoint | 76.267 MiB | `notebooks/05_two_tower_controlled_experiments.ipynb` | Yes |
| `artifacts/two_tower/experiments/TT_V2_MULTI_UNIFORM/checkpoint_best.pt` | V2 Multi | Multi-behavior checkpoint | 76.267 MiB | `notebooks/05_two_tower_controlled_experiments.ipynb` | Yes |
| `artifacts/two_tower/experiments/TT_V1_BUY_UNIFORM_SEED2027/best.pt` | Seed replication | Buy-only alternate-seed checkpoint | 76.267 MiB | `scripts/06_repeat_buy_seed.py` | Yes |
| `artifacts/multistage_recall/validation_candidates.parquet` | Candidate union | Validation union with source provenance | 25.519 MiB | `scripts/07_build_multistage_recall.py` | Yes |
| `artifacts/multistage_recall/test_candidates.parquet` | Candidate union | Label-free generated test candidate union | 25.436 MiB | `scripts/07_build_multistage_recall.py` | Yes |
| `artifacts/ranking/ranker.pkl` | Ranking | Full 52-feature LightGBM LambdaRank model | 1.162 MiB | `scripts/08_train_ranker.py` | Yes |
| `artifacts/ranking/validation_ranking_dataset.parquet` | Ranking | Validation feature/label training table | 85.625 MiB | `scripts/08_train_ranker.py` | Yes |
| `artifacts/ranking/test_ranking_dataset.parquet` | Ranking | Frozen test feature/score table plus offline labels | 88.657 MiB | `scripts/08_train_ranker.py` | Yes |
| `artifacts/ranking/test_recommendations.parquet` | Ranking | Final ranked test Top-50 | 4.937 MiB | `scripts/08_train_ranker.py` | Yes |
| `artifacts/analysis/analysis_summary.json` | Cold/long-tail | Machine-readable diagnostic summary | 0.024 MiB | `scripts/09_analyze_cold_start_long_tail.py` | Yes |
| `artifacts/generative/v1_item_embeddings.npy` | GenRec input | 302,016×64 normalized V1 item embeddings | 73.734 MiB | `scripts/10_export_v1_item_embeddings.py` | Yes |
| `artifacts/generative/rqvae_best.pt` | RQ-VAE | Best 3×128-codebook quantizer | 0.225 MiB | `scripts/11_train_rqvae.py` | Yes |
| `artifacts/generative/semantic_ids.parquet` | Semantic IDs | Item → `(c0,c1,c2,suffix)` catalog | 2.809 MiB | `scripts/12_build_semantic_ids.py` | Yes |
| `artifacts/generative/semantic_id_lookup.npz` | Semantic IDs | Compact bidirectional SID lookup | 2.096 MiB | `scripts/12_build_semantic_ids.py` | Yes |
| `artifacts/generative/generator_best.pt` | GenRec | Validation-selected Transformer checkpoint | 2.965 MiB | `scripts/13_train_generative_retriever.py` | Yes |
| `artifacts/generative/validation_generated_recommendations.parquet` | GenRec validation | Constrained validation recommendations | 0.861 MiB | `scripts/13_train_generative_retriever.py` | Yes |
| `artifacts/generative/test_generated_recommendations.parquet` | GenRec test | One-time frozen constrained test recommendations | 6.233 MiB | `scripts/14_evaluate_generative_recall.py` | Yes |
| `artifacts/generative/final_summary.json` | GenRec summary | Full final metrics, paths and test-once status | 0.024 MiB | `scripts/14_evaluate_generative_recall.py` | Yes |
| `artifacts/v3_sequence_retrieval/final_summary.json` | V3 internal retrieval | Frozen internal temporal comparison and final interpretation | 0.006 MiB | `scripts/15_train_sequence_two_tower.py` | Yes |
| `artifacts/v3_sequence_retrieval/V3A/best.pt` | V3A internal retrieval | Select-chosen ID-only in-batch/logQ checkpoint | 63.129 MiB | `scripts/15_train_sequence_two_tower.py` | Yes |
| `artifacts/v3_sequence_retrieval/V3B/best.pt` | V3B internal retrieval | Select-chosen sequence-aware in-batch/logQ checkpoint | 60.740 MiB | `scripts/15_train_sequence_two_tower.py` | Yes |
| `artifacts/final/project_summary.json` | Final packaging | Cross-stage machine-readable project summary | generated below | Documentation packaging from frozen metrics | Yes |

## Supporting artifact groups

- `artifacts/two_tower/experiments/*/{config,history,metrics,concentration}`: controlled-experiment configuration and evidence.
- `artifacts/multistage_recall/*.json`: candidate statistics, overlap, contribution, protocol and execution checks.
- `artifacts/ranking/{config,feature_manifest,summary,test_metrics,feature_importance,ablation_metrics}`: ranker contract and frozen evaluation.
- `artifacts/analysis/*`: cold quadrants, closed-catalog ceiling, segment metrics, exposure, novelty and bias amplification.
- `artifacts/generative/{embedding_manifest,rqvae_metrics,semantic_id_stats,generator_config,generator_history,validation_generative_metrics,test_generative_metrics,pre_test_freeze}`: generative audit trail.
- `artifacts/v3_sequence_retrieval/{temporal_split,mapping_manifest,train_example_stats,comparison_select,comparison_eval,pre_eval_freeze,final_summary}` and `V3A/V3B/{config,history,select_metrics,eval_metrics,concentration}`: post-freeze internal retrieval audit trail. V3 checkpoints remain local because `*.pt` is ignored.

## Reproduction warning

The listed scripts document how stages were produced, but the workspace is frozen. Do not rerun them in place or overwrite these outputs. Future experiments require a separate artifact root and a new temporal validation protocol.
