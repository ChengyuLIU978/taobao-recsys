# Experiment Freeze

**Freeze status:** `PROJECT FROZEN — interview/recruiting-ready baseline`  
**Effective date:** 2026-10-08

## Scope

The current test period has been used for the final one-time diagnosis of:

1. the traditional recall → candidate union → LambdaRank pipeline;
2. cold-start and long-tail behavior;
3. the collaborative Semantic-ID generative retrieval extension.

These results are final historical evidence. Existing checkpoints, configs, histories, metrics, recommendations and summaries must not be overwritten or silently recomputed.

## Prohibited use of the current test period

Do not use the current test results to select or tune:

- Two-Tower objectives, seeds, negatives or architecture;
- recall quotas, RRF constants or fusion rules;
- ranker features, model parameters or candidate policy;
- RQ-VAE codebooks, suffix policy or reconstruction settings;
- Transformer configuration, beam size or Tail weighting;
- any new recall source intended to enter the formal union.

Do not delete `artifacts/generative/final_summary.json` or `pre_test_freeze.json` to force another generative test evaluation.

## Required protocol for future modeling

Any future optimization must first establish a **new temporal validation fold** or a rolling temporal evaluation design. Model selection must occur on that new validation evidence. The currently frozen test period may be referenced only as historical context, not as a feedback signal.

Priority order:

1. new temporal validation fold / rolling evaluation;
2. complementary warm retrieval;
3. metadata/content cold-item encoder;
4. Tail-aware generative training;
5. negative sampling and debiasing;
6. FAISS/ANN engineering benchmark.

## Safe activities after freeze

Documentation, artifact inventory, hash verification, read-only inspection and regression tests are allowed. Training, parameter search, candidate-policy changes and repeated test evaluation are not.
