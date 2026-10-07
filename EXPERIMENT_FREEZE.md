# Experiment Freeze

**Freeze status:** `PROJECT FINAL FREEZE — V3 internal extension completed`

**Effective date:** 2026-10-08

## Scope

The current test period has been used for the final one-time diagnosis of:

1. the traditional recall → candidate union → LambdaRank pipeline;
2. cold-start and long-tail behavior;
3. the collaborative Semantic-ID generative retrieval extension.

These results are final historical evidence. Existing checkpoints, configs, histories, metrics, recommendations and summaries must not be overwritten or silently recomputed.

After the original system freeze, one final controlled extension was run entirely inside historical `train.csv`: V3 Sequence-Aware Two-Tower. It used a fixed internal train/select/eval temporal fold and did not access original validation or protected-test labels. Its single internal evaluation is complete with result `MIXED`; `artifacts/v3_sequence_retrieval/pre_eval_freeze.json` and `final_summary.json` are now frozen. No V4, alternate split, additional seed, post-eval tuning, or protected-test V3 evaluation is allowed.

## Prohibited use of the current test period

Do not use the current test results to select or tune:

- Two-Tower objectives, seeds, negatives or architecture;
- recall quotas, RRF constants or fusion rules;
- ranker features, model parameters or candidate policy;
- RQ-VAE codebooks, suffix policy or reconstruction settings;
- Transformer configuration, beam size or Tail weighting;
- any new recall source intended to enter the formal union.

Do not delete `artifacts/generative/final_summary.json` or `pre_test_freeze.json` to force another generative test evaluation.

## Historical research directions, not authorized follow-up work

The project is now in final freeze and has no authorized follow-up model development. If these ideas are ever explored in a separate future project, optimization must first establish a **new temporal validation fold** or rolling temporal evaluation design. The currently frozen test period and V3_EVAL may be referenced only as historical context, not as feedback signals.

Priority order:

1. new temporal validation fold / rolling evaluation;
2. complementary warm retrieval;
3. metadata/content cold-item encoder;
4. Tail-aware generative training;
5. negative sampling and debiasing;
6. FAISS/ANN engineering benchmark.

## Safe activities after freeze

Documentation, artifact inventory, hash verification, read-only inspection and regression tests are allowed. Training, parameter search, candidate-policy changes, repeated protected-test evaluation, and repeated V3_EVAL are not.
