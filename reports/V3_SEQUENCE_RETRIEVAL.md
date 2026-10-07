# V3 Sequence-Aware Two-Tower Retrieval

## Scope and protocol

V3 is a **post-freeze controlled retrieval extension** evaluated entirely inside the historical `data/processed/dev/train.csv` period. The original validation and protected test labels were not read for V3 training, checkpoint selection, or evaluation. Historical V0/V1/V2, ranking, cold/long-tail, and generative conclusions remain unchanged.

The fixed Asia/Shanghai calendar split was determined before observing retrieval metrics:

- V3_TRAIN: 2017-11-25 through 2017-11-29
- V3_SELECT: 2017-11-30
- V3_EVAL: 2017-12-01

Mappings and the 248,244-item candidate catalog were constructed only from V3_TRAIN, in stable first-occurrence order across all behavior types. The common warm population requires a V3_TRAIN-known user, at least one mapped pre-cutoff history event, and a V3_TRAIN-known purchase target. Metrics are therefore warm closed-catalog metrics, not all-traffic metrics.

| Population | Purchase interactions | Eligible interactions | Eligible users | Cold-target interactions | Eligible coverage |
|---|---:|---:|---:|---:|---:|
| SELECT | 2,087 | 1,247 | 963 | 819 | 0.597508 |
| EVAL | 1,994 | 1,092 | 836 | 882 | 0.547643 |

## Controlled comparison

Both models used the same 10,254 purchase examples, batch order, seed 42, AdamW optimizer, learning rate `1e-3`, weight decay `1e-6`, batch size 512, 10 epochs, temperature 0.1, 64-dimensional output, in-batch softmax, duplicate-target/same-user false-negative masking, and train-purchase-frequency logQ correction. Only the user representation changed.

- **V3A ID-only:** user ID embedding followed by an MLP.
- **V3B sequence-aware:** the last 50 strictly earlier mapped events, shared item plus behavior embeddings, padding mask, masked mean pooling, then an MLP. The history item table is shared with the candidate item tower.

Every training history satisfies `history_timestamp < target_timestamp`. Events sharing one timestamp are processed as a batch: examples are created first, and those events are appended to history only afterward. V3_SELECT chose checkpoints using exact full-catalog Recall@50 with NDCG@50 as tie-break. V3A selected epoch 9; V3B selected epoch 10.

## Exact internal results

No historical-item filtering was applied. Retrieval used chunked exact inner product over the full V3_TRAIN catalog; a small-case test verified equality with dense exact Top-K.

### SELECT

| Model | R@20 | R@50 | HR@50 | NDCG@50 | Dominant Top-1 | Top-50 coverage |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.004673 | 0.008654 | 0.010384 | 0.002656 | 1.000000 | 0.000201 |
| Internal ItemCF | 0.196642 | 0.367359 | 0.404984 | 0.116874 | 0.002092 | 0.166493 |
| V3A ID + InBatch + logQ | 0.001168 | 0.002336 | 0.003115 | 0.001552 | 0.003115 | 0.159597 |
| V3B Sequence + InBatch + logQ | 0.012721 | 0.013889 | 0.016615 | 0.006499 | 0.002077 | 0.161192 |

### EVAL

| Model | R@20 | R@50 | HR@50 | NDCG@50 | Dominant Top-1 | Top-50 coverage |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.002990 | 0.006579 | 0.008373 | 0.001784 | 1.000000 | 0.000201 |
| Internal ItemCF | 0.166071 | 0.284179 | 0.319378 | 0.090782 | 0.002392 | 0.147927 |
| V3A ID + InBatch + logQ | 0.001396 | 0.002791 | 0.003589 | 0.001418 | 0.002392 | 0.142525 |
| V3B Sequence + InBatch + logQ | 0.008971 | 0.016547 | 0.019139 | 0.006376 | 0.002392 | 0.141582 |

On EVAL, sequence versus ID deltas were Recall@50 `+0.013756`, HitRate@50 `+0.015550`, NDCG@50 `+0.004959`, dominant Top-1 share `+0.000000`, and Top-50 catalog coverage `-0.000943`. V3B therefore improved relevance metrics under the controlled comparison, but did not improve both preregistered concentration diagnostics. Its Recall@50 remained `0.267632` below internal ItemCF.

## Interpretation and freeze

The causal comparison is V3A versus V3B, because their data, objective, batch order, optimizer, logQ correction, training budget, and evaluation protocol are identical. It supports the conclusion that explicit recent behavior helps neural retrieval under this internal protocol. It does **not** justify attributing a comparison against historical V1 solely to sequence modeling, and it is not a new protected-test result.

The outcome is **mixed**: sequence representation repaired part of the ID-only weakness, but concentration did not improve and neural retrieval remained far behind local item-item collaborative retrieval. After the single pre-frozen V3_EVAL run, the experiment is final. No V4, post-eval tuning, date reselection, or protected-test evaluation is permitted.

**V3 RESULT MIXED — PROJECT FINAL FREEZE**
