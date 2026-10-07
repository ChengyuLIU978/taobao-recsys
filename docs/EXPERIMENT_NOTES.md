# Experiment Notes and Failure Analysis

This document summarizes the experiments that most influenced the final system design. It focuses on observed behavior, corrective engineering work, and conclusions that generalized across stages rather than reproducing the complete development log.

## 1. Recovering the Two-Tower ID Mapping

The first Two-Tower checkpoint contained embedding-table weights and model dimensions, but not the raw `user_id`/`item_id` to embedding-row mappings. The tensors could be loaded without error, yet retrieval results could not be translated back to real IDs safely. Rebuilding a vocabulary with a different order would have produced a silent semantic error: matrix operations would still work while every row referred to the wrong entity.

The mapping was recovered by replaying the exact notebook logic over training PV events: select `user_id` and `item_id`, keep the first occurrence, and preserve pandas `unique()` order. The recovered vocabulary contained 9,836 users and 302,016 items. Checkpoint dimensions, strict state-dict loading, and the training-file SHA-256 were used as independent checks. The resulting mappings are stored under `artifacts/two_tower/`; later experiments save mapping references and hashes alongside their configurations.

## 2. PV Objective vs. Purchase Retrieval

The initial model optimized page-view pairs with uniform sampled negatives, while offline evaluation measured future purchases. Its warm purchase Recall@50 was `0.003846`, compared with `0.371696` for ItemCF on the corresponding validation protocol. A decreasing sampled BCE loss therefore did not imply useful full-catalog purchase retrieval.

Controlled follow-ups aligned positives with the evaluation target. Buy-only V1 reached Recall@50=`0.011966`, while the first weighted multi-behavior variant reached `0.007058`. Both improved on PV-only training, but neither approached ItemCF. The main lesson was methodological: the positive definition and checkpoint-selection metric need to match the downstream retrieval task, and sampled loss should remain a diagnostic rather than the final model-selection criterion.

## 3. Recommendation Collapse in the Initial Two-Tower

The V0 retriever produced highly concentrated results. Across 1,170 warm users, only two items appeared at rank one; item `3,845,720` was the Top-1 recommendation for `91.97%` of users. The union of all Top-50 lists contained only 301 of 302,016 catalog items. This made the weakness visible beyond Recall alone.

Purchase-aligned training reduced the collapse substantially. Buy-only V1 lowered the dominant Top-1 share to `5.38%` and expanded the Top-50 union to 3,934 items. The weighted multi-behavior variant still placed the same dominant item first for `81.20%` of users and exposed only 688 Top-50 items. Adding behavior types was therefore not sufficient; their effective sampling mass and the retrieval objective mattered more than the number of signals.

## 4. Why Buy-Only Helped but Remained Unstable

Buy-only supervision was sparse: 11,844 mapped pairs covered 5,141 users and 10,819 items. Matching the original step budget repeated each pair roughly 44 times per epoch, which improved target alignment but also created overfitting and initialization sensitivity. Seed 42 reached Recall@50=`0.011966`.

One fixed alternate-seed replication produced Recall@50=`0.008405`. It remained 2.185 times above the PV baseline but was `29.76%` below the seed-42 run. This supports a directional benefit from purchase-aligned positives, not strict numerical stability. Both runs are retained because the variance is part of the result rather than noise to hide with repeated reruns.

## 5. Why ItemCF Outperformed the Neural Retriever

ItemCF consistently dominated the ID-only neural models. On the reported test-warm population, ItemCF Recall@50 was `0.241398`, while the historical Two-Tower result was `0.005376`. In the later internal V3 fold, ItemCF reached `0.284179`, compared with `0.016547` for the sequence-aware neural retriever.

The short observation window gives local item-item co-occurrence a strong inductive bias: recent neighboring items can transfer evidence directly without learning a dense representation for the entire catalog. The neural models had sparse purchase targets, incomplete item-specific supervision, and no content features. This is a useful counterexample to the assumption that a more flexible model automatically beats a simple collaborative baseline.

## 6. Why Naive RRF Failed

The three recall sources had little candidate overlap, but that did not mean their new candidates were relevant. On validation, ItemCF Recall@50 was `0.371696`; equal-formula Reciprocal Rank Fusion reduced it to `0.179821`. RRF treated a high rank from a weak source as evidence comparable to a high rank from ItemCF, even though source quality and score calibration differed substantially.

An ItemCF-first ordering preserved ItemCF's Top-50 results but added no early-cutoff gain. The final pipeline instead retained source flags, ranks, scores, and overlap counts as features for LambdaRank. The broader lesson is that candidate diversity should be evaluated through incremental target hits and oracle recall, not pairwise overlap alone.

## 7. Candidate Oracle and the Retrieval Bottleneck

The final LambdaRank model achieved test-warm Recall@10/20/50 of `0.166771/0.202315/0.238792` and NDCG@50=`0.124189`. The same fixed candidate set had oracle Recall@50=`0.255548`, leaving only `0.016756` absolute Recall headroom for reordering. The ranker greatly improved early ordering but could not recover targets absent from the candidate union.

Segment-level oracle gaps were also small: `0.018478` for Head, `0.014837` for Torso, and `0.013652` for Tail. These measurements changed the development priority from a more complex ranker to better candidate generation. A new recall source would need to demonstrate incremental oracle hits before increasing downstream ranking cost.

## 8. Cold Items vs. Long-Tail Items

Cold and long-tail items are different failure modes. Of 2,512 test purchase interactions, 962 targets (`38.30%`) were absent from the training catalog. No closed-catalog collaborative model can retrieve them, giving an interaction-level ceiling of `0.617038`. Addressing this group requires item metadata or content available at serving time.

Tail items were present in training but had little evidence. For train-known targets, the ranker's Head/Torso/Tail Recall@50 was `0.271739/0.246291/0.184300`; the corresponding candidate-oracle values were `0.290217/0.261128/0.197952`. The historical Two-Tower had zero Tail hits and zero Tail-exclusive hits. Combining cold and Tail into one bucket would obscure both the structural catalog limit and the separate sparse-supervision problem.

## 9. Semantic-ID Collisions and the Deterministic Suffix

The three-level RQ-VAE produced healthy quantization statistics: all three 128-code codebooks had 100% utilization, and validation reconstruction loss reached `0.00188850`. However, the raw `(c0,c1,c2)` tuples were not unique. For 302,016 active items, only 248,774 tuples were unique, yielding 53,242 excess collision items and a collision-excess rate of `17.6289%`.

Items within each collision group were sorted by `item_id`, then assigned a deterministic suffix. The final `(c0,c1,c2,suffix)` identifier had zero collisions and supported complete item→SID→item round trips. The suffix solves identity and reversibility; it does not add semantic capacity to the first three quantized codes.

## 10. Generative Retrieval: Valid Generation but No Test Increment

Trie-constrained beam search generated valid catalog paths for 100% of validation and test outputs, with no invalid paths or duplicate items. The generative retriever achieved validation Recall@20/50=`0.063319/0.065456` and added two Head targets beyond the existing union, moving validation oracle recall from `0.384716` to `0.386426`.

That small signal did not transfer to the held-out test period. Test Recall@20/50 was `0.042458/0.042458`, and incremental hits relative to both ItemCF and the current union were zero; Tail incremental hits were also zero. The experiment demonstrates that tokenizer reconstruction, codebook utilization, and valid decoding are engineering properties, while complementary recommendation value must be measured separately.

## 11. Sequence-Aware Two-Tower Follow-up

V3 used an internal temporal split within the historical training period: 2017-11-25 through 2017-11-29 for training, 2017-11-30 for checkpoint selection, and 2017-12-01 for evaluation. V3A used an ID-only user tower; V3B pooled the last 50 behavior-aware events. Both used the same 10,254 strict-history purchase examples, in-batch softmax, logQ correction, batch order, optimizer, and 248,244-item catalog.

On the internal evaluation fold, V3A Recall@50 was `0.002791` and V3B reached `0.016547`, with HitRate@50 increasing from `0.003589` to `0.019139`. Dominant Top-1 share was unchanged and Top-50 coverage moved from `0.142525` to `0.141582`; ItemCF remained far stronger at Recall@50=`0.284179`. The result is mixed: explicit history improved relevance, but did not improve concentration or close the gap to local collaborative retrieval. Because V3B also sends history-side gradients through the shared item table, the comparison is controlled but not a perfectly isolated single-parameter intervention.

## 12. Full-Catalog Representation Coverage in V3

A read-only coverage audit found 9,318 unique purchase-target items (`3.753565%` of the catalog) and 101,095 unique retained-history items (`40.724046%`). Their intersection contained 7,991 items. The union covered 102,422 catalog items (`41.258600%`), leaving 145,822 (`58.741400%`) without item-specific data gradients in V3B.

In-batch negatives are other purchase targets in the same batch, not arbitrary catalog samples. V3A therefore left `96.246435%` of base item embeddings without target-side evidence. Sharing the item table with the V3B history encoder broadened exposure considerably, but did not cover the full catalog. Mixed full-catalog negatives, an auxiliary behavior/item objective, or pretrained/content representations are reasonable follow-up designs for a separate evaluation protocol.
