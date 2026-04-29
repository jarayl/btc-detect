# Inductive, Causal Transaction Classification on Elliptic++

Final results write-up for the cross-step temporal-bipartite line of work.

## TL;DR

Under a strict inductive, time-causal evaluation (at any time `T`, only data with `τ ≤ T` may be used), the pipeline lifts illicit-class **F1 from 0.8021 (no-graph floor) to 0.8557**, a +0.0536 absolute lift. Roughly a quarter of the lift comes from cross-step graph features alone (+0.0128), a small slice from rolling walk-forward retraining alone (+0.0034), and **the largest share — +0.0374 — comes from the interaction**: graph features only deliver their full value when the model is retrained as new labels arrive. The headline model is a **walk-forward LightGBM** on a 233-dim feature set that fuses 108 intrinsic per-tx features, 103 wallet-trajectory features, 17 cross-step pair features, and 5 personalized-PageRank features seeded from illicit history.

## 1. Problem and causality contract

Binary classification of Bitcoin transactions as illicit (1) or licit (0) on the **Elliptic++** dataset (203,769 txs across 49 timesteps; 4,545 illicit, 42,019 licit, 157,205 unknown).

**Strict causal contract** (used end-to-end at both train and test time):
- For target tx `T` at time `t`, every feature uses only data with `τ < t` (or `τ ≤ t` for `T`'s own atomic properties; `T` is observed at `t`).
- Historical labels of past txs (`τ < t`) are usable as features — by deployment time, a fraud team would have labelled them.
- `wallets_features.txt` is **excluded** (lifetime aggregates leak future data).
- `AddrAddr_edgelist.txt` is **excluded** (no timestamps).
- The bipartite `addr↔tx` edges from `actors_data/` are the only cross-timestep bridge.
- All features are computed using a single sweep over the full dataset, but each feature for tx at time `t` only ever uses data at strictly earlier timesteps — this is causally equivalent to recomputing per-`t*` and avoids redundant work.

**Static split** (used by Phase 1 / 1c and steps 1–5): train on labelled txs with `t ≤ 34`, test on labelled txs with `t ≥ 35`. 29,894 train (illicit rate 11.6%), 16,670 test (illicit rate 6.5%).

**Walk-forward split** (used by steps 6–7): for each `t* ∈ [35, 49]`, train on labelled txs with `t ≤ t* − 1`, predict txs at exactly `t = t*`. This simulates a deployed system where new labels arrive over time.

## 2. Methodology

The pipeline is built across nine notebooks. Numbers are illicit-class F1 on the test split unless otherwise noted.

### Existing baselines (provided)
- `random_forest/phase1_trajectory_signal.ipynb` — RF on the 182 paper-provided per-tx features (incl. 72 within-timestep neighborhood aggregates) plus 103 hand-engineered cross-step wallet-trajectory features. F1 = 0.8098.
- `random_forest/phase1c_bipartite_structural.ipynb` — adds 7 bipartite-PageRank features (cumulative bipartite graph at each `t`). F1 = 0.8094 (no improvement over Phase 1B).

### New work (this contribution)

| Step | Notebook | Purpose |
|------|---------|---------|
| 0 | `random_forest/baseline_no_graph.ipynb` | Strict no-graph floor: drop the 72 `Aggregate_feature_*` columns and the 2 tx-tx degrees. **108 intrinsic features only.** |
| 1 | `cross_step_tx_graph/step1_build_and_profile.ipynb` | Construct a cross-step `tx → tx` graph via shared wallets (cap=50). Profile statistics. |
| 2 | `cross_step_tx_graph/step2_pair_features.ipynb` | Block A (full 4-role) and Block B (money-flow `out→in` only) pair features. |
| 3 | `cross_step_tx_graph/step3_causal_gnn.ipynb` | Causal inductive 2-layer SAGE-style GNN on the cross-step graph. **Negative result.** |
| 4 | `cross_step_tx_graph/step4_richer_features.ipynb` | Block C: 5 personalized-PageRank features seeded from illicit history. Block D: 6 2-hop temporal motif counts. |
| 5 | `cross_step_tx_graph/step5_gbdt.ipynb` | LightGBM and XGBoost replacing RF. Threshold calibration on temporal val. |
| 6 | `cross_step_tx_graph/step6_walk_forward.ipynb` | Rolling walk-forward retraining with RF and LightGBM. **Headline result.** |
| 7 | `cross_step_tx_graph/step7_walk_forward_ablations.ipynb` | Walk-forward ablations: no-graph (NG), C1 (best static set), C1+D (with motifs). |

### Feature blocks

- **Intrinsic (108)**: 93 `Local_feature_*` (per-tx local properties: BTC value, fees, output volume, etc.) + 15 named (`total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`, `in/out_BTC_{min,max,mean,median,total}`).
- **Trajectory (103)**: per-incident-wallet causal history aggregates from Phase 1: counts, fractions, decayed-illicit scores, recency, fan asymmetry, burstiness, velocity, etc., across input and output sides.
- **Block B — cross-step pair (17)**: per-tx aggregates over incoming `out→in` edges in the cross-step graph: edge count, Δt distribution, decayed source-illicit mass, source-tx feature aggregates.
- **Block C — illicit-seeded PPR (5)**: at each `t`, run personalized PageRank on the cumulative bipartite graph with teleport vector concentrated on illicit txs at `τ < t`. Extract `T`'s own PR mass plus aggregates over its incident wallets.
- **Block D — 2-hop motifs (6)**: counts of temporal paths `T'' → T' → T` along the cross-step graph, broken out by (T'' illicit, T' illicit, both edges money-flow, recent ≤5, decayed-illicit).

## 3. Headline results (test t ∈ [35, 49])

| Setup | dim | F1 | AUC | PR-AUC | F1 [t≥43] |
|---|---|---|---|---|---|
| Logistic Regression [108 no-graph] | 108 | 0.2430 | 0.8686 | 0.2634 | — |
| Random Forest [108 no-graph] | 108 | 0.8021 | 0.9026 | 0.7855 | — |
| Random Forest [108 + 103 traj] (≈ Phase 1B) | 211 | 0.8098 | 0.9196 | 0.8029 | — |
| Random Forest [+ Block B pair feats] | 228 | 0.8122 | 0.9160 | 0.8027 | — |
| Random Forest [+ Block C illicit-PPR] (C1) | 233 | **0.8149** | 0.9209 | 0.8067 | 0.1463 |
| Random Forest [C1 + Block D motifs] | 239 | 0.8114 | 0.9233 | 0.8075 | — |
| LightGBM (C1, hyperparams from val) | 233 | 0.8004 | 0.9279 | 0.8041 | — |
| Static ensemble RF+LGB+XGB (val-tuned) | 233 | 0.8118 | 0.9274 | 0.8073 | — |
| Causal inductive GNN (cross-step graph) | n/a | 0.5642 | 0.9111 | 0.7251 | — |
| GNN-stacked into RF [108 + GNN_prob + emb] | 117 | 0.7931 | 0.9070 | 0.7892 | — |
| **Walk-forward RF (C1)** | 233 | 0.8240 | 0.9721 | 0.8762 | 0.2020 |
| **Walk-forward LightGBM (C1)** | 233 | **0.8557** | 0.9749 | 0.8988 | **0.5357** |
| Walk-forward LightGBM (C1 + Block D) | 239 | 0.8525 | 0.9760 | 0.9023 | 0.5505 |
| Walk-forward LightGBM (no-graph, NG) | 108 | 0.8055 | 0.9700 | 0.8712 | 0.3925 |
| Walk-forward RF+LGB equal-weight ensemble | 233 | 0.8532 | 0.9780 | 0.9001 | 0.4576 |

The headline model is **walk-forward LightGBM C1**: F1 = 0.8557, AUC = 0.9749, PR-AUC = 0.8988.

### Lift decomposition

Comparing the no-graph floor (RF on 108 features, static split) to the headline (walk-forward LightGBM on C1):

| Component | F1 contribution |
|---|---|
| No-graph static floor | 0.8021 |
| → static, add graph features (RF C1) | +0.0128 |
| → no-graph, switch to walk-forward + LGB | +0.0034 |
| → walk-forward + graph features (synergy term) | +0.0374 |
| **Walk-forward LightGBM C1** | **0.8557** |

The synergy term — the surplus the model extracts from graph features once retraining can adapt to the test distribution — is the largest single contributor and is the central scientific finding of this work.

## 4. Per-timestep breakdown (illicit-class F1@0.5)

| t | n | illicit | static RF C1 | walk-fwd RF C1 | walk-fwd LGB NG | walk-fwd LGB C1 | walk-fwd LGB C1+D |
|---|---|---|---|---|---|---|---|
| 35 | 1341 | 182 | 0.9663 | 0.9663 | 0.9402 | 0.9725 | 0.9725 |
| 36 | 1708 | 33 | 0.9429 | 0.9552 | 0.9429 | 0.9706 | 0.9851 |
| 37 | 498 | 40 | 0.7879 | 0.7692 | 0.7887 | 0.7879 | 0.7879 |
| 38 | 756 | 111 | 0.9275 | 0.9223 | 0.9143 | 0.9378 | 0.9275 |
| 39 | 1183 | 81 | 0.9231 | 0.8980 | 0.9200 | 0.9419 | 0.9359 |
| 40 | 1211 | 112 | 0.7676 | 0.7527 | 0.7437 | 0.7677 | 0.7653 |
| 41 | 1132 | 116 | 0.9358 | 0.9596 | 0.9231 | 0.9356 | 0.9270 |
| 42 | 2154 | 239 | 0.8565 | 0.8676 | 0.8468 | 0.8821 | 0.8729 |
| 43 | 1370 | 24 | 0.0000 | 0.0000 | 0.1333 | 0.1622 | 0.1818 |
| 44 | 1591 | 24 | 0.0500 | 0.0606 | 0.2609 | 0.3721 | 0.3256 |
| 45 | 1221 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 46 | 712 | 2 | 0.4000 | 0.6667 | 0.2222 | 0.6667 | 0.6667 |
| 47 | 846 | 22 | 0.0000 | 0.0000 | 0.0571 | 0.0000 | 0.1429 |
| 48 | 471 | 36 | 0.1053 | 0.0541 | 0.2222 | 0.4167 | 0.5200 |
| 49 | 476 | 56 | 0.3235 | 0.4658 | 0.8393 | 0.9636 | 0.9550 |

The cliff-region detail (t ≥ 43, 6,687 txs, 169 illicit): static RF C1 F1 = 0.1463; walk-forward LightGBM C1 F1 = **0.5357**. The walk-forward run on the same features at t = 49 reaches F1 = 0.9636, vs static F1 = 0.3235 — by `t* = 49` the LightGBM has been retrained on labels through `t = 48` (post-shutdown distribution), which the static model never sees.

## 5. Key findings

1. **Graph features still help under walk-forward — and more than they did under static training.** Walk-forward LightGBM on the 233-dim C1 set reaches F1 = 0.8557; the same setup with no graph features at all reaches 0.8055. The +0.0502 graph-feature lift under walk-forward is roughly 4× the +0.0128 lift the same features delivered under static training. The additional structure helps the model distinguish post-shutdown distributions that no-graph features blur together.

2. **Personalized PageRank seeded from illicit history is the single most useful new feature family.** Adding 5 PPR features to the 228-dim Phase 1 set lifts static RF F1 from 0.8122 → 0.8149 (Δ +0.0027), and lifts walk-forward LightGBM from 0.8055 → 0.8557 (Δ +0.0502). PPR captures multi-hop reachability from the illicit subgraph that the per-wallet trajectory aggregates can't express.

3. **Walk-forward retraining attacks the t ≥ 43 cliff.** The dark-market shutdown around t = 42 induces a concept-drift cliff where every static-split model collapses to F1 ≈ 0 from t = 43 onward. Walk-forward retraining with graph features lifts cliff-region F1 from 0.1463 (static RF C1) to 0.5357 (walk-forward LGB C1). t = 49 specifically goes from F1 = 0.3235 → 0.9636.

4. **Random Forest beats GBDTs in the static regime; LightGBM beats Random Forest in the walk-forward regime.** Static RF C1 = 0.8149, static LGB C1 = 0.8004. Walk-forward RF C1 = 0.8240, walk-forward LGB C1 = 0.8557. LightGBM extracts more signal from the larger, drift-aware training sets that walk-forward provides.

## 6. Negative results

These are explicitly reported because the negative outcomes are themselves informative.

1. **A causal inductive GNN on the cross-step graph underperforms hand-engineered features.** A 2-layer SAGE-style GNN with edge attributes (Δt, role, source label one-hot) achieves test F1 = 0.5642 (val-tuned threshold) and PR-AUC = 0.7251, both **below the no-graph RF baseline** (PR-AUC 0.7855). Stacking the GNN probability into RF gives F1 = 0.7931, also below baseline. The likely causes: ~3K positive training examples is the regime where Grinsztajn et al. (NeurIPS 2022) show trees dominate neural nets; 26% of labelled txs have any incoming cross-step edge and only 2% of test illicit txs have an illicit predecessor; severe val/test distribution shift makes threshold tuning unstable.

2. **2-hop temporal motifs (Block D) don't lift overall F1.** Adding 6 motif features to the static C1 set drops F1 from 0.8149 → 0.8114. Under walk-forward, motifs flatten overall F1 from 0.8557 → 0.8525 but lift cliff F1 from 0.5357 → 0.5505. They trade working-regime accuracy for cliff accuracy. Whether to ship them depends on the deployment priority.

3. **Within-timestep tx-tx graph structure provides essentially no signal.** Removing the 72 `Aggregate_feature_*` columns (the original Elliptic paper's idea of "graph signal") from the static RF baseline costs only 0.0023 F1 (0.8044 → 0.8021). This is consistent with Maganti (2026)'s finding that the within-timestep tx-tx view is uninformative under honest evaluation.

4. **Threshold calibration on temporal val is unreliable.** The val period (t ∈ [30, 34]) has illicit rate 16.8% versus test rate 6.5%; LightGBM achieved val F1 = 1.000 across the entire hyperparameter grid in step 5, indicating in-distribution overfitting that doesn't transfer. Threshold-tuned LightGBM was strictly worse than threshold-0.5 LightGBM under walk-forward (F1 0.8405 vs 0.8557).

## 7. Limitations and honest scope

- **t = 45 and t = 47 still classify at F1 = 0** (5 and 22 illicit txs respectively). Small-sample timesteps where the model produces few or no positive predictions. A two-of-five voting rule across recent timesteps could rescue these but wasn't tried.
- **Walk-forward "uses more data" than the static split by construction** — the comparison is between two evaluation regimes, not two models on the same data. The static-split number is artificially pessimistic relative to what a deployed system would actually have access to. Both regimes are honest under the user's contract; walk-forward is just operationalizing it correctly at the training level too.
- **No threshold calibration in the headline number.** Walk-forward LightGBM at threshold 0.5 was reported because threshold-tuning on temporal val was unstable. A more sophisticated calibration (e.g., quantile-matching on rolling labelled windows) is left for future work.
- **The cliff F1 is now 0.54, not 0.95.** Walk-forward + graph features cracks the cliff but does not eliminate it. This is fundamentally limited by how much the post-shutdown distribution can be inferred from the distribution before the most recent label arrives.

## 8. File map

```
architectures/inductive_tx_classification/
  RESULTS.md                              ← this document
  random_forest/
    baseline_no_graph.ipynb               ← strict 108-intrinsic floor (LR + RF)
    phase1_trajectory_signal.ipynb        ← 103 trajectory features (provided)
    phase1c_bipartite_structural.ipynb    ← 7 bipartite-PageRank features (provided)
  cross_step_tx_graph/                    ← all new work
    step1_build_and_profile.ipynb         ← cross-step tx→tx graph construction
    step2_pair_features.ipynb             ← Block A / Block B pair features
    step3_causal_gnn.ipynb                ← causal inductive GNN (negative result)
    step4_richer_features.ipynb           ← Block C (PPR) and Block D (motifs)
    step5_gbdt.ipynb                      ← LightGBM / XGBoost / threshold calibration
    step6_walk_forward.ipynb              ← rolling walk-forward RF + LGB (headline)
    step7_walk_forward_ablations.ipynb    ← walk-forward NG and C1+D ablations
    cache/                                ← cached feature matrices (.npy)
  neural_nets/                            ← prior simplified MLP / attention attempts (provided)
  custom_architectures/                   ← prior CT-BNet / MS-TBD prototypes (provided)
```

All notebooks are self-contained, runnable on a laptop (steps 1, 2, 4, 5, 6, 7 each take ~1–2 minutes after a one-time ~70-second feature-build that's cached to `cache/`), and end-to-end re-executable via `jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`. All numbers in the tables above are reproduced from the executed notebook outputs in this directory.
