# stat175-final

Anti–money-laundering classification on the Elliptic++ dataset, with a focus
on causality-respecting (strictly inductive) graph models.

## Repository layout

```
stat175-final/
├── causal_elliptic++/            # rebuilt, leakage-free versions of the dataset
│   ├── transactions_features/    #   txs_features.csv, txs_classes.csv
│   ├── wallet_features/          #   wallets_features_causal.csv, wallets_classes.txt
│   ├── AddrAddr_edgelist_causal.csv  # AddrAddr edges with first_t timestamp
│   ├── AddrTx_edgelist.txt
│   ├── TxAddr_edgelist.txt
│   └── txs_edgelist.csv
├── architectures/
│   ├── elliptic++_causal/        # current work: causal/inductive models on causal_elliptic++
│   │   ├── inductive_feature_gen/  # scripts that produce causal_elliptic++ from raw
│   │   └── models/                 # gnn, rgt, rgt_baseline notebooks
│   ├── inductive_tx_classification/  # tx-level inductive experiments
│   │   ├── cross_step_tx_graph/  # 7-step pipeline: build → features → GNN → GBDT → walk-forward
│   │   ├── custom_architectures/ # ct_bnet, ms_tbd
│   │   ├── neural_nets/          # MLP, attention, transformer baselines
│   │   └── random_forest/        # RF baselines (with and without graph features)
├── requirements.txt
└── README.md
```

## Reading the structure

- **`causal_elliptic++/`** is the Elliptic++ dataset. Files here have had
  future-information leakage removed; see
  `architectures/elliptic++_causal/inductive_feature_gen/README.md` for the
  exact transformations and per-feature definitions.
- **`architectures/`** is organized by experiment family, not by model
  type. Each subdirectory is self-contained: notebooks read from
  `causal_elliptic++/` (or, for older work, from `actors_data/` directly)
  and write any caches alongside themselves.
- **`architectures/elliptic++_causal/`** is the current line of work.
  `inductive_feature_gen/` produces the causal dataset; `models/` contains
  the GNN and Relational Graph Transformer notebooks that consume it.
- **`architectures/inductive_tx_classification/cross_step_tx_graph/`** is a
  numbered pipeline (`step1_…` through `step7_…`); run in order. Its
  `cache/` holds intermediate `.npy` blocks.
- **`architectures/deprecated/`** contains superseded notebooks
  (`embeddings`, `gnn`, `graph_hawkes`, `wallet_sota`,
  `initial_experiements`, `RESULTS.md/html`). Kept locally for reference,
  not version-controlled.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then place the Elliptic++ raw files into `actors_data/` and
`transactions_data/`, and run the two scripts in
`architectures/elliptic++_causal/inductive_feature_gen/` to produce
`causal_elliptic++/`.
