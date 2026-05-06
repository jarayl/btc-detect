"""
Add a `first_t` column to the wallet-wallet (AddrAddr) edge list.

Original `actors_data/AddrAddr_edgelist.txt` has columns
    input_address, output_address
with no timestep, so any temporal model that uses these edges leaks future
information (the edge might encode a co-transaction that hasn't happened yet
at the inference time).

This script computes, for each directed edge `(input_address, output_address)`,
the earliest timestep `first_t` at which the two wallets co-appeared on the
same transaction (input on one side, output on the other). The edge becomes
LEGAL to add to a causal/inductive graph at time t once `t >= first_t`.

Output: `inductive_feature_gen/data/AddrAddr_edgelist_causal.csv` with three
columns: `input_address, output_address, first_t`.

Run from anywhere:
    python inductive_feature_gen/build_addr_addr_first_t.py
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACTORS = os.path.join(REPO, "actors_data")
TX = os.path.join(REPO, "transactions_data")
OUT_PATH = os.path.join(ACTORS, "AddrAddr_edgelist_causal.csv")


def main():
    print(f"repo: {REPO}")

    print("\n[1/4] Loading tx times (txId -> Time step)...")
    tic = time.time()
    txdf = pd.read_csv(os.path.join(TX, "txs_features.csv"),
                       usecols=["txId", "Time step"])
    tx_t = dict(zip(txdf["txId"].values, txdf["Time step"].values.astype(int)))
    print(f"  {len(txdf):,} txs ({time.time() - tic:.1f}s)")

    print("\n[2/4] Loading bipartite edges (AddrTx, TxAddr)...")
    tic = time.time()
    in_df = pd.read_csv(os.path.join(ACTORS, "AddrTx_edgelist.txt"))
    out_df = pd.read_csv(os.path.join(ACTORS, "TxAddr_edgelist.txt"))
    print(f"  input edges:  {len(in_df):,}")
    print(f"  output edges: {len(out_df):,}  ({time.time() - tic:.1f}s)")

    print("\n[3/4] Loading AddrAddr edges and indexing...")
    tic = time.time()
    aa = pd.read_csv(os.path.join(ACTORS, "AddrAddr_edgelist.txt"))
    n_edges = len(aa)
    print(f"  {n_edges:,} AddrAddr edges")

    # Hash set for O(1) lookup of valid pairs
    edge_idx = {}  # (in, out) -> row position in aa
    for i, (a, b) in enumerate(zip(aa["input_address"].values,
                                   aa["output_address"].values)):
        # If duplicate edges exist, keep the first row position; first_t will
        # still be the same for all duplicates since it's a set property.
        if (a, b) not in edge_idx:
            edge_idx[(a, b)] = i
    print(f"  {len(edge_idx):,} unique (in, out) pairs ({time.time() - tic:.1f}s)")

    print("\n[4/4] Per-tx scan: emit (in, out) pairs and record earliest co-tx time...")
    tic = time.time()

    # Group bipartite edges by tx
    tx_in_wallets = defaultdict(list)
    for w, tx in zip(in_df["input_address"].values, in_df["txId"].values):
        tx_in_wallets[tx].append(w)
    tx_out_wallets = defaultdict(list)
    for tx, w in zip(out_df["txId"].values, out_df["output_address"].values):
        tx_out_wallets[tx].append(w)

    # Iterate txs in TIME order so first_t is recorded as the earliest hit.
    txs_sorted = sorted(tx_t.items(), key=lambda kv: kv[1])

    first_t = {}  # (in, out) -> earliest t observed
    n_pair_checks = 0
    for tx, t in txs_sorted:
        ins = tx_in_wallets.get(tx)
        outs = tx_out_wallets.get(tx)
        if not ins or not outs:
            continue
        # Cartesian product of inputs x outputs is the set of co-tx pairs
        # induced by this transaction.
        for a in ins:
            for b in outs:
                pair = (a, b)
                if pair in edge_idx and pair not in first_t:
                    first_t[pair] = t
                n_pair_checks += 1

    print(f"  {n_pair_checks:,} pair checks, {len(first_t):,} edges resolved ({time.time() - tic:.0f}s)")
    n_unresolved = len(edge_idx) - len(first_t)
    if n_unresolved:
        print(f"  WARNING: {n_unresolved:,} unique edges have no matching co-tx in the data (will be NaN)")

    # Assemble output column in the SAME ROW ORDER as the input file.
    print("\nWriting output...")
    tic = time.time()
    ft_col = np.full(n_edges, np.nan, dtype=np.float64)
    miss_examples = []
    for i, (a, b) in enumerate(zip(aa["input_address"].values,
                                   aa["output_address"].values)):
        ft = first_t.get((a, b))
        if ft is not None:
            ft_col[i] = ft
        elif len(miss_examples) < 5:
            miss_examples.append((a, b))
    aa["first_t"] = ft_col
    aa.to_csv(OUT_PATH, index=False)
    print(f"  done ({time.time() - tic:.1f}s)")

    n_missing = int(np.isnan(ft_col).sum())
    print(f"\nWrote: {OUT_PATH}")
    print(f"  rows: {n_edges:,}")
    print(f"  edges with first_t: {n_edges - n_missing:,}")
    print(f"  edges with NaN first_t: {n_missing:,}")
    if miss_examples:
        print(f"  example unresolved: {miss_examples[:3]}")
    if n_edges - n_missing:
        ts_arr = ft_col[~np.isnan(ft_col)]
        print(f"  first_t range: [{int(ts_arr.min())}, {int(ts_arr.max())}]")
        print(f"  first_t distribution (deciles):")
        deciles = np.percentile(ts_arr, [0, 10, 25, 50, 75, 90, 100])
        for p, v in zip([0, 10, 25, 50, 75, 90, 100], deciles):
            print(f"    p{p:>3} = {int(v)}")


if __name__ == "__main__":
    main()
