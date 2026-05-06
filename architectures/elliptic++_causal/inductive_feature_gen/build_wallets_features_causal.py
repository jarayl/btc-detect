"""
Build a causal/cumulative version of wallets_features.txt.

Original Elliptic++ `wallets_features.txt` has one row per (wallet, timestep)
where the wallet appeared, but the feature values on every row are LIFETIME
aggregates - byte-for-byte constant across all timesteps for the same wallet.
Verified empirically:

    grep "^13vHWR3iLsHeYwT42RnuKYNBoVPrKKZgRv," wallets_features.txt
        ... col 'total_txs' = 1471 at every timestep (lifetime value).

Using these features as wallet node features at any timestep T < end_of_life
leaks future information. This script rebuilds the file so that the row at
(wallet=w, Time step=T) reflects only events with t' <= T - i.e. as if T
were the end of w's life.

Output schema is byte-identical to the original (same column names, same row
count, same (address, Time step) row identifiers). Drop-in replacement.

See README.md for per-feature semantics. Run from anywhere:

    python inductive_feature_gen/build_wallets_features_causal.py
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from statistics import median

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACTORS = os.path.join(REPO, "actors_data")
TX = os.path.join(REPO, "transactions_data")
OUT_PATH = os.path.join(ACTORS, "wallets_features_causal.csv")

# Column order matches actors_data/wallets_features.txt exactly, including
# the typo "num_txs_as receiver" (space, not underscore) in the original.
OUT_COLS = [
    "address", "Time step",
    "num_txs_as_sender", "num_txs_as receiver",
    "first_block_appeared_in", "last_block_appeared_in", "lifetime_in_blocks",
    "total_txs", "first_sent_block", "first_received_block",
    "num_timesteps_appeared_in",
    "btc_transacted_total", "btc_transacted_min", "btc_transacted_max",
    "btc_transacted_mean", "btc_transacted_median",
    "btc_sent_total", "btc_sent_min", "btc_sent_max",
    "btc_sent_mean", "btc_sent_median",
    "btc_received_total", "btc_received_min", "btc_received_max",
    "btc_received_mean", "btc_received_median",
    "fees_total", "fees_min", "fees_max", "fees_mean", "fees_median",
    "fees_as_share_total", "fees_as_share_min", "fees_as_share_max",
    "fees_as_share_mean", "fees_as_share_median",
    "blocks_btwn_txs_total", "blocks_btwn_txs_min", "blocks_btwn_txs_max",
    "blocks_btwn_txs_mean", "blocks_btwn_txs_median",
    "blocks_btwn_input_txs_total", "blocks_btwn_input_txs_min",
    "blocks_btwn_input_txs_max", "blocks_btwn_input_txs_mean",
    "blocks_btwn_input_txs_median",
    "blocks_btwn_output_txs_total", "blocks_btwn_output_txs_min",
    "blocks_btwn_output_txs_max", "blocks_btwn_output_txs_mean",
    "blocks_btwn_output_txs_median",
    "num_addr_transacted_multiple",
    "transacted_w_address_total", "transacted_w_address_min",
    "transacted_w_address_max", "transacted_w_address_mean",
    "transacted_w_address_median",
]


def _stats(xs):
    """Return (total, min, max, mean, median). Empty list -> all zeros."""
    if not xs:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    total = sum(xs)
    return (total, min(xs), max(xs), total / len(xs), median(xs))


def _gaps(ts):
    """Gaps between consecutive distinct timesteps in sorted order."""
    if not ts:
        return []
    s = sorted(set(ts))
    if len(s) < 2:
        return []
    return [s[i + 1] - s[i] for i in range(len(s) - 1)]


def _row_dict(w, T, n_in, n_out, first_t, first_in_t, first_out_t,
              all_btc, in_btc, out_btc, all_fee, all_share,
              all_t, in_t, out_t, counterparty_count, all_n_count,
              distinct_txs):
    last_t = T  # wallet appears at T by construction of emission point
    lifetime = (last_t - first_t) if first_t is not None else 0
    n_distinct_t = len(set(all_t)) if all_t else 0
    n_distinct_tx = len(distinct_txs)
    btc = _stats(all_btc)
    bs = _stats(in_btc)
    br = _stats(out_btc)
    f = _stats(all_fee)
    fs = _stats(all_share)
    g = _stats(_gaps(all_t))
    gi = _stats(_gaps(in_t))
    go = _stats(_gaps(out_t))
    nc = _stats(all_n_count)
    n_addr_mult = sum(1 for c in counterparty_count.values() if c > 1)
    return {
        "address": w,
        "Time step": int(T),
        "num_txs_as_sender": n_in,
        "num_txs_as receiver": n_out,
        "first_block_appeared_in": first_t if first_t is not None else 0,
        "last_block_appeared_in": last_t,
        "lifetime_in_blocks": lifetime,
        "total_txs": n_distinct_tx,
        "first_sent_block": first_in_t if first_in_t is not None else 0,
        "first_received_block": first_out_t if first_out_t is not None else 0,
        "num_timesteps_appeared_in": n_distinct_t,
        "btc_transacted_total": btc[0], "btc_transacted_min": btc[1],
        "btc_transacted_max": btc[2], "btc_transacted_mean": btc[3],
        "btc_transacted_median": btc[4],
        "btc_sent_total": bs[0], "btc_sent_min": bs[1],
        "btc_sent_max": bs[2], "btc_sent_mean": bs[3],
        "btc_sent_median": bs[4],
        "btc_received_total": br[0], "btc_received_min": br[1],
        "btc_received_max": br[2], "btc_received_mean": br[3],
        "btc_received_median": br[4],
        "fees_total": f[0], "fees_min": f[1], "fees_max": f[2],
        "fees_mean": f[3], "fees_median": f[4],
        "fees_as_share_total": fs[0], "fees_as_share_min": fs[1],
        "fees_as_share_max": fs[2], "fees_as_share_mean": fs[3],
        "fees_as_share_median": fs[4],
        "blocks_btwn_txs_total": g[0], "blocks_btwn_txs_min": g[1],
        "blocks_btwn_txs_max": g[2], "blocks_btwn_txs_mean": g[3],
        "blocks_btwn_txs_median": g[4],
        "blocks_btwn_input_txs_total": gi[0], "blocks_btwn_input_txs_min": gi[1],
        "blocks_btwn_input_txs_max": gi[2], "blocks_btwn_input_txs_mean": gi[3],
        "blocks_btwn_input_txs_median": gi[4],
        "blocks_btwn_output_txs_total": go[0], "blocks_btwn_output_txs_min": go[1],
        "blocks_btwn_output_txs_max": go[2], "blocks_btwn_output_txs_mean": go[3],
        "blocks_btwn_output_txs_median": go[4],
        "num_addr_transacted_multiple": n_addr_mult,
        "transacted_w_address_total": nc[0], "transacted_w_address_min": nc[1],
        "transacted_w_address_max": nc[2], "transacted_w_address_mean": nc[3],
        "transacted_w_address_median": nc[4],
    }


def main():
    print(f"repo: {REPO}")

    print("\n[1/5] Loading tx features (txId, Time step, total_BTC, fees)...")
    tic = time.time()
    txdf = pd.read_csv(
        os.path.join(TX, "txs_features.csv"),
        usecols=["txId", "Time step", "total_BTC", "fees"],
    )
    txid = txdf["txId"].values
    tx_t = dict(zip(txid, txdf["Time step"].values.astype(int)))
    tx_btc = dict(zip(txid, txdf["total_BTC"].fillna(0.0).values.astype(float)))
    tx_fee = dict(zip(txid, txdf["fees"].fillna(0.0).values.astype(float)))
    print(f"  {len(txdf):,} txs ({time.time() - tic:.1f}s)")

    print("\n[2/5] Loading bipartite edges (AddrTx, TxAddr)...")
    tic = time.time()
    in_df = pd.read_csv(os.path.join(ACTORS, "AddrTx_edgelist.txt"))
    out_df = pd.read_csv(os.path.join(ACTORS, "TxAddr_edgelist.txt"))
    print(f"  input edges:  {len(in_df):,}")
    print(f"  output edges: {len(out_df):,}  ({time.time() - tic:.1f}s)")

    print("\n[3/5] Building per-tx wallet sets and per-wallet event timelines...")
    tic = time.time()
    tx_in_wallets = defaultdict(list)
    for w, t in zip(in_df["input_address"].values, in_df["txId"].values):
        tx_in_wallets[t].append(w)
    tx_out_wallets = defaultdict(list)
    for t, w in zip(out_df["txId"].values, out_df["output_address"].values):
        tx_out_wallets[t].append(w)

    wallet_events = defaultdict(list)
    for w, tx in zip(in_df["input_address"].values, in_df["txId"].values):
        if tx in tx_t:
            wallet_events[w].append(
                (tx_t[tx], tx, "in", tx_btc[tx], tx_fee[tx])
            )
    for tx, w in zip(out_df["txId"].values, out_df["output_address"].values):
        if tx in tx_t:
            wallet_events[w].append(
                (tx_t[tx], tx, "out", tx_btc[tx], tx_fee[tx])
            )
    n_events = sum(len(v) for v in wallet_events.values())
    print(f"  {len(wallet_events):,} wallets, {n_events:,} events ({time.time() - tic:.1f}s)")

    print("\n[4/5] Reading original (address, Time step) row index...")
    tic = time.time()
    orig = pd.read_csv(
        os.path.join(ACTORS, "wallets_features.txt"),
        usecols=["address", "Time step"],
    )
    orig["Time step"] = orig["Time step"].astype(int)
    address_emit_ts = defaultdict(list)
    for a, t in zip(orig["address"].values, orig["Time step"].values):
        address_emit_ts[a].append(int(t))
    for a in address_emit_ts:
        address_emit_ts[a].sort()
    n_orig = len(orig)
    print(f"  {n_orig:,} (address, t) rows  ({time.time() - tic:.1f}s)")

    print("\n[5/5] Computing causal cumulative features and streaming to CSV...")
    tic = time.time()
    n_processed = 0
    n_emitted = 0
    n_total = len(wallet_events)
    with open(OUT_PATH, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUT_COLS)
        writer.writeheader()

        for w, events in wallet_events.items():
            emit_ts = address_emit_ts.get(w)
            if not emit_ts:
                continue
            events.sort()  # by (t, tx, ...)

            all_btc, in_btc, out_btc = [], [], []
            all_fee, all_share, all_n_count = [], [], []
            all_t, in_t, out_t = [], [], []
            n_in = n_out = 0
            first_t = first_in_t = first_out_t = None
            counterparty_count = defaultdict(int)
            distinct_txs = set()
            ei = 0

            for emit_t in emit_ts:
                while ei < len(events) and events[ei][0] <= emit_t:
                    t, tx, role, btc, fee = events[ei]
                    all_btc.append(btc)
                    all_fee.append(fee)
                    all_share.append((fee / btc) if btc > 0 else 0.0)
                    all_t.append(t)
                    distinct_txs.add(tx)
                    if first_t is None:
                        first_t = t
                    if role == "in":
                        n_in += 1
                        in_btc.append(btc)
                        in_t.append(t)
                        if first_in_t is None:
                            first_in_t = t
                        # Counterparties = output wallets of host tx, excluding self.
                        cps = [cp for cp in tx_out_wallets.get(tx, ()) if cp != w]
                    else:
                        n_out += 1
                        out_btc.append(btc)
                        out_t.append(t)
                        if first_out_t is None:
                            first_out_t = t
                        cps = [cp for cp in tx_in_wallets.get(tx, ()) if cp != w]
                    for cp in cps:
                        counterparty_count[cp] += 1
                    all_n_count.append(len(cps))
                    ei += 1

                writer.writerow(_row_dict(
                    w, emit_t, n_in, n_out, first_t, first_in_t, first_out_t,
                    all_btc, in_btc, out_btc, all_fee, all_share,
                    all_t, in_t, out_t, counterparty_count, all_n_count,
                    distinct_txs,
                ))
                n_emitted += 1

            n_processed += 1
            if n_processed % 100_000 == 0:
                pct = 100.0 * n_processed / n_total
                print(f"  {n_processed:,}/{n_total:,} wallets ({pct:.0f}%, {time.time() - tic:.0f}s elapsed)")

    print(f"  emitted {n_emitted:,} rows ({time.time() - tic:.0f}s)")
    print(f"\nWrote: {OUT_PATH}")
    if n_emitted != n_orig:
        print(f"WARNING: emitted row count {n_emitted:,} != original {n_orig:,}")
    else:
        print(f"Row count matches original ({n_emitted:,}).")


if __name__ == "__main__":
    main()
