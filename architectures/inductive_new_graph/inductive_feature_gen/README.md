# inductive_feature_gen

Causality-respecting versions of the two Elliptic++ actor-side files that
otherwise leak future information into the present.

## The leakage problem

Elliptic++ ships two actor-side files that, in their original form, cannot be
used in any temporal / strictly inductive model without leaking future
information:

1. **`actors_data/wallets_features.txt`** — has one row per `(wallet, timestep)`
   where the wallet appeared, but the **feature values on every row are
   lifetime aggregates**. They are byte-for-byte identical across all
   timesteps for the same wallet. So the row for wallet `w` at timestep `T`
   already encodes things that have not happened yet at `T`. Verified:

   ```
   $ grep "^13vHWR3iLsHeYwT42RnuKYNBoVPrKKZgRv," wallets_features.txt
     ... at every timestep: total_txs=1471, btc_transacted_total=3997.20, ...
   ```

2. **`actors_data/AddrAddr_edgelist.txt`** — has 2,868,964 directed edges
   `(input_address, output_address)` with **no timestamp**. The edge
   represents "these two wallets co-transacted at some point during the
   dataset's lifetime," but we don't know **when**, so we can't add it to a
   temporal graph without leaking.

This directory rebuilds both files so they are safe to use in causal /
strictly inductive models.

## Outputs

After running both build scripts, you get:

| File | Schema | Size |
|---|---|---|
| `actors_data/wallets_features_causal.csv` | same 57 columns + same row count as the original | ~1.27M rows |
| `actors_data/AddrAddr_edgelist_causal.csv` | original 2 columns + new `first_t` | ~2.87M rows |

Both are drop-in replacements (`wallets_features_causal.csv` has the same
schema as the original, including the typo `num_txs_as receiver` with a
space).

## How to run

```bash
python inductive_feature_gen/build_wallets_features_causal.py
python inductive_feature_gen/build_addr_addr_first_t.py
```

They read from `actors_data/` and `transactions_data/` and write
*alongside* the originals into `actors_data/` (with the `_causal` suffix).
Originals are not modified.

---

## Wallet feature semantics (cumulative through `Time step` T)

For each row `(address=w, Time step=T)` in the output, every feature is
computed using **only events of `w` with `t' ≤ T`**, where an "event" is an
instance of `w` appearing on a transaction (either as input or output)
according to `AddrTx_edgelist.txt` / `TxAddr_edgelist.txt`. The feature
values therefore monotonically build up through T.

### Resolution caveat: "blocks" → timesteps

The original feature names say `*_block*` (e.g.
`first_block_appeared_in`, `lifetime_in_blocks`, `blocks_btwn_txs_*`) and
the original file populates them with **real Bitcoin block numbers**. Our
`txs_features.csv` only carries the **timestep** (49 timesteps total, each
spanning ~2 weeks / ~2,000 blocks), so we cannot recover per-tx block
numbers. We therefore use **timesteps as the time unit everywhere**, while
**keeping the original column names** so this file is a drop-in
replacement. Treat any "block" column as "timestep" semantically.

### Per-feature definitions

Notation:
- `events(w, ≤T)` = `w`'s appearances on txs with `t' ≤ T`, sorted by `t`.
- `evt_in` = events where `w` is an input address; `evt_out` = output side.
- For each event, we have the host tx's `total_BTC`, `fees`,
  `num_input_addresses`, `num_output_addresses`.
- "Counterparties of event e": for `e ∈ evt_in`, the **output** addresses
  of e's host tx (excluding `w` itself); for `e ∈ evt_out`, the **input**
  addresses of e's host tx (excluding `w` itself).

| Column | Definition |
|---|---|
| `address` | `w` |
| `Time step` | `T` |
| `num_txs_as_sender` | `\|evt_in\|` |
| `num_txs_as receiver` *(sic, space)* | `\|evt_out\|` |
| `first_block_appeared_in` | `min t over events(w, ≤T)` (timestep proxy) |
| `last_block_appeared_in` | `T` (since `w` appears at `T` by definition of this row) |
| `lifetime_in_blocks` | `T - first_block_appeared_in` (timestep proxy) |
| `total_txs` | num **distinct** txs in `events(w, ≤T)` (a tx where `w` is both input and output counts once, matching the original Elliptic++ definition) |
| `first_sent_block` | `min t over evt_in`, else 0 |
| `first_received_block` | `min t over evt_out`, else 0 |
| `num_timesteps_appeared_in` | num distinct timesteps in `events(w, ≤T)` |
| `btc_transacted_{total,min,max,mean,median}` | aggregates over the **host tx's `total_BTC`** for each event |
| `btc_sent_{total,...,median}` | same, restricted to `evt_in` |
| `btc_received_{total,...,median}` | same, restricted to `evt_out` |
| `fees_{total,...,median}` | aggregates over the host tx's `fees` |
| `fees_as_share_{total,...,median}` | aggregates over `fees / total_BTC` per event (0 if `total_BTC=0`) |
| `blocks_btwn_txs_{total,...,median}` | gaps between consecutive **distinct timesteps** in `events(w, ≤T)` |
| `blocks_btwn_input_txs_{total,...,median}` | same for `evt_in` |
| `blocks_btwn_output_txs_{total,...,median}` | same for `evt_out` |
| `num_addr_transacted_multiple` | count of distinct counterparty wallets seen ≥ 2 times across `events(w, ≤T)` |
| `transacted_w_address_total` | sum over events of (counterparty count of host tx) — i.e. for each event in `evt_in` it adds the tx's `num_output_addresses` (minus 1 if `w` is also output), and analogously for `evt_out` |
| `transacted_w_address_{min,max,mean,median}` | distribution stats over those per-event counterparty counts |

### Decisions that depart from "exact byte-for-byte match"

These features cannot be reproduced byte-for-byte from public data; we
chose explicit, defensible cumulative rules (consistent with the column
names but documented here so they can't be misread):

1. **`btc_sent` / `btc_received` use the host tx's `total_BTC`**, not the
   wallet's share. Bitcoin transactions can have multiple input and output
   wallets and the dataset doesn't ship per-(wallet, tx) BTC attributions,
   so we credit each participating wallet with the whole-tx BTC. This is
   consistent with the way `total_BTC` is populated in `txs_features.csv`
   and avoids fabricating a share that would be guesswork.

2. **`fees_as_share` is `fees / total_BTC` per event**, then aggregated.
   When `total_BTC == 0` we use 0 (no fee-as-share is well-defined).

3. **`*_block*` columns hold timesteps, not Bitcoin block numbers** (see
   Resolution caveat above).

4. **`first_block_appeared_in`** is `min t` over events, which equals the
   wallet's first observed timestep up to T. Once a wallet has been
   observed it can never get earlier, so this column is monotone
   non-decreasing... wait, actually constant once set. Verified by build.

5. **Counterparty stats** (`transacted_w_address_*`,
   `num_addr_transacted_multiple`) come from the bipartite addr↔tx graph:
   for each tx where `w` is on side X, the counterparties are the wallets
   on the *other* side of the same tx (excluding `w` itself if it appears
   on both sides — this happens for change addresses). This is the
   simplest defensible rule and matches the spirit of "address transacted
   with."

6. **Empty-list aggregates yield 0**, not NaN. (E.g. a wallet that has
   only ever been a receiver has `btc_sent_*` all zero.) This matches the
   original file's typical behavior.

7. **Counterparty self-exclusion**: when `w` is both input and output of
   the same tx, `w` is excluded from its own counterparty list. This
   correctly handles change-address self-loops.

---

## AddrAddr edge semantics

For each directed edge `(input_address, output_address)` in the original
`AddrAddr_edgelist.txt`, we add a column **`first_t`** equal to the
**earliest timestep** at which the two wallets co-appeared on the same
transaction, with `input_address` on the input side and `output_address`
on the output side.

Algorithm (single pass over txs in time order):

```
for tx in sorted(transactions, key=t):
    for w_in in inputs(tx):
        for w_out in outputs(tx):
            if (w_in, w_out) is a known AddrAddr edge and not yet seen:
                first_t[(w_in, w_out)] = t
```

In a strictly inductive temporal model, you should add edge `e` to the
graph at time `T` only if `T >= first_t(e)`. This makes AddrAddr edges
safe to use without leakage.

### What about edges that are never resolved?

If an AddrAddr edge has no matching `(input ∈ inputs(tx), output ∈
outputs(tx))` co-occurrence anywhere in the data, `first_t` is `NaN`. The
build script reports the count and three example pairs. In practice the
script should resolve essentially every edge if AddrAddr was derived from
the same txs we have.
