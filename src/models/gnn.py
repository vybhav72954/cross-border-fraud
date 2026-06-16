"""
GNN feature extractor for the ring-membership signal.

The ring signature is a TIME-WINDOWED merchant fan-in: ``cards_per_ring``
distinct cards hit one shared merchant inside a short window (see
``inject_ring``). A static lifetime card<->merchant graph blurs this completely
-- a ring merchant just looks like a slightly more popular merchant, and a ring
card looks like a normal card -- which is why the tabular baseline whiffs on ring
(~0.58 AUC). The structure has to be *windowed* for the ring to be visible.

This module exposes two graph-derived signals, both in the CLAUDE.md output set
("embedding norm, degree centrality, ... 2-hop neighborhood size"):

  * ``merchant_window_features`` -- the degree of the merchant node in the
    time-windowed bipartite transaction graph: distinct cards / txn count at a
    transaction's merchant within +/- ``window_hours``. This is the honest
    structural scalar, and it is exactly what a single bipartite message-passing
    step computes. Fast, no training, no torch.

  * ``RingSAGE`` -- a GraphSAGE over a (card, merchant-time-bucket) bipartite
    graph that *learns* the fan-in end-to-end from raw node features via
    degree-sensitive (sum) aggregation, emitting a per-transaction ring score.
    This is the "neural method recovers the signal" demonstration; it should
    match the hand-built structural feature without being handed the degree.

Requires (RingSAGE only): torch, torch-geometric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# ── structural feature: windowed merchant-node degree ───────────────────────

def merchant_window_features(
    df: pd.DataFrame,
    window_hours: float = 2.0,
    time_col: str = "trans_date_trans_time",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Time-windowed merchant fan-in, per transaction.

    For each row at merchant ``m`` and time ``t`` returns:
      - ``merch_win_cards`` : distinct cards transacting at ``m`` in [t-W, t+W]
      - ``merch_win_txns``  : transactions at ``m`` in [t-W, t+W]

    A ring (``cards_per_ring`` distinct cards in a 2h window) lifts
    ``merch_win_cards`` to ~``cards_per_ring``; a legit merchant (a few txns a
    day) sits near 1. This is the merchant node's degree in the bipartite graph
    restricted to a +/- ``window_hours`` time slice.

    Single O(n) sliding-window pass per merchant (two pointers over a
    merchant-then-time sort), so it scales to the full dataset.
    """
    t_ns = pd.to_datetime(df[time_col]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    merch = pd.factorize(df["merchant"])[0]
    card = pd.factorize(df["cc_num"])[0]
    n = len(df)
    w = int(window_hours * 3600 * 1_000_000_000)

    order = np.lexsort((t_ns, merch))  # primary: merchant, secondary: time
    ts, ms, cs = t_ns[order], merch[order], card[order]

    win_cards = np.zeros(n, dtype=np.int32)
    win_txns = np.zeros(n, dtype=np.int32)

    bounds = np.flatnonzero(np.diff(ms)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n]))

    it = zip(starts, ends)
    if show_progress:
        it = tqdm(list(it), desc=f"merchant fan-in (+/-{window_hours}h)")
    for s, e in it:
        gt = ts[s:e].tolist()  # python lists -> fast scalar indexing
        gc = cs[s:e].tolist()
        L = e - s
        counts: dict[int, int] = {}
        lo = hi = distinct = 0
        for i in range(L):
            hi_bound = gt[i] + w
            lo_bound = gt[i] - w
            while hi < L and gt[hi] <= hi_bound:
                c = gc[hi]
                if counts.get(c, 0) == 0:
                    distinct += 1
                counts[c] = counts.get(c, 0) + 1
                hi += 1
            while gt[lo] < lo_bound:
                c = gc[lo]
                counts[c] -= 1
                if counts[c] == 0:
                    distinct -= 1
                lo += 1
            win_cards[s + i] = distinct
            win_txns[s + i] = hi - lo

    res = np.empty((n, 2), dtype=np.int32)
    res[order, 0] = win_cards
    res[order, 1] = win_txns
    return pd.DataFrame(
        {"merch_win_cards": res[:, 0], "merch_win_txns": res[:, 1]},
        index=df.index,
    ).astype(float)


# ── learned signal: GraphSAGE over a (card, merchant-time-bucket) graph ──────

def _card_node_table(df: pd.DataFrame, time_col: str) -> tuple[np.ndarray, dict]:
    """Standardised per-card features [age_z, log_city_pop_z], shared train/test."""
    cards = np.sort(df["cc_num"].unique())
    idx = {c: i for i, c in enumerate(cards)}
    g = df.drop_duplicates("cc_num").set_index("cc_num").loc[cards]
    ref = pd.to_datetime(df[time_col]).max()
    age = ((ref - pd.to_datetime(g["dob"])).dt.days / 365.25).to_numpy()
    logpop = np.log1p(g["city_pop"].to_numpy())
    feat = np.stack([_z(age), _z(logpop)], axis=1).astype(np.float32)
    return feat, idx


def _z(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / s if s > 0 else a - a.mean()


class RingSAGE:
    """GraphSAGE over a bipartite (card, merchant-time-bucket) graph.

    Time is bucketed at ``window_hours`` so a ring's distinct cards all attach to
    the SAME (merchant, bucket) node, giving that node a high card fan-in.
    Degree-sensitive (sum) aggregation lets the bucket embedding reflect that
    fan-in WITHOUT the degree ever being handed in as an input feature -- the
    network has to recover it by message passing. A transaction is scored by its
    (card, bucket) edge: ``MLP([z_card, z_bucket]) -> ring logit``.

    Training uses a class-balanced subsample (all fraud rows + ``n_legit``
    sampled legit) to keep the graph small; inference runs a single full-graph
    forward pass, so it is inductive over unseen buckets.
    """

    def __init__(self, window_hours: float = 2.0, hidden: int = 32,
                 epochs: int = 60, lr: float = 5e-3, n_legit: int = 120_000,
                 seed: int = 0, time_col: str = "trans_date_trans_time") -> None:
        self.window_hours = window_hours
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.n_legit = n_legit
        self.seed = seed
        self.time_col = time_col
        self._model = None
        self._card_idx: dict = {}

    # graph construction -----------------------------------------------------
    def _bucket_ids(self, df: pd.DataFrame) -> np.ndarray:
        t_ns = (pd.to_datetime(df[self.time_col]).to_numpy()
                .astype("datetime64[ns]").astype(np.int64))
        w = int(self.window_hours * 3600 * 1_000_000_000)
        bucket = (t_ns // w)
        merch = pd.factorize(df["merchant"])[0]
        # unique (merchant, bucket) -> contiguous id
        key = merch.astype(np.int64) * (bucket.max() + 2) + bucket
        return pd.factorize(key)[0]

    def _build_graph(self, df: pd.DataFrame, card_idx: dict):
        import torch

        cidx = df["cc_num"].map(card_idx).fillna(0).astype(int).to_numpy()
        bidx = self._bucket_ids(df)
        n_cards = len(card_idx)
        n_buckets = int(bidx.max()) + 1
        b_node = bidx + n_cards  # bucket node ids follow card node ids

        # node features: cards carry identity, buckets are blank+type-flag so the
        # net must derive bucket degree from its card neighbours, not read it off.
        x = np.zeros((n_cards + n_buckets, 3), dtype=np.float32)
        x[:n_cards, :2] = self._card_feat
        x[n_cards:, 2] = 1.0

        src = np.concatenate([cidx, b_node])
        dst = np.concatenate([b_node, cidx])  # undirected
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        return (torch.tensor(x), edge_index,
                torch.tensor(cidx, dtype=torch.long),
                torch.tensor(b_node, dtype=torch.long))

    def _build_model(self):
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv

        class _Net(nn.Module):
            def __init__(self, hid):
                super().__init__()
                self.c1 = SAGEConv(3, hid, aggr="sum")
                self.c2 = SAGEConv(hid, hid, aggr="sum")
                self.head = nn.Sequential(
                    nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 1)
                )

            def encode(self, x, edge_index):
                import torch.nn.functional as F
                h = F.relu(self.c1(x, edge_index))
                return self.c2(h, edge_index)

            def score(self, z, e_card, e_bucket):
                import torch
                return self.head(torch.cat([z[e_card], z[e_bucket]], dim=1)).squeeze(-1)

        return _Net(self.hidden)

    # fit / score ------------------------------------------------------------
    def fit(self, df: pd.DataFrame, ring: np.ndarray) -> "RingSAGE":
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._card_feat, self._card_idx = _card_node_table(df, self.time_col)

        ring = np.asarray(ring).astype(int)
        fraud = np.flatnonzero(ring == 1)
        legit_all = np.flatnonzero(ring == 0)
        n_legit = min(self.n_legit, legit_all.size)
        legit = rng.choice(legit_all, size=n_legit, replace=False)
        sub = np.sort(np.concatenate([fraud, legit]))
        sdf = df.iloc[sub]
        y = torch.tensor(ring[sub], dtype=torch.float32)

        x, edge_index, e_card, e_bucket = self._build_graph(sdf, self._card_idx)
        model = self._build_model()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])

        model.train()
        for _ in tqdm(range(self.epochs), desc="RingSAGE train"):
            opt.zero_grad()
            z = model.encode(x, edge_index)
            logit = model.score(z, e_card, e_bucket)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
            loss.backward()
            opt.step()
        self._model = model
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Per-transaction ring probability via one full-graph forward pass."""
        import torch

        x, edge_index, e_card, e_bucket = self._build_graph(df, self._card_idx)
        self._model.eval()
        with torch.no_grad():
            z = self._model.encode(x, edge_index)
            p = torch.sigmoid(self._model.score(z, e_card, e_bucket))
        return p.numpy()


# ── track E: GraphSAGE on overlapping centered SNAPSHOT graphs ───────────────

class SnapshotRingSAGE(RingSAGE):
    """RingSAGE on overlapping *centered* snapshots instead of fixed ``t//W`` buckets.

    ``RingSAGE`` buckets time at the floor ``t // W``: a ring whose cards straddle a
    bucket boundary is split across two ``(merchant, bucket)`` nodes, so each node sees
    only part of the fan-in -- the discretisation artifact that caps the learned model
    at ~0.841 while the *sliding* ``merchant_window_features`` oracle (a centered +/-W
    window) keeps every ring whole at ~0.959.

    This builds the standard temporal-GNN remedy: merchant nodes are width-``2*W``
    snapshots centered at multiples of ``W`` (=``window_hours``), overlapping 50%. Each
    transaction joins the **two** snapshots whose windows cover it (message passing sees
    both), but is SCORED in the one it is most centered in (nearest center) -- matching
    the oracle's centered window. A ring of spread <= ``W`` therefore lands fully inside
    its most-centered snapshot for rows near the center, recovering the fan-in the floor
    bucket halves. Only the graph construction changes; the 2-layer sum-aggregation
    SAGEConv, balanced-subsample training, and inductive full-graph scoring are inherited.
    """

    def _build_graph(self, df: pd.DataFrame, card_idx: dict):
        import torch

        t_ns = (pd.to_datetime(df[self.time_col]).to_numpy()
                .astype("datetime64[ns]").astype(np.int64))
        w = int(self.window_hours * 3600 * 1_000_000_000)  # = stride; window is +/- w
        merch = pd.factorize(df["merchant"])[0].astype(np.int64)
        cidx = df["cc_num"].map(card_idx).fillna(0).astype(int).to_numpy()
        n = len(df)

        pos = t_ns / w
        m0 = np.floor(pos).astype(np.int64)   # the two covering snapshot centers (in W units)
        m1 = m0 + 1
        jstar = np.where(pos - m0 < 0.5, m0, m1)  # most-centered snapshot -> scored here

        # (merchant, snapshot) -> shared contiguous node id across BOTH memberships, so a
        # row's jstar node coincides with the same node reached via a neighbour's membership.
        span = int(m1.max()) + 2
        codes = pd.factorize(np.concatenate([merch * span + m0, merch * span + m1]))[0]
        b0, b1 = codes[:n], codes[n:]
        bstar = np.where(jstar == m0, b0, b1)

        n_cards = len(card_idx)
        n_buckets = int(codes.max()) + 1
        x = np.zeros((n_cards + n_buckets, 3), dtype=np.float32)
        x[:n_cards, :2] = self._card_feat
        x[n_cards:, 2] = 1.0
        bn0, bn1, bnstar = b0 + n_cards, b1 + n_cards, bstar + n_cards

        # undirected edges for BOTH snapshot memberships
        src = np.concatenate([cidx, cidx, bn0, bn1])
        dst = np.concatenate([bn0, bn1, cidx, cidx])
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        return (torch.tensor(x), edge_index,
                torch.tensor(cidx, dtype=torch.long),
                torch.tensor(bnstar, dtype=torch.long))
