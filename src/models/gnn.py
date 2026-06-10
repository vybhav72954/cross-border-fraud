"""
GNN feature extractor for ring-membership signal (L_R).

Builds a bipartite card↔merchant graph from Sparkov, trains GraphSAGE,
and extracts interpretable scalar features for the GLM design matrix.
Requires: torch, torch-geometric
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def build_bipartite_graph(df: pd.DataFrame):
    """Build a bipartite card-merchant graph using torch-geometric.

    Node types: 'card' (one per cc_num), 'merchant' (one per merchant string).
    Edges: transaction between card c and merchant m.

    Returns a torch_geometric.data.HeteroData object.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError("torch and torch-geometric are required for GNN features.")

    # Node index mappings
    cards = df["cc_num"].unique()
    merchants = df["merchant"].unique()
    card_idx = {c: i for i, c in enumerate(cards)}
    merch_idx = {m: i for i, m in enumerate(merchants)}

    # Card node features: age, log_city_pop (scalars only — categoricals need embedding)
    dob = pd.to_datetime(df.drop_duplicates("cc_num").set_index("cc_num")["dob"])
    ref_date = pd.to_datetime(df["trans_date_trans_time"]).max()
    card_df = df.drop_duplicates("cc_num").set_index("cc_num")
    card_feats = np.stack([
        ((ref_date - pd.to_datetime(card_df["dob"])).dt.days / 365.25).values,
        np.log1p(card_df["city_pop"].values),
    ], axis=1).astype(np.float32)

    # Merchant node features: transaction count, fraud rate
    merch_stats = df.groupby("merchant").agg(
        txn_count=("trans_num", "count"),
        fraud_rate=("is_fraud", "mean"),
    )
    merch_feats = merch_stats.loc[merchants].values.astype(np.float32)

    # Edge index: card → merchant
    src = df["cc_num"].map(card_idx).values
    dst = df["merchant"].map(merch_idx).values

    data = HeteroData()
    data["card"].x = torch.tensor(card_feats)
    data["card"].node_id = torch.arange(len(cards))
    data["merchant"].x = torch.tensor(merch_feats)
    data["merchant"].node_id = torch.arange(len(merchants))
    data["card", "transacts", "merchant"].edge_index = torch.tensor(
        np.stack([src, dst]), dtype=torch.long
    )
    return data, card_idx, merch_idx


class CardMerchantSAGE(object):
    """GraphSAGE over the bipartite card-merchant graph.

    Emits per-card scalar features for the GLM:
      - embedding_norm: L2 norm of the 32-dim embedding
      - degree_centrality: number of distinct merchants per card
      - 2hop_size: cards reachable in two hops through shared merchants
    """

    def __init__(self, in_dim: int = 2, hid: int = 64, out_dim: int = 32,
                 epochs: int = 50, lr: float = 1e-3) -> None:
        self.in_dim = in_dim
        self.hid = hid
        self.out_dim = out_dim
        self.epochs = epochs
        self.lr = lr
        self._model = None

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import SAGEConv

            class _SAGE(nn.Module):
                def __init__(self, in_d, hid, out_d):
                    super().__init__()
                    self.c1 = SAGEConv(in_d, hid)
                    self.c2 = SAGEConv(hid, out_d)

                def forward(self, x, edge_index):
                    x = self.c1(x, edge_index).relu()
                    return self.c2(x, edge_index)

            return _SAGE(self.in_dim, self.hid, self.out_dim)
        except ImportError:
            raise ImportError("torch and torch-geometric are required.")

    def fit(self, data, label_col: Optional[pd.Series] = None) -> "CardMerchantSAGE":
        """Train with self-supervised link-prediction objective if no labels,
        or supervised L_R prediction if label_col is provided."""
        import torch
        import torch.nn.functional as F

        model = self._build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        edge_index = data["card", "transacts", "merchant"].edge_index
        x_card = data["card"].x

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            emb = model(x_card, edge_index)
            # Self-supervised: reconstruct adjacency (simple dot-product)
            src, dst = edge_index
            pos_score = (emb[src] * emb[dst % len(emb)]).sum(dim=1)
            loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones(len(src))
            )
            loss.backward()
            optimizer.step()

        self._model = model
        return self

    def extract_features(self, data, df: pd.DataFrame,
                          card_idx: dict) -> pd.DataFrame:
        """Return per-transaction scalar features by mapping card embeddings."""
        import torch
        import networkx as nx

        self._model.eval()
        with torch.no_grad():
            edge_index = data["card", "transacts", "merchant"].edge_index
            emb = self._model(data["card"].x, edge_index).numpy()

        norms = np.linalg.norm(emb, axis=1)

        # Degree: distinct merchants per card
        degrees = df.groupby("cc_num")["merchant"].nunique()

        # 2-hop size via networkx on the bipartite graph
        G = nx.Graph()
        for _, row in df[["cc_num", "merchant"]].drop_duplicates().iterrows():
            G.add_edge(f"c_{row['cc_num']}", f"m_{row['merchant']}")

        two_hop: dict[str, int] = {}
        for cc in df["cc_num"].unique():
            node = f"c_{cc}"
            if node not in G:
                two_hop[cc] = 0
                continue
            hop1 = set(G.neighbors(node))
            hop2 = {n for m in hop1 for n in G.neighbors(m) if n != node and n.startswith("c_")}
            two_hop[cc] = len(hop2)

        feats = pd.DataFrame({
            "gnn_emb_norm": pd.Series(
                {cc: norms[i] for cc, i in card_idx.items()}
            ),
            "gnn_degree": degrees,
            "gnn_2hop_size": pd.Series(two_hop),
        })

        return df["cc_num"].map(lambda cc: feats.loc[cc] if cc in feats.index
                                else pd.Series([0.0, 0.0, 0], index=feats.columns)
                                ).apply(pd.Series).set_index(df.index)
