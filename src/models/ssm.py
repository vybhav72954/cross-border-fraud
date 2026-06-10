"""
Mamba / State-Space Model feature extractor for temporal fraud signals (L_V, L_T).

Each card's transaction history is treated as a time-ordered sequence.
The final SSM hidden state is the per-card behavioral embedding, reduced
to interpretable scalars for the GLM design matrix.

GPU path: mamba-ssm (install separately, requires CUDA 12+)
CPU path: discretized SSM implemented in numpy/torch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class DiscretizedSSM(nn.Module):
    """Simple discretized linear SSM — CPU-compatible fallback.

    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C * h_t

    A_bar, B_bar derived from continuous A, B via zero-order hold.
    """

    def __init__(self, d_input: int = 6, d_state: int = 32) -> None:
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input

        # Continuous-time parameters (learnable)
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_input) * 0.01)
        self.C = nn.Parameter(torch.randn(1, d_state) * 0.01)
        self.log_delta = nn.Parameter(torch.zeros(1))  # log discretization step

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x: (batch, seq_len, d_input)

        Returns
        -------
        y:  (batch, seq_len, 1) — output sequence
        h:  (batch, d_state)   — final hidden state (the behavioral embedding)
        """
        batch, seq_len, _ = x.shape
        delta = torch.exp(self.log_delta)

        # Zero-order hold discretization
        A_bar = torch.matrix_exp(self.A * delta)
        B_bar = torch.linalg.solve(self.A, (A_bar - torch.eye(self.d_state)) @ self.B)

        h = torch.zeros(batch, self.d_state, device=x.device)
        ys = []
        for t in range(seq_len):
            h = h @ A_bar.T + x[:, t, :] @ B_bar.T
            y_t = h @ self.C.T
            ys.append(y_t)

        return torch.stack(ys, dim=1), h


def _build_card_sequences(df: pd.DataFrame, max_len: int = 256) -> tuple[np.ndarray, list]:
    """Build (n_cards, max_len, d_input) input tensor from Sparkov per-card histories.

    Input features per token:
      0: log inter-arrival seconds (or 0 for first transaction)
      1: log(amt + 1)
      2: merch_lat (normalized)
      3: merch_long (normalized)
      4: hour-of-day / 24 (cyclic sin)
      5: hour-of-day / 24 (cyclic cos)
    """
    df = df.copy()
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_dt"])

    lat_mean, lat_std = df["merch_lat"].mean(), df["merch_lat"].std() + 1e-8
    lon_mean, lon_std = df["merch_long"].mean(), df["merch_long"].std() + 1e-8

    cards = df["cc_num"].unique().tolist()
    n_cards = len(cards)
    d = 6
    X = np.zeros((n_cards, max_len, d), dtype=np.float32)

    for i, cc in enumerate(cards):
        g = df[df["cc_num"] == cc].reset_index(drop=True)
        n = min(len(g), max_len)
        times = g["trans_dt"].values

        for t in range(n):
            if t == 0:
                iat = 0.0
            else:
                diff = (times[t] - times[t - 1]) / np.timedelta64(1, "s")
                iat = np.log1p(max(diff, 0))
            hour = g.at[t, "trans_dt"].hour if hasattr(g.at[t, "trans_dt"], "hour") else pd.Timestamp(times[t]).hour
            X[i, t, 0] = iat
            X[i, t, 1] = np.log1p(g.at[t, "amt"])
            X[i, t, 2] = (g.at[t, "merch_lat"] - lat_mean) / lat_std
            X[i, t, 3] = (g.at[t, "merch_long"] - lon_mean) / lon_std
            X[i, t, 4] = np.sin(2 * np.pi * hour / 24)
            X[i, t, 5] = np.cos(2 * np.pi * hour / 24)

    return X, cards


class MambaExtractor:
    """Train the SSM and extract per-card scalar features for the GLM.

    Features handed to the GLM:
      ssm_emb_mean   — mean of final hidden state
      ssm_emb_std    — std of final hidden state
      ssm_emb_max    — max activation
      ssm_burst_score — scalar head: P(L_V=1 | embedding)
      ssm_timing_score — scalar head: P(L_T=1 | embedding)
    """

    def __init__(self, d_state: int = 32, epochs: int = 20,
                 lr: float = 1e-3, batch_size: int = 64,
                 max_seq_len: int = 256) -> None:
        self.d_state = d_state
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.ssm: DiscretizedSSM | None = None
        self.burst_head: nn.Linear | None = None
        self.timing_head: nn.Linear | None = None

    def fit(self, df: pd.DataFrame) -> "MambaExtractor":
        """Train SSM with reconstruction + label-prediction objectives."""
        X_np, cards = _build_card_sequences(df, self.max_seq_len)
        card_to_idx = {cc: i for i, cc in enumerate(cards)}

        # Per-card L_V and L_T labels (1 if any transaction in card is flagged)
        card_labels = df.groupby("cc_num")[["L_V", "L_T"]].max()

        self.ssm = DiscretizedSSM(d_input=6, d_state=self.d_state)
        self.burst_head = nn.Linear(self.d_state, 1)
        self.timing_head = nn.Linear(self.d_state, 1)

        params = (list(self.ssm.parameters()) +
                  list(self.burst_head.parameters()) +
                  list(self.timing_head.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.lr)

        X_t = torch.tensor(X_np)
        n = len(cards)

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                x_b = X_t[idx]
                cc_b = [cards[i] for i in idx.tolist()]

                optimizer.zero_grad()
                y_hat, h = self.ssm(x_b)

                # Reconstruction loss (predict next token)
                recon_loss = nn.functional.mse_loss(y_hat[:, :-1, :],
                                                     x_b[:, 1:, :1])

                # Supervised heads where labels exist
                lv = torch.tensor(
                    [card_labels.loc[cc, "L_V"] if cc in card_labels.index else 0
                     for cc in cc_b], dtype=torch.float32
                ).unsqueeze(1)
                lt = torch.tensor(
                    [card_labels.loc[cc, "L_T"] if cc in card_labels.index else 0
                     for cc in cc_b], dtype=torch.float32
                ).unsqueeze(1)

                burst_loss = nn.functional.binary_cross_entropy_with_logits(
                    self.burst_head(h), lv
                )
                timing_loss = nn.functional.binary_cross_entropy_with_logits(
                    self.timing_head(h), lt
                )

                loss = recon_loss + burst_loss + timing_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={epoch_loss:.4f}")

        self._X_np = X_np
        self._cards = cards
        return self

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-transaction scalar features mapped from card embeddings."""
        assert self.ssm is not None, "Call fit() first."

        X_np, cards = _build_card_sequences(df, self.max_seq_len)
        X_t = torch.tensor(X_np)

        self.ssm.eval()
        self.burst_head.eval()
        self.timing_head.eval()

        with torch.no_grad():
            _, h = self.ssm(X_t)
            burst = torch.sigmoid(self.burst_head(h)).squeeze(1).numpy()
            timing = torch.sigmoid(self.timing_head(h)).squeeze(1).numpy()
            h_np = h.numpy()

        card_feats = pd.DataFrame({
            "ssm_emb_mean": h_np.mean(axis=1),
            "ssm_emb_std": h_np.std(axis=1),
            "ssm_emb_max": h_np.max(axis=1),
            "ssm_burst_score": burst,
            "ssm_timing_score": timing,
        }, index=cards)

        return df["cc_num"].map(
            lambda cc: card_feats.loc[cc] if cc in card_feats.index
            else pd.Series([0.0] * 5, index=card_feats.columns)
        ).apply(pd.Series).set_index(df.index)
