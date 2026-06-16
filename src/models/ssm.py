"""
SSM / Mamba feature extractor for the temporal (L_T) signal.

The temporal signature is a transaction at one of the CARD'S rarest hours-of-day
(see ``inject_temporal``) -- a per-card anomaly. Global ``hour_sin``/``hour_cos``
can only express the wall-clock hour, not "rare FOR THIS CARD", so the tabular
baseline tops out around 0.70. A state-space model scanning one sequence per
card accumulates that card's timing profile in its hidden state and scores each
transaction by how out-of-distribution its hour is. The signal is
PER-TRANSACTION, so this emits a per-token score.

Two signals, mirroring the GNN ring slot:
  * ``card_hour_rarity`` -- per transaction, 1 - the card's historical share of
    this hour-of-day (high = anomalous). The interpretable oracle feature; what
    the SSM should learn. Fast, no torch.
  * ``TemporalSSM`` -- a discretized DIAGONAL SSM, S4D/HiPPO-style: the state is
    a bank of causal exponential-moving-average histograms of the hour-of-day at
    FIXED multi-timescale decays (the diagonal ``A``), and only the readout is
    learned (a small MLP head). Fixing ``A`` lets the states be precomputed once
    with ``scipy.signal.lfilter`` -- no backprop through time, so it trains in
    seconds on CPU instead of grad-through-time over ~1300-long sequences. The
    head still has to LEARN to read the current hour's habituality out of the
    decayed histogram; the rarity is never handed in as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

TIME_COL = "trans_date_trans_time"
N_HOURS = 24
DECAYS = (0.95, 0.99, 0.999)  # diagonal-A timescales: ~20 / ~100 / ~1000 txns


# ── interpretable oracle: per-card hour rarity ──────────────────────────────

def card_hour_rarity(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.Series:
    """Per transaction: 1 - the cardholder's share of this hour-of-day.

    A card that never (rarely) transacts at hour h gets ~1.0 here for a txn at
    hour h; its habitual hours sit near 0. The card-relative timing feature the
    global hour_sin/cos cannot express, and the quantity the SSM must recover.
    """
    hour = pd.to_datetime(df[time_col]).dt.hour
    cc = df["cc_num"]
    pair = cc.astype(str) + "_" + hour.astype(str)
    pair_n = pair.map(pair.value_counts())
    card_n = cc.map(cc.value_counts())
    return (1.0 - pair_n / card_n).astype(float)


# ── diagonal-SSM states: causal decayed hour-histograms ─────────────────────

def _sorted_groups(df: pd.DataFrame, time_col: str):
    dt = pd.to_datetime(df[time_col])
    t_ns = dt.to_numpy().astype("datetime64[ns]").astype(np.int64)
    card = pd.factorize(df["cc_num"])[0]
    order = np.lexsort((t_ns, card))  # primary card, secondary time
    cs = card[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(df)]))
    return order, starts, ends, dt


def hour_ema_states(df: pd.DataFrame, decays=DECAYS, time_col: str = TIME_COL) -> np.ndarray:
    """Per transaction, the card's CAUSAL decayed hour-histogram at each decay.

    For decay ``a``: h_t = a*h_{t-1} + (1-a)*onehot(hour_t), read BEFORE the
    current token (profile of strictly-earlier txns of the same card). Computed
    per card with ``scipy.signal.lfilter`` (fixed diagonal A => no grad-through-
    time). Returns (n, len(decays)*N_HOURS), aligned to df rows.
    """
    n = len(df)
    order, starts, ends, dt = _sorted_groups(df, time_col)
    hour = dt.dt.hour.to_numpy()
    oh = np.zeros((n, N_HOURS), dtype=np.float64)
    oh[np.arange(n), hour] = 1.0
    oh_s = oh[order]

    feats = np.zeros((n, len(decays) * N_HOURS), dtype=np.float32)
    for di, a in enumerate(decays):
        b, aa = [1.0 - a], [1.0, -a]
        col = di * N_HOURS
        block = np.zeros((n, N_HOURS), dtype=np.float64)
        for s, e in zip(starts, ends):
            filt = lfilter(b, aa, oh_s[s:e], axis=0)  # EMA incl. current token
            block[s + 1:e] = filt[:-1]                # shift -> strictly-earlier
        feats[order, col:col + N_HOURS] = block.astype(np.float32)
    return feats


# ── learned readout over the SSM states ─────────────────────────────────────

class _ReadoutMLP(nn.Module):
    """Per-token readout from a fixed-state SSM's features -> typology logit.

    Must learn to gather the relevant quantity (the current hour's habituality
    for temporal; the local arrival rate for velocity) out of the state bank;
    the answer is never handed in directly.
    """

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_readout(X_np: np.ndarray, label, *, hidden: int, epochs: int, lr: float,
                 batch: int, max_pos_weight: float, seed: int,
                 desc: str = "readout train") -> _ReadoutMLP:
    """Train the MLP readout over precomputed fixed-A SSM features.

    Shared by every fixed-state SSM slot (temporal, velocity): the states are
    precomputed, so this is a plain class-weighted minibatch logistic fit -- no
    backprop through time.
    """
    torch.manual_seed(seed)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(np.asarray(label).astype(np.float32))
    pos = float(y.sum())
    pos_w = torch.tensor([min((len(y) - pos) / max(pos, 1.0), max_pos_weight)])

    model = _ReadoutMLP(X.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(y)

    model.train()
    for _ in tqdm(range(epochs), desc=desc):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = torch.from_numpy(perm[i:i + batch])
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(X[idx]), y[idx], pos_weight=pos_w)
            loss.backward()
            opt.step()
    return model


def _score_readout(model: _ReadoutMLP, X_np: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X_np))).numpy()


class TemporalSSM:
    """Fixed-A diagonal SSM (decayed hour-histograms) + learned MLP readout."""

    def __init__(self, decays=DECAYS, hidden: int = 64, epochs: int = 12,
                 lr: float = 3e-3, batch: int = 16384, max_pos_weight: float = 50.0,
                 seed: int = 0, time_col: str = TIME_COL) -> None:
        self.decays = decays
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.max_pos_weight = max_pos_weight
        self.seed = seed
        self.time_col = time_col
        self._model: _ReadoutMLP | None = None
        self._cont_mean = np.zeros(2, dtype=np.float32)
        self._cont_std = np.ones(2, dtype=np.float32)

    def _features(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """[decayed hour-histograms | current one-hot | z(logdt) | z(logamt)]."""
        n = len(df)
        states = hour_ema_states(df, self.decays, self.time_col)

        order, starts, ends, dt = _sorted_groups(df, self.time_col)
        hour = dt.dt.hour.to_numpy()
        onehot = np.zeros((n, N_HOURS), dtype=np.float32)
        onehot[np.arange(n), hour] = 1.0

        t_ns = dt.to_numpy().astype("datetime64[ns]").astype(np.int64)[order]
        dsec = np.zeros(n, dtype=np.float64)
        dsec[1:] = (t_ns[1:] - t_ns[:-1]) / 1e9
        dsec[starts] = 0.0
        logdt = np.zeros(n, dtype=np.float32)
        logdt[order] = np.log1p(np.maximum(dsec, 0.0)).astype(np.float32)
        logamt = np.log1p(df["amt"].to_numpy()).astype(np.float32)
        cont = np.stack([logdt, logamt], axis=1)

        if fit_scaler:
            self._cont_mean = cont.mean(axis=0)
            self._cont_std = cont.std(axis=0)
            self._cont_std[self._cont_std == 0] = 1.0
        cont = (cont - self._cont_mean) / self._cont_std
        return np.concatenate([states, onehot, cont], axis=1).astype(np.float32)

    def fit(self, df: pd.DataFrame, label: np.ndarray) -> "TemporalSSM":
        self._model = _fit_readout(
            self._features(df, fit_scaler=True), label, hidden=self.hidden,
            epochs=self.epochs, lr=self.lr, batch=self.batch,
            max_pos_weight=self.max_pos_weight, seed=self.seed,
            desc="TemporalSSM train")
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Per-transaction temporal-anomaly probability, aligned to df rows."""
        return _score_readout(self._model, self._features(df, fit_scaler=False))


# ── velocity SSM: continuous-time decayed arrival rate ──────────────────────
#
# The velocity signature is one card firing many transactions in a short window
# (see ``inject_velocity``). The tabular GLM already nails it with a rolling 1h
# count (~0.88), so this is the "neural matches, does not beat, tabular" slot:
# the sequence view should *recover* the same rate signal, not exceed it.
#
# The SSM analogue of the rolling count is a leaky integrator of arrivals whose
# decay depends on the elapsed time since the last event -- an INPUT-DEPENDENT
# (selective, Mamba-like) diagonal A rather than the fixed per-token decay used
# for temporal. In a burst the inter-arrival dt -> 0, so the decay -> 1 and
# arrivals accumulate; between bursts a large dt collapses the state back toward
# zero. Read strictly before the current token, the state is a smooth, multi-
# timescale "how many transactions just preceded this one."

RATE_DECAYS = (600.0, 3600.0, 86400.0)  # timescales in seconds: ~10 min / 1 h / 1 day


def card_rate_states(df: pd.DataFrame, decays_sec=RATE_DECAYS,
                     time_col: str = TIME_COL) -> np.ndarray:
    """Per transaction, the card's CAUSAL time-decayed arrival count at each
    timescale, read BEFORE the current token.

    State recurrence per card, for timescale ``tau``:
        h_t = exp(-dt_t / tau) * h_{t-1} + 1   (dt_t = inter-arrival time)
    storing ``h`` *before* the current arrival is added, so a card's first txn
    reads 0. The exp() is the continuous-time discretisation of a diagonal SSM
    with input-dependent step; decays are precomputed (one vectorised exp), the
    short sequential recurrence stays O(n). Returns (n, len(decays)), aligned to
    df rows.
    """
    n = len(df)
    taus = np.asarray(decays_sec, dtype=np.float64)
    order, starts, ends, dt = _sorted_groups(df, time_col)
    t_sec = (dt.to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9)[order]

    dsec = np.zeros(n, dtype=np.float64)
    dsec[1:] = t_sec[1:] - t_sec[:-1]
    dsec[starts] = 0.0                           # cross-card gaps -> no decay carry-over
    decay = np.exp(-dsec[:, None] / taus)        # (n, K)

    block = np.zeros((n, len(taus)), dtype=np.float64)
    for s, e in zip(starts, ends):
        h = np.zeros(len(taus))
        for i in range(s, e):
            h = h * decay[i]
            block[i] = h
            h = h + 1.0
    feats = np.zeros((n, len(taus)), dtype=np.float32)
    feats[order] = block.astype(np.float32)
    return feats


class VelocitySSM:
    """Continuous-time diagonal SSM (decayed arrival rate) + learned MLP readout.

    Mirrors ``TemporalSSM`` -- fixed/closed-form states precomputed per card, only
    the readout learned -- but the state is the time-decayed arrival count
    (``card_rate_states``) instead of the hour histogram. The readout sees
    [log decayed-rate bank | z(log dt) | z(log amt)] and must learn to read a
    burst out of the rate bank; the rolling count is never handed in.
    """

    def __init__(self, decays_sec=RATE_DECAYS, hidden: int = 64, epochs: int = 25,
                 lr: float = 3e-3, batch: int = 16384, max_pos_weight: float = 50.0,
                 seed: int = 0, time_col: str = TIME_COL) -> None:
        self.decays_sec = decays_sec
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.max_pos_weight = max_pos_weight
        self.seed = seed
        self.time_col = time_col
        self._model: _ReadoutMLP | None = None
        self._cont_mean = np.zeros(2, dtype=np.float32)
        self._cont_std = np.ones(2, dtype=np.float32)

    def _features(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """[log decayed-rate bank | z(log dt) | z(log amt)]."""
        n = len(df)
        log_states = np.log1p(card_rate_states(df, self.decays_sec, self.time_col))

        order, starts, ends, dt = _sorted_groups(df, self.time_col)
        t_ns = dt.to_numpy().astype("datetime64[ns]").astype(np.int64)[order]
        dsec = np.zeros(n, dtype=np.float64)
        dsec[1:] = (t_ns[1:] - t_ns[:-1]) / 1e9
        dsec[starts] = 0.0
        logdt = np.zeros(n, dtype=np.float32)
        logdt[order] = np.log1p(np.maximum(dsec, 0.0)).astype(np.float32)
        logamt = np.log1p(df["amt"].to_numpy()).astype(np.float32)
        cont = np.stack([logdt, logamt], axis=1)

        if fit_scaler:
            self._cont_mean = cont.mean(axis=0)
            self._cont_std = cont.std(axis=0)
            self._cont_std[self._cont_std == 0] = 1.0
        cont = (cont - self._cont_mean) / self._cont_std
        return np.concatenate([log_states, cont], axis=1).astype(np.float32)

    def fit(self, df: pd.DataFrame, label: np.ndarray) -> "VelocitySSM":
        self._model = _fit_readout(
            self._features(df, fit_scaler=True), label, hidden=self.hidden,
            epochs=self.epochs, lr=self.lr, batch=self.batch,
            max_pos_weight=self.max_pos_weight, seed=self.seed,
            desc="VelocitySSM train")
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Per-transaction velocity-burst probability, aligned to df rows."""
        return _score_readout(self._model, self._features(df, fit_scaler=False))


# ── selective (Mamba-S6-style) temporal SSM: LEARNED input-dependent decay ──
#
# The fixed-A TemporalSSM mixes a fixed bank of decay timescales and only the
# readout is learned -- it tops out around 0.806 vs the 0.877 card_hour_rarity
# oracle. A *selective* SSM (Mamba S6) instead lets the decay be a LEARNED,
# input-dependent function of the token: a per-token step size dt_t modulates how
# much of the past hour-histogram survives, so the model chooses how much history
# to trust at each transaction. The input-dependent-dt VelocitySSM was a partial
# down-payment (the gate there is the fixed exp(-dt/tau), not learned); here the
# gate IS learned, end-to-end.
#
# Because the decay depends on the input AND ``A`` is learned, the states can no
# longer be precomputed with ``lfilter`` -- it needs a scan with grad. We run the
# scan batched across cards (pad to the longest sequence) with truncated BPTT to
# bound CPU memory/time. State per channel is a causal hour-histogram read
# strictly BEFORE the current token; the readout gathers the current hour's
# accumulated share across channels (its OWN learned habituality estimate, not
# the handed-in oracle) and maps it to a per-transaction logit.


def _selective_init_A(decays, k: int) -> np.ndarray:
    """A_raw such that ``-softplus(A_raw) == log(decay)`` (so at the init step
    dt=1 the per-token decay equals each timescale in ``decays``)."""
    dec = np.asarray(decays, dtype=np.float64)
    if len(dec) < k:  # pad by repeating the slowest decay
        dec = np.concatenate([dec, np.repeat(dec[-1], k - len(dec))])
    dec = dec[:k]
    a_pos = -np.log(dec)               # softplus output target (> 0)
    return np.log(np.expm1(a_pos)).astype(np.float32)


class _SelectiveSSMCore(nn.Module):
    """One diagonal S6 channel-bank + targeted readout, stepped one token at a time.

    Parameters learned: per-channel log-decay ``A`` (via ``A_raw``), the gate
    projection ``(gate_w, gate_b)`` that turns per-token features into the step
    ``dt_t``, and the readout MLP. The discretised decay is
    ``exp(dt_t * A)`` with ``dt_t = softplus(gate)`` and ``A < 0`` -- the
    selective (input-dependent) recurrence.
    """

    def __init__(self, n_channels: int, n_gate: int, hidden: int, decays):
        super().__init__()
        self.k = n_channels
        self.A_raw = nn.Parameter(torch.from_numpy(_selective_init_A(decays, n_channels)))
        self.gate_w = nn.Parameter(torch.zeros(n_gate))
        self.gate_b = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) ~= 1
        self.readout = _ReadoutMLP(n_channels + n_gate, hidden)

    def step(self, h: torch.Tensor, hour_t: torch.Tensor, x_t: torch.Tensor):
        """Advance one token. ``h`` (C,K,24), ``hour_t`` (C,), ``x_t`` (C,n_gate).

        Returns the updated state and the per-card logit for THIS token (read
        from the state strictly before the current token is folded in).
        """
        A = -F.softplus(self.A_raw)                          # (K,) < 0
        dt = F.softplus(x_t @ self.gate_w + self.gate_b)     # (C,) input-dependent
        a = torch.exp(dt[:, None] * A[None, :])              # (C,K) in (0,1)
        idx = hour_t.view(-1, 1, 1).expand(-1, self.k, 1)
        share = h.gather(2, idx).squeeze(2)                  # (C,K) read-before
        logit = self.readout(torch.cat([share, x_t], dim=1))
        oh = F.one_hot(hour_t, N_HOURS).to(h.dtype)          # (C,24)
        a3 = a[:, :, None]
        h = a3 * h + (1.0 - a3) * oh[:, None, :]             # normalised EMA update
        return h, logit


class SelectiveTemporalSSM:
    """Mamba-S6-style temporal SSM: learned input-dependent decay over a per-card
    hour-histogram, scored per transaction.

    Gate features per token are ``[z(log inter-arrival dt), z(log prior-count)]``
    -- "how recent" and "how much history so far", the two quantities that should
    govern how much of the card's hour profile to trust. ``A`` is initialised at
    the fixed bank's timescales (so it starts at parity with ``TemporalSSM``) and
    is free to move; the readout sees only the current hour's accumulated share
    per channel plus those gate features -- the rarity is never handed in.
    """

    def __init__(self, n_channels: int = len(DECAYS), hidden: int = 64,
                 epochs: int = 12, lr: float = 5e-3, tbptt: int = 64,
                 max_pos_weight: float = 50.0, seed: int = 0,
                 decays=DECAYS, time_col: str = TIME_COL) -> None:
        self.n_channels = n_channels
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.tbptt = tbptt
        self.max_pos_weight = max_pos_weight
        self.seed = seed
        self.decays = decays
        self.time_col = time_col
        self.core: _SelectiveSSMCore | None = None
        self._g_mean = np.zeros(2, dtype=np.float32)
        self._g_std = np.ones(2, dtype=np.float32)

    def _pack(self, df: pd.DataFrame, fit_scaler: bool):
        """Per-card, time-sorted sequences padded to a common length.

        Returns (hour, gate, mask, rowidx) over (n_cards, max_len): the per-token
        hour index, z-scored gate features [log dt, log prior-count], a validity
        mask, and the original df-row index each slot maps back to. Real tokens
        fill ``[0:len)`` and padding follows, so (read-before) padded slots never
        affect a real token's readout.
        """
        order, starts, ends, dt = _sorted_groups(df, self.time_col)
        hour_s = dt.dt.hour.to_numpy()[order].astype(np.int64)
        t_sec = (dt.to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9)[order]
        n_cards = len(starts)
        max_len = int((ends - starts).max())

        hour = np.zeros((n_cards, max_len), dtype=np.int64)
        gate = np.zeros((n_cards, max_len, 2), dtype=np.float32)
        mask = np.zeros((n_cards, max_len), dtype=bool)
        rowidx = np.full((n_cards, max_len), -1, dtype=np.int64)
        for ci, (s, e) in enumerate(zip(starts, ends)):
            li = e - s
            hour[ci, :li] = hour_s[s:e]
            d = np.zeros(li)
            d[1:] = t_sec[s + 1:e] - t_sec[s:e - 1]
            gate[ci, :li, 0] = np.log1p(np.maximum(d, 0.0))
            gate[ci, :li, 1] = np.log1p(np.arange(li))  # strictly-earlier count
            mask[ci, :li] = True
            rowidx[ci, :li] = order[s:e]

        if fit_scaler:
            flat = gate[mask]
            self._g_mean = flat.mean(axis=0)
            self._g_std = flat.std(axis=0)
            self._g_std[self._g_std == 0] = 1.0
        gate = ((gate - self._g_mean) / self._g_std).astype(np.float32)
        return hour, gate, mask, rowidx

    def fit(self, df: pd.DataFrame, label: np.ndarray) -> "SelectiveTemporalSSM":
        torch.manual_seed(self.seed)
        hour_np, gate_np, mask_np, rowidx = self._pack(df, fit_scaler=True)
        n_cards, max_len = hour_np.shape

        lab = np.asarray(label).astype(np.float32)
        y_np = np.zeros((n_cards, max_len), dtype=np.float32)
        valid = rowidx >= 0
        y_np[valid] = lab[rowidx[valid]]
        n_valid, pos = int(mask_np.sum()), float(lab.sum())
        pos_w = torch.tensor([min((n_valid - pos) / max(pos, 1.0), self.max_pos_weight)])

        hour = torch.from_numpy(hour_np)
        gate = torch.from_numpy(gate_np)
        mask = torch.from_numpy(mask_np)
        y = torch.from_numpy(y_np)

        self.core = _SelectiveSSMCore(self.n_channels, gate.shape[2], self.hidden, self.decays)
        opt = torch.optim.Adam(self.core.parameters(), lr=self.lr)
        self.core.train()
        for _ in tqdm(range(self.epochs), desc="SelectiveTemporalSSM train"):
            h = torch.zeros(n_cards, self.n_channels, N_HOURS)
            for c0 in range(0, max_len, self.tbptt):
                c1 = min(c0 + self.tbptt, max_len)
                h = h.detach()  # truncate BPTT at chunk boundaries
                logits = []
                for t in range(c0, c1):
                    h, lt = self.core.step(h, hour[:, t], gate[:, t])
                    logits.append(lt)
                mm = mask[:, c0:c1]
                if not mm.any():
                    continue
                logit = torch.stack(logits, dim=1)
                loss = F.binary_cross_entropy_with_logits(
                    logit[mm], y[:, c0:c1][mm], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Per-transaction temporal-anomaly probability, aligned to df rows."""
        hour_np, gate_np, _, rowidx = self._pack(df, fit_scaler=False)
        n_cards, max_len = hour_np.shape
        hour = torch.from_numpy(hour_np)
        gate = torch.from_numpy(gate_np)
        self.core.eval()
        probs = np.zeros((n_cards, max_len), dtype=np.float32)
        with torch.no_grad():
            h = torch.zeros(n_cards, self.n_channels, N_HOURS)
            for t in range(max_len):
                h, lt = self.core.step(h, hour[:, t], gate[:, t])
                probs[:, t] = torch.sigmoid(lt).numpy()
        out = np.zeros(len(df), dtype=np.float32)
        valid = rowidx >= 0
        out[rowidx[valid]] = probs[valid]
        return out
