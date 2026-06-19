"""State-space lineage + sequence baselines on a shared per-card interface.

Every model here ingests the SAME per-token stream (one sequence per card,
time-ordered) and emits a PER-TRANSACTION logit, so the bake-off compares
*representations* on equal footing, not feature engineering. The contract mirrors
``TemporalSSM``/``VelocitySSM`` in ``ssm.py``: ``fit(df, label)`` /
``score(df) -> per-row probability aligned to df``.

The unifying primitive is ``associative_scan`` -- the work-efficient parallel
prefix scan for a first-order linear recurrence ``h_t = a_t * h_{t-1} + b_t``.
Every diagonal SSM here (LRU, S5, DSS, Mamba-S6) reduces to "compute per-step
``(a_t, b_t)``, scan, read out", differing only in how ``a_t`` is parameterised:

  S4D / DSS   fixed diagonal A, time-invariant a (the ssm.py fixed bank is the
              precomputed-state cousin of this)
  LRU         fixed complex diagonal, exponential (stable) parameterisation
  S5          fixed (HiPPO-init) diagonal A, ZOH/bilinear discretisation, a = disc(A)
  Mamba-S6    input-dependent a_t = exp(dt(x_t) * A) -- the selective recurrence

Baselines (GRU, LSTM, TCN, causal Transformer) take the same packed sequences so
the contrast is clean: recurrence vs convolution vs attention vs state-space.

CPU-first: the scan is O(T log T) and vectorised; sequences are batched across
cards. ``max_seq`` caps per-card length (keeps all positives + most-recent legit)
to bound memory for the quadratic-attention Transformer.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ssm import _sorted_groups  # per-card time-ordered grouping

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

TIME_COL = "trans_date_trans_time"
N_HOURS = 24


# ── parallel associative scan ───────────────────────────────────────────────

def associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of ``h_t = a_t * h_{t-1} + b_t`` over dim=1 (h_{-1}=0).

    Hillis-Steele: each pass composes the affine maps ``f_t(h) = a_t h + b_t``
    with their step-distant predecessor, so after ceil(log2 T) passes ``b`` holds
    the full prefix. Supports real or complex tensors of shape (B, T, D); the
    work is O(T log T) but every element of a pass is independent (parallel).
    """
    T = a.shape[1]
    step = 1
    while step < T:
        a_l, b_l = a[:, :-step], b[:, :-step]      # earlier element of each pair
        a_r, b_r = a[:, step:], b[:, step:]        # later element
        a = torch.cat([a[:, :step], a_r * a_l], dim=1)
        b = torch.cat([b[:, :step], a_r * b_l + b_r], dim=1)
        step *= 2
    return b


# ── per-card sequence packing & per-token features ──────────────────────────

def _aux(df: pd.DataFrame, time_col: str):
    """Per-row (aligned to df): log inter-arrival dt, log strictly-earlier count,
    hour-of-day, plus the (order, starts, ends) card grouping."""
    n = len(df)
    order, starts, ends, dt = _sorted_groups(df, time_col)
    t_sec = (dt.to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9)[order]
    dsec = np.zeros(n)
    dsec[1:] = t_sec[1:] - t_sec[:-1]
    dsec[starts] = 0.0
    logdt = np.zeros(n, dtype=np.float32)
    logdt[order] = np.log1p(np.maximum(dsec, 0.0)).astype(np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    for s, e in zip(starts, ends):
        cnt[s:e] = np.arange(e - s)
    logcnt = np.zeros(n, dtype=np.float32)
    logcnt[order] = np.log1p(cnt).astype(np.float32)
    hour = dt.dt.hour.to_numpy().astype(np.int64)
    return logdt, logcnt, hour, (order, starts, ends)


def token_features(df, slot, scaler=None, time_col=TIME_COL):
    """Per-row float feature matrix for a slot, aligned to df.

    temporal: [one-hot hour (24) | z(log dt) | z(log amt)]  -- the model must
              learn "rare hour FOR THIS CARD" from the card's own stream.
    velocity: [z(log dt) | z(log amt) | z(log prior-count)] -- the model must
              learn the burst (short dt, rising count) without a handed-in rate.
    Returns (feats, scaler, grouping); pass the returned scaler back at score time.
    """
    logdt, logcnt, hour, grouping = _aux(df, time_col)
    logamt = np.log1p(df["amt"].to_numpy()).astype(np.float32)
    if slot == "temporal":
        cont = np.stack([logdt, logamt], axis=1)
    elif slot == "velocity":
        cont = np.stack([logdt, logamt, logcnt], axis=1)
    else:
        raise ValueError(f"unknown slot {slot!r}")

    if scaler is None:
        mean, std = cont.mean(0), cont.std(0)
        std[std == 0] = 1.0
        scaler = (mean.astype(np.float32), std.astype(np.float32))
    cont = ((cont - scaler[0]) / scaler[1]).astype(np.float32)

    if slot == "temporal":
        oh = np.zeros((len(df), N_HOURS), dtype=np.float32)
        oh[np.arange(len(df)), hour] = 1.0
        feats = np.concatenate([oh, cont], axis=1)
    else:
        feats = cont
    return feats.astype(np.float32), scaler, grouping


def pack(feats, grouping, label=None, max_seq=None, rng=None):
    """Pad per-card sequences to (n_cards, L, d). Returns X, mask, rowidx[, y].

    With ``max_seq`` a long card keeps ALL positive-label tokens plus the most
    recent legit tokens up to the budget (so injected rows are never dropped from
    training); order is preserved. ``rowidx`` maps each slot back to its df row.
    """
    order, starts, ends = grouping
    fs = feats[order]
    lab_s = None if label is None else np.asarray(label)[order]

    kept = []
    for s, e in zip(starts, ends):
        idx = np.arange(s, e)
        if max_seq is not None and len(idx) > max_seq:
            if lab_s is not None:
                pos = idx[lab_s[idx] > 0]
                budget = max(max_seq - len(pos), 0)
                recent = idx[-budget:] if budget else idx[:0]
                idx = np.union1d(pos, recent)  # sorted -> causal order preserved
            else:
                idx = idx[-max_seq:]
        kept.append(idx)

    n_cards = len(kept)
    L = max(len(k) for k in kept)
    d = feats.shape[1]
    X = np.zeros((n_cards, L, d), dtype=np.float32)
    mask = np.zeros((n_cards, L), dtype=bool)
    rowidx = np.full((n_cards, L), -1, dtype=np.int64)
    y = None if label is None else np.zeros((n_cards, L), dtype=np.float32)
    for ci, idx in enumerate(kept):
        li = len(idx)
        X[ci, :li] = fs[idx]
        mask[ci, :li] = True
        rowidx[ci, :li] = order[idx]
        if y is not None:
            y[ci, :li] = lab_s[idx]
    return (X, mask, rowidx) if y is None else (X, mask, rowidx, y)


# ── architectures: each maps packed X (B,L,d) -> per-token logit (B,L) ───────

class _Head(nn.Module):
    """Shared 2-layer readout from a per-token feature vector to one logit."""

    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _hippo_diag(n: int) -> np.ndarray:
    """Diagonal HiPPO-LegS-style init: real part -1/2, imaginary parts the
    LegS skew eigenvalue spacing ~ i*pi*k. Returns complex A (Re<0)."""
    k = np.arange(n)
    return (-0.5 + 1j * math.pi * k).astype(np.complex64)


class _DiagSSM(nn.Module):
    """Diagonal complex/real SSM via the parallel scan -- the shared engine for
    LRU / S5 / DSS / Mamba-S6. ``selective`` makes the decay input-dependent
    (the S6 recurrence); ``hippo`` / ``disc`` toggle the init and discretisation
    ablations. Readout sees [Re h | Im h] so C is folded into the head."""

    def __init__(self, d_in, n_state=32, hidden=64, *, complex_state=True,
                 hippo=True, selective=False, disc="zoh", learn_dt=True):
        super().__init__()
        self.n = n_state
        self.complex_state = complex_state
        self.selective = selective
        self.disc = disc

        if complex_state:
            A0 = _hippo_diag(n_state) if hippo else (
                -np.exp(np.random.randn(n_state)) + 1j * np.random.randn(n_state)
            ).astype(np.complex64)
            self.A_re = nn.Parameter(torch.tensor(np.log(-A0.real)))   # Re A = -exp()
            self.A_im = nn.Parameter(torch.tensor(A0.imag.copy()))
        else:  # real diagonal (Mamba/SSD style)
            self.A_re = nn.Parameter(torch.rand(n_state) * 0.5 + 0.5)  # -> A = -A_re
            self.A_im = None

        self.B = nn.Linear(d_in, n_state)
        self.log_dt = nn.Parameter(torch.zeros(n_state)) if learn_dt else None
        if selective:
            self.dt_gate = nn.Linear(d_in, n_state)
        feat_dim = 2 * n_state if complex_state else n_state
        self.head = _Head(feat_dim + d_in, hidden)

    def _A(self):
        if self.complex_state:
            return torch.complex(-torch.exp(self.A_re), self.A_im)
        return -F.softplus(self.A_re)

    def forward(self, x):                                  # x: (B,L,d)
        A = self._A()
        b_in = self.B(x)                                  # (B,L,N)
        if self.selective:
            dt = F.softplus(self.dt_gate(x))              # (B,L,N) input-dependent
        else:
            dt = F.softplus(self.log_dt) if self.log_dt is not None else torch.ones(self.n)
            dt = dt.view(1, 1, -1).expand_as(b_in)
        if self.complex_state:
            dtA = dt.to(A.dtype) * A.view(1, 1, -1)
            b_in = b_in.to(A.dtype)
        else:
            dtA = dt * A.view(1, 1, -1)
        if self.disc == "bilinear":                       # Tukey / bilinear
            a = (1 + dtA / 2) / (1 - dtA / 2)
            b = (dt.to(b_in.dtype) / (1 - dtA / 2)) * b_in
        else:                                             # zero-order hold
            a = torch.exp(dtA)
            b = ((a - 1) / A.view(1, 1, -1)) * b_in        # expm1 has no complex impl
        h = associative_scan(a, b)                        # (B,L,N)
        feat = torch.cat([h.real, h.imag], -1) if self.complex_state else h
        return self.head(torch.cat([feat.to(x.dtype), x], -1))


class GRUSeq(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0]).squeeze(-1)


class LSTMSeq(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0]).squeeze(-1)


class TCNSeq(nn.Module):
    """Dilated CAUSAL temporal conv net -- the non-recurrent sequence baseline."""

    def __init__(self, d_in, hidden=64, k=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = d_in
        for d in dilations:
            self.blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(c_in, hidden, k, dilation=d, padding=(k - 1) * d),
                "res": nn.Conv1d(c_in, hidden, 1) if c_in != hidden else None,
            }))
            c_in = hidden
        self.crop = [(k - 1) * d for d in dilations]
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = x.transpose(1, 2)                              # (B,d,L)
        for blk, crop in zip(self.blocks, self.crop):
            y = blk["conv"](h)[..., :-crop] if crop else blk["conv"](h)
            r = h if blk["res"] is None else blk["res"](h)
            h = F.gelu(y) + r
        return self.head(h.transpose(1, 2)).squeeze(-1)


class TransformerSeq(nn.Module):
    """Causal self-attention encoder -- the attention contrast to state-space."""

    def __init__(self, d_in, d_model=64, nhead=4, layers=2, ff=128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.d_model = d_model
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True,
                                         activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def _pe(self, L, device):
        pos = torch.arange(L, device=device).unsqueeze(1)
        i = torch.arange(0, self.d_model, 2, device=device)
        ang = pos / (10000 ** (i / self.d_model))
        pe = torch.zeros(L, self.d_model, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
        return pe

    def forward(self, x, pad_mask=None):
        L = x.shape[1]
        h = self.proj(x) + self._pe(L, x.device)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        h = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.head(h).squeeze(-1)


ARCHITECTURES = {
    "gru": lambda d, n, h: GRUSeq(d, h),
    "lstm": lambda d, n, h: LSTMSeq(d, h),
    "tcn": lambda d, n, h: TCNSeq(d, h),
    "transformer": lambda d, n, h: TransformerSeq(d, d_model=h),
    "lru": lambda d, n, h: _DiagSSM(d, n, h, complex_state=True, hippo=False,
                                    selective=False),
    "s5": lambda d, n, h: _DiagSSM(d, n, h, complex_state=True, hippo=True,
                                   selective=False),
    "dss": lambda d, n, h: _DiagSSM(d, n, h, complex_state=True, hippo=True,
                                    selective=False, learn_dt=False),
    "mamba_s6": lambda d, n, h: _DiagSSM(d, n, h, complex_state=False, hippo=False,
                                         selective=True),
}


# ── unified fit/score wrapper ───────────────────────────────────────────────

class SequenceModel:
    """Wrap any architecture into the ``fit(df, label)`` / ``score(df)`` contract.

    Trains masked, class-weighted per-token BCE batched across cards (full BPTT;
    the scan/conv/attention are all O(T) or O(T log T)). ``score`` returns a
    per-transaction probability aligned to df rows.
    """

    def __init__(self, arch, slot, *, n_state=32, hidden=64, epochs=15, lr=3e-3,
                 batch_cards=256, max_seq=512, max_pos_weight=50.0, seed=0,
                 time_col=TIME_COL, **arch_kw):
        self.arch = arch
        self.slot = slot
        self.n_state = n_state
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_cards = batch_cards
        self.max_seq = max_seq
        self.max_pos_weight = max_pos_weight
        self.seed = seed
        self.time_col = time_col
        self.arch_kw = arch_kw
        self.model: nn.Module | None = None
        self._scaler = None
        self.train_seconds = 0.0

    def _build(self, d_in):
        if self.arch == "diag":   # parametrised diagonal SSM for ablations
            return _DiagSSM(d_in, self.n_state, self.hidden, **self.arch_kw)
        if self.arch in ARCHITECTURES:
            return ARCHITECTURES[self.arch](d_in, self.n_state, self.hidden)
        raise ValueError(f"unknown arch {self.arch!r}")

    def fit(self, df, label):
        torch.manual_seed(self.seed)
        feats, self._scaler, grouping = token_features(df, self.slot, None, self.time_col)
        X, mask, _, y = pack(feats, grouping, label=label, max_seq=self.max_seq)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        mt = torch.from_numpy(mask)
        pos = float(y[mask].sum())
        pos_w = torch.tensor([min((mask.sum() - pos) / max(pos, 1.0), self.max_pos_weight)])

        self.model = self._build(X.shape[2])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        nc = X.shape[0]
        t0 = time.perf_counter()
        self.model.train()
        for _ in tqdm(range(self.epochs), desc=f"{self.arch}/{self.slot}"):
            perm = rng.permutation(nc)
            for i in range(0, nc, self.batch_cards):
                sel = perm[i:i + self.batch_cards]
                bx, by, bm = Xt[sel], yt[sel], mt[sel]
                logit = self._forward(bx, bm)
                if not bm.any():
                    continue
                loss = F.binary_cross_entropy_with_logits(logit[bm], by[bm], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.train_seconds = time.perf_counter() - t0
        return self

    def _forward(self, bx, bm):
        if self.arch == "transformer":
            return self.model(bx, pad_mask=~bm)
        return self.model(bx)

    def score(self, df):
        feats, _, grouping = token_features(df, self.slot, self._scaler, self.time_col)
        X, mask, rowidx = pack(feats, grouping, label=None, max_seq=None)
        out = np.zeros(len(df), dtype=np.float32)
        # O(T) models score full sequences; the quadratic Transformer is scored in
        # non-overlapping causal blocks of max_seq to bound attention memory.
        tchunk = self.max_seq if self.arch == "transformer" else X.shape[1]
        self.model.eval()
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_cards):
                sl = slice(i, i + self.batch_cards)
                Xb, mb, rb = X[sl], mask[sl], rowidx[sl]
                for c0 in range(0, Xb.shape[1], tchunk):
                    xs = torch.from_numpy(Xb[:, c0:c0 + tchunk])
                    ms = torch.from_numpy(mb[:, c0:c0 + tchunk])
                    if not ms.any():
                        continue
                    probs = torch.sigmoid(self._forward(xs, ms)).numpy()
                    rr, mm = rb[:, c0:c0 + tchunk], mb[:, c0:c0 + tchunk]
                    out[rr[mm]] = probs[mm]
        return out

    def n_params(self):
        return int(sum(p.numel() for p in self.model.parameters()))
