"""Parallel scan correctness + a fit/score smoke for every architecture."""
import numpy as np
import pytest
import torch

from src.models.sequence import ARCHITECTURES, SequenceModel, associative_scan


def _sequential(a, b):
    """Reference: h_t = a_t * h_{t-1} + b_t, looped."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("T", [1, 2, 5, 16, 17])
def test_associative_scan_matches_sequential(dtype, T):
    torch.manual_seed(0)
    B, D = 3, 4
    if dtype == torch.complex64:
        a = (0.6 * torch.randn(B, T, D) + 0.6j * torch.randn(B, T, D))
        b = (torch.randn(B, T, D) + 1j * torch.randn(B, T, D))
    else:
        a = 0.5 * torch.rand(B, T, D)
        b = torch.randn(B, T, D)
    got = associative_scan(a.clone(), b.clone())
    assert torch.allclose(got, _sequential(a, b), atol=1e-4)


def test_scan_is_differentiable():
    a = torch.full((1, 8, 1), 0.5, requires_grad=True)
    b = torch.ones(1, 8, 1, requires_grad=True)
    associative_scan(a, b).sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
@pytest.mark.parametrize("slot", ["temporal", "velocity"])
def test_architecture_fit_score_smoke(legit_df, arch, slot):
    rng = np.random.default_rng(0)
    label = (rng.uniform(size=len(legit_df)) < 0.1).astype(int)
    m = SequenceModel(arch, slot, epochs=2, n_state=8, hidden=16,
                      max_seq=64, seed=0).fit(legit_df, label)
    s = m.score(legit_df)
    assert s.shape == (len(legit_df),)
    assert ((s >= 0) & (s <= 1)).all()
    assert m.n_params() > 0
