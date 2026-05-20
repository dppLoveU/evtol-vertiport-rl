"""Stage 4B-2 tests for ``src/models/ema.py``.

Pure CPU; uses a toy nn.Module so the tests stay fast and don't touch
the real OD tensor or the smoke U-Net.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.ema import EMA


class _ToyModel(nn.Module):
    """Tiny model with both Linear and Conv params + a BatchNorm buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.conv = nn.Conv2d(2, 2, kernel_size=3)
        self.bn = nn.BatchNorm1d(4)


def _fresh_pair() -> tuple[_ToyModel, EMA]:
    torch.manual_seed(0)
    m = _ToyModel()
    ema = EMA(m, decay=0.9)
    return m, ema


# --- construction ---------------------------------------------------------


def test_ema_buffer_count_matches_trainable_params() -> None:
    m, ema = _fresh_pair()
    n_params = sum(1 for p in m.parameters() if p.requires_grad)
    n_bufs = sum(1 for _ in ema.buffers())
    assert n_bufs == n_params


def test_ema_init_buffers_equal_model_params() -> None:
    m, ema = _fresh_pair()
    m_copy = _ToyModel()
    ema.copy_to(m_copy)
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m_copy.named_parameters()):
        if not p1.requires_grad:
            continue
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_ema_rejects_invalid_decay() -> None:
    m = _ToyModel()
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            EMA(m, decay=bad)


# --- update ---------------------------------------------------------------


def test_ema_update_changes_buffers() -> None:
    m, ema = _fresh_pair()
    initial = {n: b.clone() for n, b in ema.named_buffers()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p))
    ema.update(m)
    changed = sum(
        1 for n, b in ema.named_buffers() if not torch.allclose(b, initial[n])
    )
    assert changed > 0


def test_ema_update_decay_correctness() -> None:
    m, ema = _fresh_pair()  # decay=0.9
    # Snapshot initial EMA via copy_to a fresh model (avoids using EMA internals).
    m_init = _ToyModel()
    ema.copy_to(m_init)
    init_params = {n: p.detach().clone() for n, p in m_init.named_parameters()}
    # Set model params to zero so the update math is easy.
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    ema.update(m)
    # EMA should now be 0.9 * init + 0.1 * 0 = 0.9 * init.
    m_check = _ToyModel()
    ema.copy_to(m_check)
    for n, p in m_check.named_parameters():
        if not p.requires_grad:
            continue
        expected = 0.9 * init_params[n]
        assert torch.allclose(p, expected, atol=1e-6), n


# --- copy_to / store / restore -------------------------------------------


def test_ema_copy_to_overrides_model() -> None:
    m, ema = _fresh_pair()
    with torch.no_grad():
        for b in ema.buffers():
            b.add_(1.0)
    target = _ToyModel()
    ema.copy_to(target)
    # Each target param now matches the corresponding EMA buffer.
    name_map = dict(m.named_parameters())  # to get the buffer-name mapping reuse below
    for name, p in target.named_parameters():
        if not p.requires_grad:
            continue
        buf = getattr(ema, name.replace(".", "_"))
        assert torch.allclose(p, buf)


def test_ema_store_restore_roundtrip() -> None:
    m, ema = _fresh_pair()
    ema.store(m)
    snapshot = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    # After mutation the model differs from the snapshot.
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        assert not torch.allclose(p, snapshot[n])
    ema.restore(m)
    for n, p in m.named_parameters():
        assert torch.allclose(p, snapshot[n])


def test_ema_restore_before_store_raises() -> None:
    m, ema = _fresh_pair()
    with pytest.raises(RuntimeError):
        ema.restore(m)


# --- state_dict --------------------------------------------------------


def test_ema_state_dict_roundtrip() -> None:
    m1, ema1 = _fresh_pair()
    with torch.no_grad():
        for b in ema1.buffers():
            b.add_(0.5)
    sd = ema1.state_dict()

    m2 = _ToyModel()  # fresh weights, irrelevant
    ema2 = EMA(m2, decay=0.9)
    ema2.load_state_dict(sd)
    for (n1, b1), (n2, b2) in zip(ema1.named_buffers(), ema2.named_buffers()):
        assert n1 == n2
        assert torch.allclose(b1, b2)


def test_ema_to_device_moves_buffers() -> None:
    m, ema = _fresh_pair()
    ema = ema.to("cpu")
    for n, b in ema.named_buffers():
        assert b.device.type == "cpu", f"{n} on {b.device}"
