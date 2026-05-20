"""Stage-4A: OD-slice dataset for the diffusion model.

``ODDataset`` turns the Stage-3 OD tensor ``od_evtol.npy``
(``[T, |Z|, |Z|]`` int32) into normalized, padded OD-slice "images"
ready for a diffusion U-Net, together with a per-slice time condition
(hour-of-day, day-of-week, is-weekend).

This module is pure NumPy -- it has no torch dependency (Stage 4A does
not train a model). It already exposes ``__len__`` / ``__getitem__`` so
it works as a ``torch.utils.data.Dataset`` by duck typing once torch is
added in Stage 4B.

Normalization schemes (see ``configs/diffusion.yaml::data.scheme``):

  * ``global_clip`` (default; the Stage 4B baseline). Pipeline:

      1. ``log1p(count)`` -- compress the long tail.
      2. standardize with a *global scalar* ``mu`` / ``sigma`` computed
         over the TRAIN split (not per-pixel).
      3. clip to ``[-clip_val, +clip_val]``.
      4. scale to ``[-1, 1]`` (divide by ``clip_val``).

  * ``zero_pinned_nonzero`` (Stage 4B-5B). Pipeline:

      1. raw count == 0  ->  normalized = -1.0 (pinned).
      2. raw count >  0  ->  ``log1p(count)``, standardize with
         ``mu_nz`` / ``sigma_nz`` computed over only the *nonzero*
         entries of the TRAIN-split slots, clip to
         ``[-clip_val, +clip_val]``, scale to ``[-1, 1]``.

    Inverse: ``x < -0.5`` is rounded back to exact 0; everything else
    is inverted via ``expm1`` and clipped to non-negative.

    Motivation: the ``global_clip`` scheme collapses the zero / nonzero
    gap to ~0.25 on the 0.117%-sparse eVTOL tensor (see
    ``docs/decisions.md`` 2026-05-20). ``zero_pinned_nonzero`` restores
    a ~1.0-wide gap so noise-MSE penalises misplaced nonzero pixels.

Padding: ``|Z|`` is zero-padded at the bottom/right to ``pad_size``
(``ceil(|Z| / pad_multiple) * pad_multiple``) so each spatial dim is
divisible by ``2^depth`` for the U-Net. Padding is applied to the raw
counts *before* normalization, so the padded region carries the same
normalized value as a real zero-count OD pair.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Default train/val/test split over the 11-day window (T = 528, 48 slots
# per 30-min-binned day). Contiguous-day ranges [start_day, end_day);
# mirrors configs/diffusion.yaml::data.split.
DEFAULT_SPLIT: dict[str, list[int]] = {
    "train_days": [0, 9],
    "val_days": [9, 10],
    "test_days": [10, 11],
}

_SPLIT_KEYS = {"train": "train_days", "val": "val_days", "test": "test_days"}

# Supported normalization schemes (see module docstring).
VALID_SCHEMES = ("global_clip", "zero_pinned_nonzero")
# Inverse-transform threshold for zero_pinned_nonzero: any normalized
# value strictly below this is restored to exact 0. Calibrated so it
# sits well between the pinned zero (-1.0) and the smallest nonzero
# image (count=1 maps to ~-0.013 on realistic eVTOL stats with
# clip_val=20). The threshold is a constant, not a stat -- if a future
# scheme variant needs a different value, expose it then.
_ZPIN_ZERO_THRESHOLD = -0.5


# --- padding ---------------------------------------------------------------


def next_pad_size(n: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``n``."""
    if n <= 0 or multiple <= 0:
        raise ValueError(f"n and multiple must be positive (got n={n}, multiple={multiple})")
    return math.ceil(n / multiple) * multiple


def pad_hw(x: np.ndarray, pad_size: int) -> np.ndarray:
    """Zero-pad the last two dims of ``x`` to ``(pad_size, pad_size)``.

    Padding is added at the bottom/right; the original content keeps its
    top-left position so ``unpad`` recovers it exactly.
    """
    h, w = x.shape[-2], x.shape[-1]
    if h > pad_size or w > pad_size:
        raise ValueError(f"cannot pad ({h}, {w}) down to pad_size={pad_size}")
    widths = [(0, 0)] * (x.ndim - 2) + [(0, pad_size - h), (0, pad_size - w)]
    return np.pad(x, widths, mode="constant", constant_values=0)


def unpad(x: np.ndarray, n_zones: int) -> np.ndarray:
    """Slice the last two dims of ``x`` back to ``(n_zones, n_zones)``."""
    if x.shape[-1] < n_zones or x.shape[-2] < n_zones:
        raise ValueError(f"array {x.shape} is smaller than n_zones={n_zones}")
    return x[..., :n_zones, :n_zones]


# --- normalization ---------------------------------------------------------


def compute_norm_stats(
    od: np.ndarray, train_slots: range | list[int], clip_val: float
) -> dict[str, Any]:
    """Compute global scalar ``mu`` / ``sigma`` of ``log1p(count)``.

    ``global_clip`` scheme. Statistics are accumulated over every entry
    of every TRAIN-split slot (not per-pixel). ``od`` may be an
    ``np.memmap``; slots are read one at a time so the full tensor is
    never resident.
    """
    total = 0.0
    total_sq = 0.0
    count = 0
    for s in train_slots:
        v = np.log1p(np.asarray(od[s], dtype=np.float64))
        total += float(v.sum())
        total_sq += float(np.square(v).sum())
        count += int(v.size)
    if count == 0:
        raise ValueError("train_slots is empty -- cannot compute norm stats")
    mu = total / count
    var = max(total_sq / count - mu * mu, 0.0)
    sigma = math.sqrt(var)
    if sigma == 0.0:
        # Degenerate (all-equal) train data -- avoid divide-by-zero.
        sigma = 1.0
    return {
        "mu": mu,
        "sigma": sigma,
        "clip_val": float(clip_val),
        "scheme": "global_clip",
    }


def compute_norm_stats_nonzero(
    od: np.ndarray, train_slots: range | list[int], clip_val: float
) -> dict[str, Any]:
    """Compute ``mu_nz`` / ``sigma_nz`` of ``log1p(count)`` over nonzero entries.

    ``zero_pinned_nonzero`` scheme. Statistics are accumulated only over
    entries where the raw count is strictly positive; zero-count
    entries are excluded (they are pinned to -1.0 by the forward
    transform and do not contribute to ``mu`` / ``sigma``).
    """
    total = 0.0
    total_sq = 0.0
    count = 0
    for s in train_slots:
        sl = np.asarray(od[s], dtype=np.float64)
        nz = sl[sl > 0]
        if nz.size == 0:
            continue
        v = np.log1p(nz)
        total += float(v.sum())
        total_sq += float(np.square(v).sum())
        count += int(v.size)
    if count == 0:
        raise ValueError(
            "no nonzero entries in train_slots -- cannot compute "
            "zero_pinned_nonzero stats"
        )
    mu_nz = total / count
    var = max(total_sq / count - mu_nz * mu_nz, 0.0)
    sigma_nz = math.sqrt(var)
    if sigma_nz == 0.0:
        sigma_nz = 1.0
    return {
        "mu_nz": mu_nz,
        "sigma_nz": sigma_nz,
        "clip_val": float(clip_val),
        "scheme": "zero_pinned_nonzero",
    }


def apply_norm(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """``global_clip`` forward: raw OD counts -> normalized ``[-1, 1]``."""
    cv = stats["clip_val"]
    v = np.log1p(np.asarray(x, dtype=np.float64))
    v = (v - stats["mu"]) / stats["sigma"]
    v = np.clip(v, -cv, cv) / cv
    return v.astype(np.float32)


def inverse_norm(x_norm: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """``global_clip`` inverse: normalized ``[-1, 1]`` -> non-negative counts.

    The forward clip means values whose standardized ``log1p`` exceeded
    ``clip_val`` are not recoverable; everything else round-trips.
    """
    cv = stats["clip_val"]
    v = np.asarray(x_norm, dtype=np.float64) * cv
    v = v * stats["sigma"] + stats["mu"]
    v = np.expm1(v)
    return np.maximum(v, 0.0)


def apply_norm_zero_pinned(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """``zero_pinned_nonzero`` forward: raw counts -> normalized ``[-1, 1]``.

    Zero entries are pinned to exactly -1.0. Nonzero entries pass
    through ``log1p`` -> standardize with ``mu_nz`` / ``sigma_nz`` ->
    clip -> divide by ``clip_val``.
    """
    cv = stats["clip_val"]
    mu = stats["mu_nz"]
    sigma = stats["sigma_nz"]
    arr = np.asarray(x, dtype=np.float64)
    mask = arr > 0
    v = np.log1p(arr)
    v_std = (v - mu) / sigma
    v_clip = np.clip(v_std, -cv, cv) / cv
    out = np.where(mask, v_clip, -1.0)
    return out.astype(np.float32)


def inverse_norm_zero_pinned(
    x_norm: np.ndarray, stats: dict[str, Any]
) -> np.ndarray:
    """``zero_pinned_nonzero`` inverse: normalized -> non-negative counts.

    ``x < _ZPIN_ZERO_THRESHOLD`` (-0.5) is restored to exact 0; every
    other value is inverted via ``expm1`` of the unstandardized
    ``log1p`` and clipped to non-negative.
    """
    cv = stats["clip_val"]
    mu = stats["mu_nz"]
    sigma = stats["sigma_nz"]
    arr = np.asarray(x_norm, dtype=np.float64)
    is_zero = arr < _ZPIN_ZERO_THRESHOLD
    v = arr * cv * sigma + mu
    v = np.expm1(v)
    v = np.maximum(v, 0.0)
    return np.where(is_zero, 0.0, v)


def save_norm_stats(stats: dict[str, Any], path: str | Path) -> None:
    """Write norm stats to JSON. Scheme tag (if present) is preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)


def load_norm_stats(path: str | Path) -> dict[str, Any]:
    """Read norm stats from a JSON file written by ``save_norm_stats``.

    Loads any of the known scheme keys (``mu``, ``sigma``, ``mu_nz``,
    ``sigma_nz``, ``clip_val``) plus the optional ``scheme`` tag. Keys
    that are not present in the source JSON are not added.
    """
    with open(path) as fh:
        raw = json.load(fh)
    out: dict[str, Any] = {}
    for k in ("mu", "sigma", "mu_nz", "sigma_nz", "clip_val"):
        if k in raw:
            out[k] = float(raw[k])
    if "scheme" in raw:
        out["scheme"] = str(raw["scheme"])
    return out


# --- conditioning ----------------------------------------------------------


def slot_to_condition(
    slot: int, start_dt: _dt.datetime, slots_per_day: int, bin_min: int
) -> dict[str, int]:
    """Map a time slot to its ``(hour, day_of_week, is_weekend)`` condition.

    ``day_of_week`` is 0=Monday .. 6=Sunday; ``is_weekend`` is 1 for
    Saturday/Sunday.
    """
    minutes = slot * bin_min
    hour = (minutes // 60) % 24
    day_index = slot // slots_per_day
    dow = (start_dt.weekday() + day_index) % 7
    return {"hour": int(hour), "day_of_week": int(dow), "is_weekend": int(dow >= 5)}


# --- dataset ---------------------------------------------------------------


class ODDataset:
    """OD-slice dataset for the Stage-4 diffusion model.

    Parameters
    ----------
    od_path : path to the Stage-3 OD tensor ``[T, |Z|, |Z|]`` int .npy.
    meta_path : path to ``od_meta.json`` (``T``, ``n_zones``,
        ``start_datetime``, ``time_bin_min``).
    split : one of ``"train"`` / ``"val"`` / ``"test"``.
    window : number of consecutive slots per sample (channels). Default 1.
    pad_multiple : spatial dims are padded to a multiple of this.
    clip_val : symmetric clip applied after standardization.
    scheme : ``"global_clip"`` (default; baseline) or
        ``"zero_pinned_nonzero"`` (Stage 4B-5B). See module docstring.
    norm_stats : precomputed stats dict. For ``global_clip`` shape is
        ``{mu, sigma, clip_val}`` (plus optional ``scheme`` tag); for
        ``zero_pinned_nonzero`` shape is ``{mu_nz, sigma_nz, clip_val,
        scheme}``. When ``None`` the stats are computed from the TRAIN
        split (regardless of ``split``) so val/test never leak their
        own statistics. If a ``scheme`` tag is present in
        ``norm_stats`` it must match the ``scheme`` argument.
    split_cfg : day-range split config; defaults to ``DEFAULT_SPLIT``.

    A sample is ``(x, condition)`` where ``x`` is a normalized, padded
    ``[window, pad_size, pad_size]`` float32 array and ``condition`` is a
    dict with ``hour`` / ``day_of_week`` / ``is_weekend`` for the
    window's first slot.
    """

    def __init__(
        self,
        od_path: str | Path,
        meta_path: str | Path,
        split: str,
        *,
        window: int = 1,
        pad_multiple: int = 16,
        clip_val: float = 3.0,
        scheme: str = "global_clip",
        norm_stats: dict[str, Any] | None = None,
        split_cfg: dict[str, list[int]] | None = None,
    ) -> None:
        if split not in _SPLIT_KEYS:
            raise ValueError(f"split must be one of {sorted(_SPLIT_KEYS)}, got {split!r}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if scheme not in VALID_SCHEMES:
            raise ValueError(
                f"scheme must be one of {VALID_SCHEMES}, got {scheme!r}"
            )

        with open(meta_path) as fh:
            meta = json.load(fh)
        self.n_slots: int = int(meta["T"])
        self.n_zones: int = int(meta["n_zones"])
        self.bin_min: int = int(meta.get("time_bin_min", 30))
        self.start_dt: _dt.datetime = _dt.datetime.fromisoformat(meta["start_datetime"])
        self.slots_per_day: int = 24 * 60 // self.bin_min

        self.split = split
        self.window = window
        self.pad_multiple = pad_multiple
        self.pad_size: int = next_pad_size(self.n_zones, pad_multiple)
        self.scheme = scheme
        self._split_cfg = split_cfg if split_cfg is not None else DEFAULT_SPLIT

        # Memory-map the OD tensor -- the 593 MB array is never resident.
        self._od: np.ndarray = np.load(od_path, mmap_mode="r")
        if self._od.shape != (self.n_slots, self.n_zones, self.n_zones):
            raise ValueError(
                f"od tensor shape {self._od.shape} does not match meta "
                f"(T={self.n_slots}, n_zones={self.n_zones})"
            )

        self._slots: list[int] = self._split_slots(split)
        # Valid window start indices: the window must stay inside the
        # split's contiguous slot range.
        n_starts = len(self._slots) - window + 1
        if n_starts <= 0:
            raise ValueError(
                f"split {split!r} has {len(self._slots)} slots, too few for window={window}"
            )
        self._starts: list[int] = self._slots[:n_starts]

        if norm_stats is None:
            train_slots = self._split_slots("train")
            if scheme == "global_clip":
                self.norm_stats = compute_norm_stats(self._od, train_slots, clip_val)
            else:  # zero_pinned_nonzero
                self.norm_stats = compute_norm_stats_nonzero(
                    self._od, train_slots, clip_val
                )
        else:
            # Pass through verbatim so callers that constructed the dict
            # themselves see exactly the same object. If the stats dict
            # carries an explicit scheme tag, it must match.
            ns = dict(norm_stats)
            expected_scheme = ns.get("scheme", scheme)
            if expected_scheme != scheme:
                raise ValueError(
                    f"norm_stats scheme {expected_scheme!r} does not match "
                    f"dataset scheme {scheme!r}"
                )
            self.norm_stats = ns

    def _split_slots(self, split: str) -> list[int]:
        """Return the sorted slot indices belonging to ``split``."""
        day_lo, day_hi = self._split_cfg[_SPLIT_KEYS[split]]
        lo = day_lo * self.slots_per_day
        hi = min(day_hi * self.slots_per_day, self.n_slots)
        return list(range(lo, hi))

    def __len__(self) -> int:
        return len(self._starts)

    def _apply_norm(self, padded: np.ndarray) -> np.ndarray:
        if self.scheme == "global_clip":
            return apply_norm(padded, self.norm_stats)
        return apply_norm_zero_pinned(padded, self.norm_stats)

    def _inverse_norm(self, x: np.ndarray) -> np.ndarray:
        if self.scheme == "global_clip":
            return inverse_norm(x, self.norm_stats)
        return inverse_norm_zero_pinned(x, self.norm_stats)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict[str, int]]:
        start = self._starts[idx]
        raw = np.asarray(self._od[start : start + self.window], dtype=np.float64)
        padded = pad_hw(raw, self.pad_size)
        x = self._apply_norm(padded)
        cond = slot_to_condition(start, self.start_dt, self.slots_per_day, self.bin_min)
        return x, cond

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Normalized padded array -> non-negative OD counts ``[..., |Z|, |Z|]``."""
        counts = self._inverse_norm(x)
        return unpad(counts, self.n_zones)

    def summary(self) -> dict[str, Any]:
        """Diagnostic summary used by the Stage-4A smoke script."""
        return {
            "split": self.split,
            "n_samples": len(self),
            "n_slots": self.n_slots,
            "n_zones": self.n_zones,
            "window": self.window,
            "pad_size": self.pad_size,
            "scheme": self.scheme,
            "split_slot_range": (self._slots[0], self._slots[-1]),
            "norm_stats": dict(self.norm_stats),
        }
