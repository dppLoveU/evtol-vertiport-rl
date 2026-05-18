"""Stage-3 tasks 1-2: time-bin and zone assignment for OD construction.

``assign_time_bin`` maps each order's departure timestamp to a discrete
time slot; ``assign_zones`` maps each order's origin and destination to
the Stage-2 H3 demand zones. Both split out rows that fall outside the
valid range and return per-step statistics for auditing.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from src.constants import H3_RESOLUTION, NUM_TIME_BINS, TIME_BIN_MIN

# Decimal places for the endpoint-rounding key. Matches src/data/zones.py
# so that an order's O/D resolves to the same H3 cell Stage 2 used when
# it built the demand zones. 5 dp is ~1.1 m, far finer than the res-7
# cell (~1.2 km edge), so rounding never changes cell membership.
_DEDUP_DECIMALS = 5


def assign_time_bin(
    orders: pd.DataFrame,
    *,
    t0: pd.Timestamp | None = None,
    time_bin_min: int = TIME_BIN_MIN,
    num_time_bins: int = NUM_TIME_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Assign each order a discrete departure time slot.

    ``slot = floor((dep_time - t0) / time_bin_min)``. When ``t0`` is
    ``None`` it defaults to ``min(dep_time)``. Orders whose slot falls
    outside ``[0, num_time_bins)`` -- e.g. an order exactly at
    ``t0 + 7 days`` lands on ``slot == num_time_bins`` -- are split out
    rather than clamped, so the in-range frame is always safe to index.

    Returns ``(in_range, out_of_range, stats)``. ``in_range`` gains an
    int32 ``slot`` column with every value in ``[0, num_time_bins)``.
    """
    orders = orders.reset_index(drop=True)
    dep_time = orders["dep_time"]
    if t0 is None:
        t0 = dep_time.min()

    delta_s = (dep_time - t0).dt.total_seconds()
    slot = np.floor(delta_s / (time_bin_min * 60)).astype("int64")
    in_range_mask = (slot >= 0) & (slot < num_time_bins)

    in_range = orders[in_range_mask].copy()
    in_range["slot"] = slot[in_range_mask].astype(np.int32).to_numpy()
    out_of_range = orders[~in_range_mask].copy()

    stats: dict[str, object] = {
        "t0": str(t0),
        "time_bin_min": time_bin_min,
        "num_time_bins": num_time_bins,
        "n_total": int(len(orders)),
        "n_in_range": int(in_range_mask.sum()),
        "n_out_of_range": int((~in_range_mask).sum()),
        "min_slot": int(slot.min()) if len(slot) else None,
        "max_slot": int(slot.max()) if len(slot) else None,
    }
    return in_range, out_of_range, stats


def build_zone_lookup(zones: pd.DataFrame) -> dict[str, int]:
    """Map ``h3_index -> zone_id`` from a Stage-2 zones frame."""
    return {str(h): int(z) for h, z in zip(zones["h3_index"], zones["zone_id"])}


def _h3_index_points(
    lon: np.ndarray, lat: np.ndarray, resolution: int
) -> pd.Series:
    """H3 index per ``(lon, lat)`` row.

    Coordinates are rounded to ``_DEDUP_DECIMALS`` and de-duplicated
    before indexing, so the H3 call runs once per distinct location
    rather than once per order -- the same trick src/data/zones.py uses.
    """
    klon = np.round(np.asarray(lon, dtype=np.float64), _DEDUP_DECIMALS)
    klat = np.round(np.asarray(lat, dtype=np.float64), _DEDUP_DECIMALS)
    frame = pd.DataFrame({"klon": klon, "klat": klat})
    distinct = frame.drop_duplicates().reset_index(drop=True)
    # h3-py uses (lat, lon) order -- opposite the GIS convention.
    distinct["h3_index"] = [
        h3.geo_to_h3(la, lo, resolution)
        for lo, la in zip(distinct["klon"].to_numpy(), distinct["klat"].to_numpy())
    ]
    return frame.merge(distinct, on=["klon", "klat"], how="left")["h3_index"]


def assign_zones(
    orders: pd.DataFrame,
    zone_lookup: dict[str, int],
    *,
    resolution: int = H3_RESOLUTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Assign each order an origin and destination demand zone.

    The order's O and D coordinates are indexed at H3 ``resolution`` and
    mapped to ``zone_id`` via ``zone_lookup``. Orders whose O or D lands
    in an H3 cell absent from the lookup (a sparse cell Stage 2 dropped)
    are split out rather than kept with a sentinel.

    Returns ``(assigned, dropped, stats)``. ``assigned`` gains int32
    ``o_zone`` / ``d_zone`` columns; ``dropped`` keeps the original
    columns for downstream auditing (e.g. an area-name breakdown).
    """
    orders = orders.reset_index(drop=True)
    o_h3 = _h3_index_points(
        orders["o_lon"].to_numpy(), orders["o_lat"].to_numpy(), resolution
    )
    d_h3 = _h3_index_points(
        orders["d_lon"].to_numpy(), orders["d_lat"].to_numpy(), resolution
    )
    o_zone = o_h3.map(zone_lookup)
    d_zone = d_h3.map(zone_lookup)
    known = o_zone.notna() & d_zone.notna()

    assigned = orders[known].copy()
    assigned["o_zone"] = o_zone[known].astype(np.int32).to_numpy()
    assigned["d_zone"] = d_zone[known].astype(np.int32).to_numpy()
    dropped = orders[~known].copy()

    n_total = len(orders)
    stats: dict[str, object] = {
        "n_total": int(n_total),
        "n_zone_assigned": int(known.sum()),
        "drop_count": int((~known).sum()),
        "drop_rate": float((~known).sum()) / n_total if n_total else 0.0,
        "n_unknown_o": int(o_zone.isna().sum()),
        "n_unknown_d": int(d_zone.isna().sum()),
    }
    return assigned, dropped, stats
