"""Tests for src/data/od.py (stage-3 tasks 1-2: time bin + zone assignment).

Uses small synthetic frames only; no full-parquet read.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest

from src.constants import H3_RESOLUTION, NUM_TIME_BINS
from src.data.od import (
    assign_time_bin,
    assign_zones,
    build_od_tensor,
    build_zone_lookup,
    is_evtol_eligible,
)

RES = H3_RESOLUTION
T0 = pd.Timestamp("2023-07-16 00:00:00")


# --- task 1: assign_time_bin ---------------------------------------------


def _orders_at_offsets(offsets_min: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(len(offsets_min))],
            "dep_time": [T0 + pd.Timedelta(minutes=m) for m in offsets_min],
        }
    )


def test_assign_time_bin_floors_to_slot() -> None:
    in_range, out_of_range, stats = assign_time_bin(
        _orders_at_offsets([0, 15, 29, 30, 59, 60])
    )
    # 30-min bins: 0/15/29 -> slot 0, 30/59 -> slot 1, 60 -> slot 2.
    assert in_range["slot"].tolist() == [0, 0, 0, 1, 1, 2]
    assert in_range["slot"].dtype == np.int32
    assert len(out_of_range) == 0
    assert stats["n_in_range"] == 6
    assert stats["n_out_of_range"] == 0
    assert stats["t0"] == str(T0)


def test_assign_time_bin_drops_out_of_range() -> None:
    # Span = NUM_TIME_BINS * 30 min = 15840 min (11 days). An order
    # exactly at t0 + 15840 min lands on slot 528 (== NUM_TIME_BINS, out
    # of range); t0 + 15839 min is still slot 527 (the last valid slot).
    in_range, out_of_range, stats = assign_time_bin(
        _orders_at_offsets([0, 15839, 15840, 15870])
    )
    assert ((in_range["slot"] >= 0) & (in_range["slot"] < NUM_TIME_BINS)).all()
    assert in_range["slot"].tolist() == [0, 527]
    assert len(out_of_range) == 2
    assert stats["n_out_of_range"] == 2
    assert stats["max_slot"] == 529


# --- task 2: assign_zones ------------------------------------------------

A = (120.50, 31.30)
B = (120.80, 31.55)
C = (120.95, 31.70)
X = (121.10, 31.90)  # cell intentionally absent from the lookup


def _cell(lon: float, lat: float) -> str:
    return h3.geo_to_h3(lat, lon, RES)


def _orders_od(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(len(pairs))],
            "o_lon": np.array([o[0] for o, _ in pairs], dtype=np.float32),
            "o_lat": np.array([o[1] for o, _ in pairs], dtype=np.float32),
            "d_lon": np.array([d[0] for _, d in pairs], dtype=np.float32),
            "d_lat": np.array([d[1] for _, d in pairs], dtype=np.float32),
        }
    )


def _lookup() -> dict[str, int]:
    zones = pd.DataFrame(
        {
            "h3_index": [_cell(*A), _cell(*B), _cell(*C)],
            "zone_id": [0, 1, 2],
        }
    )
    return build_zone_lookup(zones)


def test_build_zone_lookup_maps_index_to_id() -> None:
    zones = pd.DataFrame({"h3_index": ["aaa", "bbb"], "zone_id": [0, 1]})
    assert build_zone_lookup(zones) == {"aaa": 0, "bbb": 1}


def test_assign_zones_maps_o_and_d() -> None:
    assigned, dropped, stats = assign_zones(
        _orders_od([(A, B), (B, C), (A, A)]), _lookup()
    )
    assert len(assigned) == 3
    assert len(dropped) == 0
    assert assigned["o_zone"].tolist() == [0, 1, 0]
    assert assigned["d_zone"].tolist() == [1, 2, 0]
    assert assigned["o_zone"].dtype == np.int32
    assert assigned["d_zone"].dtype == np.int32
    assert stats["drop_count"] == 0
    assert stats["n_zone_assigned"] == 3


def test_assign_zones_drops_unknown_zone() -> None:
    # o2's destination X maps to an H3 cell not in the lookup.
    assigned, dropped, stats = assign_zones(
        _orders_od([(A, B), (B, C), (A, X)]), _lookup()
    )
    assert len(assigned) == 2
    assert dropped["order_id"].tolist() == ["o2"]
    assert stats["drop_count"] == 1
    assert stats["drop_rate"] == 1 / 3
    assert stats["n_unknown_o"] == 0
    assert stats["n_unknown_d"] == 1


# --- task 4: is_evtol_eligible ------------------------------------------


def test_is_evtol_eligible_filters_dist_duration_and_intrazone() -> None:
    df = pd.DataFrame(
        {
            "geo_dist_km": [20.0, 10.0, 20.0, 20.0],
            "duration_min": [30.0, 30.0, 10.0, 30.0],
            "o_zone": [0, 0, 0, 2],
            "d_zone": [1, 1, 1, 2],
        }
    )
    mask = is_evtol_eligible(df, min_dist_km=15.0, min_duration_min=25.0)
    # row0 eligible; row1 too short; row2 too brief; row3 intra-zone.
    assert mask.tolist() == [True, False, False, False]


def test_is_evtol_eligible_excludes_diagonal_even_when_long() -> None:
    # A long, slow trip that is intra-zone must still be ineligible.
    df = pd.DataFrame(
        {
            "geo_dist_km": [80.0],
            "duration_min": [120.0],
            "o_zone": [4],
            "d_zone": [4],
        }
    )
    mask = is_evtol_eligible(df, min_dist_km=15.0, min_duration_min=25.0)
    assert mask.tolist() == [False]


# --- task 3: build_od_tensor --------------------------------------------


def test_build_od_tensor_count_aggregates() -> None:
    df = pd.DataFrame(
        {
            "slot": [0, 0, 0, 1],
            "o_zone": [0, 0, 1, 0],
            "d_zone": [1, 1, 2, 1],
        }
    )
    tensor, stats = build_od_tensor(df, n_time_bins=3, n_zones=3)
    assert tensor.shape == (3, 3, 3)
    assert tensor.dtype == np.int32
    assert tensor[0, 0, 1] == 2  # two (0,0,1) rows
    assert tensor[0, 1, 2] == 1
    assert tensor[1, 0, 1] == 1
    assert tensor.sum() == 4
    assert stats["nonzero_count"] == 3
    assert stats["dtype"] == "int32"


def test_build_od_tensor_weighted_sums_value_col() -> None:
    df = pd.DataFrame(
        {
            "slot": [0, 0, 1],
            "o_zone": [0, 0, 1],
            "d_zone": [1, 1, 2],
            "fare_yuan": [10.0, 5.0, 7.0],
        }
    )
    tensor, _ = build_od_tensor(df, n_time_bins=2, n_zones=3, value_col="fare_yuan")
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 1] == pytest.approx(15.0)  # 10 + 5
    assert tensor[1, 1, 2] == pytest.approx(7.0)
    assert tensor.sum() == pytest.approx(22.0)


def test_od_evtol_has_zero_diagonal() -> None:
    # End-to-end: the eVTOL filter drops the intra-zone row, so every
    # slot's diagonal in the resulting od_evtol must be zero.
    df = pd.DataFrame(
        {
            "slot": [0, 0, 1],
            "o_zone": [0, 1, 2],
            "d_zone": [1, 1, 0],  # row1 (1->1) is intra-zone
            "geo_dist_km": [20.0, 20.0, 20.0],
            "duration_min": [30.0, 30.0, 30.0],
        }
    )
    mask = is_evtol_eligible(df, min_dist_km=15.0, min_duration_min=25.0)
    od_evtol, _ = build_od_tensor(df[mask], n_time_bins=2, n_zones=3)
    for t in range(od_evtol.shape[0]):
        assert np.diagonal(od_evtol[t]).sum() == 0
    assert od_evtol.sum() == 2  # rows 0 and 2 survive


def test_build_od_tensor_rejects_out_of_range_indices() -> None:
    bad_slot = pd.DataFrame({"slot": [5], "o_zone": [0], "d_zone": [1]})
    with pytest.raises(ValueError):
        build_od_tensor(bad_slot, n_time_bins=3, n_zones=3)
    bad_zone = pd.DataFrame({"slot": [0], "o_zone": [9], "d_zone": [1]})
    with pytest.raises(ValueError):
        build_od_tensor(bad_zone, n_time_bins=3, n_zones=3)
