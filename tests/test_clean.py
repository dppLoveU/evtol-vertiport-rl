"""Tests for stage-1 cleaning steps (tasks 1-3).

Outlier filtering and the parquet write are not yet implemented; their
tests will be added in the task 5-7 turn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import fix_coordinates, load_raw, parse_times
from src.data.schema import KEEP_COLUMNS
from src.utils.geo import haversine_km

FIXTURE = Path(__file__).parent / "fixtures" / "orders_sample.csv"


# ---------- haversine ----------

def test_haversine_beijing_shanghai_within_5pct():
    # Beijing (39.9042, 116.4074) -> Shanghai (31.2304, 121.4737) ~ 1064 km.
    d = haversine_km(39.9042, 116.4074, 31.2304, 121.4737)
    assert abs(float(d) - 1064.0) / 1064.0 < 0.05


def test_haversine_zero_distance_is_zero():
    d = haversine_km(31.30, 120.60, 31.30, 120.60)
    assert float(d) == pytest.approx(0.0, abs=1e-9)


def test_haversine_vectorized_broadcasting():
    lat1 = np.array([31.30, 39.9042])
    lon1 = np.array([120.60, 116.4074])
    lat2 = np.array([31.30, 31.2304])
    lon2 = np.array([120.60, 121.4737])
    d = haversine_km(lat1, lon1, lat2, lon2)
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert abs(d[1] - 1064.0) / 1064.0 < 0.05


def test_haversine_symmetric():
    # Distance must be invariant under endpoint swap.
    a = haversine_km(31.30, 120.60, 31.50, 120.90)
    b = haversine_km(31.50, 120.90, 31.30, 120.60)
    assert float(a) == pytest.approx(float(b), rel=1e-12)


# ---------- load_raw ----------

def test_load_raw_reads_fixture_with_correct_columns():
    df = load_raw(FIXTURE, chunksize=None)
    assert list(df.columns) == KEEP_COLUMNS
    assert len(df) == 30


def test_load_raw_chunked_matches_single_shot():
    df_single  = load_raw(FIXTURE, chunksize=None)
    df_chunked = load_raw(FIXTURE, chunksize=10)
    pd.testing.assert_frame_equal(
        df_single.reset_index(drop=True),
        df_chunked.reset_index(drop=True),
    )


def test_load_raw_int32_coords_and_string_times():
    df = load_raw(FIXTURE, chunksize=None)
    # Int32 is the pandas-nullable extension dtype; its name is "Int32".
    assert str(df["fDepLongitude"].dtype) == "Int32"
    assert str(df["fDepTime"].dtype) == "string"


# ---------- fix_coordinates ----------

def test_fix_coordinates_divides_by_1e6_and_casts_float32():
    df = pd.DataFrame({
        "fDepLongitude":  pd.array([120557806, 120601994], dtype="Int32"),
        "fDepLatitude":   pd.array([31318374,  31319905],  dtype="Int32"),
        "fDestLongitude": pd.array([120568916, 120620530], dtype="Int32"),
        "fDestLatitude":  pd.array([31883451,  31301116],  dtype="Int32"),
    })
    out = fix_coordinates(df)
    for c in ("o_lon", "o_lat", "d_lon", "d_lat"):
        assert c in out.columns
        assert out[c].dtype == np.float32
    assert out["o_lon"].iloc[0] == pytest.approx(120.557806, abs=1e-4)
    assert out["o_lat"].iloc[0] == pytest.approx(31.318374,  abs=1e-4)
    assert out["d_lon"].iloc[1] == pytest.approx(120.620530, abs=1e-4)
    # Raw columns must be dropped.
    for c in ("fDepLongitude", "fDepLatitude", "fDestLongitude", "fDestLatitude"):
        assert c not in out.columns


# ---------- parse_times ----------

def test_parse_times_valid_strings_with_seconds():
    df = pd.DataFrame({
        "fDepTime":  ["2023/7/16 10:38:04", "2023/7/16 10:31:13"],
        "fDestTime": ["2023/7/16 10:56:59", "2023/7/16 10:58:08"],
    })
    out, n_dropped = parse_times(df)
    assert n_dropped == 0
    assert len(out) == 2
    assert pd.api.types.is_datetime64_any_dtype(out["dep_time"])
    assert pd.api.types.is_datetime64_any_dtype(out["dest_time"])
    assert out["dep_time"].iloc[0]  == pd.Timestamp("2023-07-16 10:38:04")
    assert out["dest_time"].iloc[0] == pd.Timestamp("2023-07-16 10:56:59")
    # Raw cols dropped.
    assert "fDepTime" not in out.columns
    assert "fDestTime" not in out.columns


def test_parse_times_drops_unparseable_rows_and_returns_count():
    df = pd.DataFrame({
        "fDepTime":  ["2023/7/16 10:38:04", "garbage",      "2023/7/17 09:00:00"],
        "fDestTime": ["2023/7/16 10:56:59", "more garbage", "2023-13-45 99:99:99"],
    })
    out, n_dropped = parse_times(df)
    # row 0 valid; row 1 both bad; row 2 dest bad -> 2 dropped.
    assert n_dropped == 2
    assert len(out) == 1
    assert out["dep_time"].iloc[0] == pd.Timestamp("2023-07-16 10:38:04")


def test_parse_times_preserves_seconds_precision():
    df = pd.DataFrame({
        "fDepTime":  ["2023/7/16 10:38:04"],
        "fDestTime": ["2023/7/16 10:56:59"],
    })
    out, _ = parse_times(df)
    assert out["dep_time"].iloc[0].second  == 4
    assert out["dest_time"].iloc[0].second == 59


# ---------- pipeline end-to-end on the fixture ----------

def test_pipeline_fixture_end_to_end():
    df = load_raw(FIXTURE, chunksize=None)
    df = fix_coordinates(df)
    df, n_dropped = parse_times(df)
    # rows 21 and 22 carry invalid time strings -> both dropped.
    assert n_dropped == 2
    assert len(df) == 28
    for c in ("o_lon", "o_lat", "d_lon", "d_lat"):
        assert df[c].dtype == np.float32
    for c in ("dep_time", "dest_time"):
        assert pd.api.types.is_datetime64_any_dtype(df[c])
    # Coord plausibility (bbox filter is task 5; here we only assert
    # that the non-placeholder rows look roughly like Suzhou).
    nonzero = df[(df["o_lon"] > 0) & (df["o_lat"] > 0)]
    assert (nonzero["o_lon"].between(115.0, 122.0)).all()
    assert (nonzero["o_lat"].between(30.0, 41.0)).all()
