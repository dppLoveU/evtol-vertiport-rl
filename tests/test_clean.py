"""Tests for stage-1 cleaning steps (tasks 1-3).

Outlier filtering and the parquet write are not yet implemented; their
tests will be added in the task 5-7 turn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    add_derived_columns,
    apply_outlier_filters,
    apply_simple_renames,
    fix_coordinates,
    load_raw,
    parse_times,
    select_output_columns,
)
from src.data.schema import KEEP_COLUMNS, OUTPUT_COLUMNS
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


# ---------- task 4: derived columns ----------

def test_derived_columns_compute_correctly():
    df = pd.DataFrame({
        "dep_time":   [pd.Timestamp("2023-07-16 10:00:00")],
        "dest_time":  [pd.Timestamp("2023-07-16 10:30:00")],
        "fDriveTime": pd.array([1800], dtype="Int32"),  # 1800 sec = 30 min
        "o_lat":      np.array([39.9042], dtype=np.float32),
        "o_lon":      np.array([116.4074], dtype=np.float32),
        "d_lat":      np.array([31.2304], dtype=np.float32),
        "d_lon":      np.array([121.4737], dtype=np.float32),
    })
    out = add_derived_columns(df)

    assert out["duration_min"].iloc[0] == pytest.approx(30.0, abs=1e-3)
    assert out["drive_min"].iloc[0]    == pytest.approx(30.0, abs=1e-3)
    # Beijing -> Shanghai ~ 1064 km.
    assert abs(float(out["geo_dist_km"].iloc[0]) - 1064.0) / 1064.0 < 0.05

    for c in ("duration_min", "drive_min", "geo_dist_km"):
        assert out[c].dtype == np.float32

    # fDriveTime is consumed and should be dropped.
    assert "fDriveTime" not in out.columns


# ---------- task 5: outlier filters ----------

def test_outlier_filter_each_step_drops_expected_rows():
    """Each of the 6 plan filters (a)-(f) drops at least one fixture row."""
    df = load_raw(FIXTURE, chunksize=None)
    df = fix_coordinates(df)
    df, _ = parse_times(df)
    df = add_derived_columns(df)
    df = apply_simple_renames(df)
    out, audit = apply_outlier_filters(df)

    # Initial = 28 (parse_times dropped rows 21, 22 with bad time strings).
    assert audit["initial"] == 28
    # (a) bbox: drops Beijing (id-011) and zero-coord (id-012).
    assert audit["after_bbox"] == 26
    # (b) duration: drops 30s (id-013) and 4hr (id-014).
    assert audit["after_duration"] == 24
    # (c) drive_km: drops 0.2 km (id-015) and 150 km (id-016).
    assert audit["after_drive_km"] == 22
    # (d) geo_dist: drops origin==dest (id-017).
    assert audit["after_geo_dist"] == 21
    # (e) geo/drive ratio: drops big-geo / tiny-drive (id-018).
    assert audit["after_geo_drive_ratio"] == 20
    # (f) fare: drops fare=0 (id-019) and fare=1500 (id-020).
    assert audit["after_fare"] == 18
    # (g) wait_min: every fixture row has wait_min=0.0, so no drops here.
    assert audit["after_wait"] == 18

    assert len(out) == 18


# ---------- task 6: column selection ----------

def test_column_selection_dtypes_and_order():
    df = load_raw(FIXTURE, chunksize=None)
    df = fix_coordinates(df)
    df, _ = parse_times(df)
    df = add_derived_columns(df)
    df = apply_simple_renames(df)
    df, _ = apply_outlier_filters(df)
    # New select_output_columns contract: input must be NA-free. The
    # caller (this test, mirroring what the CLI will do) drops NA-age
    # explicitly here. id-023 is the only fixture row with NaN age and
    # it survives all 7 outlier filters.
    df = df.dropna(subset=["age"])
    final = select_output_columns(df)

    # Column order must match OUTPUT_COLUMNS exactly.
    assert list(final.columns) == OUTPUT_COLUMNS

    # Numeric dtypes per the plan output schema.
    assert final["age"].dtype == np.int8
    for c in ("o_lon", "o_lat", "d_lon", "d_lat",
              "duration_min", "drive_km", "drive_min", "geo_dist_km",
              "wait_min", "fare_yuan"):
        assert final[c].dtype == np.float32, f"{c}: expected float32, got {final[c].dtype}"

    for c in ("dep_time", "dest_time"):
        assert pd.api.types.is_datetime64_any_dtype(final[c])

    for c in ("order_id", "area_name", "gender"):
        # StringDtype or plain object both satisfy the plan's "str" entry.
        assert pd.api.types.is_string_dtype(final[c]) or final[c].dtype == object

    # Acceptance criterion: zero NaN anywhere in the final frame.
    assert int(final.isna().sum().sum()) == 0

    # NA-age row (id-023) is dropped by the caller-side dropna above.
    assert "id-023" not in set(final["order_id"].astype(str))
    # age=999 row (id-024) survives but is clipped to 127.
    assert int(final.loc[final["order_id"] == "id-024", "age"].iloc[0]) == 127

    # Final = 18 (after 7 filters) - 1 (caller-side dropna for NA-age) = 17.
    assert len(final) == 17


def test_select_output_columns_raises_on_na_input():
    """New contract: select_output_columns surfaces NA via ValueError."""
    df = load_raw(FIXTURE, chunksize=None)
    df = fix_coordinates(df)
    df, _ = parse_times(df)
    df = add_derived_columns(df)
    df = apply_simple_renames(df)
    df, _ = apply_outlier_filters(df)
    # id-023 has NaN age and survives all 7 outlier filters.
    with pytest.raises(ValueError, match="NA"):
        select_output_columns(df)
