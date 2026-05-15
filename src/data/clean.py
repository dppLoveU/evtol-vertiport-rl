"""Stage-1 cleaning building blocks (tasks 1-6).

Implements load, coordinate fix, time parse, derived columns, the 6
outlier filters from the plan, and final output column selection.
Task 7 (Parquet write) lives in the CLI entrypoint, not here. Each
function returns a new DataFrame so steps compose in a single
pipeline expression.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.constants import COORD_SCALE, SUZHOU_BBOX
from src.data.schema import (
    FILTER_THRESHOLDS,
    KEEP_COLUMNS,
    OUTPUT_COLUMNS,
    RAW_DTYPES,
    SIMPLE_RENAME,
)
from src.utils.geo import haversine_km

# Real CSV uses zero-padded-free dates with seconds, e.g. "2023/7/16 10:43:05".
# (Plan doc lists "%Y/%m/%d %H:%M"; see docs/decisions.md 2026-05-14.)
DEFAULT_DATETIME_FORMAT: str = "%Y/%m/%d %H:%M:%S"
DEFAULT_ENCODING: str = "utf-8"


def load_raw(
    csv_path: str | Path,
    *,
    chunksize: int | None = 500_000,
    nrows: int | None = None,
    encoding: str = DEFAULT_ENCODING,
    usecols: Iterable[str] = KEEP_COLUMNS,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes to bound memory.

    Reads only ``usecols``. When ``chunksize`` is set, iterates and
    concatenates so peak memory stays roughly one chunk + result.
    ``nrows`` caps the total rows read (handy for smoke tests on a
    subset of the 4M-row file). Returned frame still uses raw column
    names; rename / coord fix / time parse happen downstream.
    """
    csv_path = Path(csv_path)
    if dtype is None:
        dtype = RAW_DTYPES
    usecols = list(usecols)

    if chunksize is None or chunksize <= 0:
        return pd.read_csv(
            csv_path,
            usecols=usecols,
            dtype=dtype,
            encoding=encoding,
            nrows=nrows,
        )

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype,
        encoding=encoding,
        chunksize=chunksize,
        nrows=nrows,
    )
    return pd.concat(reader, ignore_index=True, copy=False)


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw int coordinates to float32 degrees.

    Adds ``o_lon``, ``o_lat``, ``d_lon``, ``d_lat`` and drops the four
    raw ``fDep*`` / ``fDest*`` columns. The intermediate division uses
    float64 because float32 cannot represent 9-digit ints exactly
    (precision threshold is 2**24 ~= 1.68e7).
    """
    out = df.copy()
    coord_pairs = [
        ("fDepLongitude",  "o_lon"),
        ("fDepLatitude",   "o_lat"),
        ("fDestLongitude", "d_lon"),
        ("fDestLatitude",  "d_lat"),
    ]
    for raw, new in coord_pairs:
        # to_numpy(dtype="float64") tolerates Int32 NA by emitting NaN.
        arr64 = out[raw].to_numpy(dtype="float64") / COORD_SCALE
        out[new] = arr64.astype("float32")
    return out.drop(columns=[raw for raw, _ in coord_pairs])


def parse_times(
    df: pd.DataFrame,
    *,
    fmt: str = DEFAULT_DATETIME_FORMAT,
) -> tuple[pd.DataFrame, int]:
    """Parse fDepTime / fDestTime; drop rows where either fails.

    Returns ``(cleaned_df, n_dropped)``. Unparseable rows are dropped
    silently in production; the count is returned so callers can log it
    for the EDA audit.
    """
    out = df.copy()
    out["dep_time"]  = pd.to_datetime(out["fDepTime"],  format=fmt, errors="coerce")
    out["dest_time"] = pd.to_datetime(out["fDestTime"], format=fmt, errors="coerce")
    bad_mask = out["dep_time"].isna() | out["dest_time"].isna()
    n_dropped = int(bad_mask.sum())
    out = out.loc[~bad_mask].copy()
    return out.drop(columns=["fDepTime", "fDestTime"]), n_dropped


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``duration_min``, ``drive_min``, ``geo_dist_km``; drop fDriveTime.

    All three derived columns are float32 to match the output schema.
    ``duration_min`` carries second-level precision from parse_times.
    """
    out = df.copy()
    secs = (out["dest_time"] - out["dep_time"]).dt.total_seconds()
    out["duration_min"] = (secs / 60.0).astype("float32")
    out["drive_min"]    = (out["fDriveTime"].to_numpy(dtype="float64") / 60.0).astype("float32")
    out["geo_dist_km"]  = haversine_km(
        out["o_lat"], out["o_lon"], out["d_lat"], out["d_lon"],
    ).astype("float32")
    return out.drop(columns=["fDriveTime"])


def apply_simple_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pass-through column renames per :data:`SIMPLE_RENAME`.

    Idempotent: calling on an already-renamed frame is a no-op (pandas
    silently leaves columns alone if their old names are absent). Must
    run before :func:`apply_outlier_filters` so filters can reference
    ``drive_km`` and ``fare_yuan`` (which are post-rename names).
    """
    return df.rename(columns=SIMPLE_RENAME)


def apply_outlier_filters(
    df: pd.DataFrame,
    *,
    bbox: dict[str, float] | None = None,
    duration_min_range: tuple[float, float] = (2.0, 180.0),
    drive_km_range: tuple[float, float] = (0.5, 100.0),
    geo_dist_km_min: float = 0.3,
    geo_drive_ratio_max: float = 3.0,
    fare_yuan_range: tuple[float, float] = (0.0, 1000.0),
    wait_min_range: tuple[float, float] = (0.0, 120.0),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the 7 outlier filters in order (a)-(g).

    Filters (a)-(f) are the original plan spec. Filter (g) on
    ``wait_min`` was added 2026-05-14 after the full-data NA audit
    surfaced wait values up to 764k minutes (see
    ``docs/decisions.md``).

    Returns ``(cleaned_df, audit)``. The audit dict records the row
    count remaining after each step under keys ``initial``,
    ``after_bbox``, ``after_duration``, ``after_drive_km``,
    ``after_geo_dist``, ``after_geo_drive_ratio``, ``after_fare``,
    ``after_wait``. NA values in any filtered column produce a False
    mask and are dropped along with out-of-range values. Order
    matters: (e) divides by drive_km, which (c) lower-bounded at
    0.5, so no division by zero.
    """
    if bbox is None:
        bbox = SUZHOU_BBOX

    audit: dict[str, int] = {"initial": len(df)}
    out = df

    # (a) bounding box on all four coordinates.
    in_bbox = (
        out["o_lon"].between(bbox["lon_min"], bbox["lon_max"])
        & out["o_lat"].between(bbox["lat_min"], bbox["lat_max"])
        & out["d_lon"].between(bbox["lon_min"], bbox["lon_max"])
        & out["d_lat"].between(bbox["lat_min"], bbox["lat_max"])
    )
    out = out[in_bbox]
    audit["after_bbox"] = len(out)

    # (b) trip duration window.
    dur_lo, dur_hi = duration_min_range
    out = out[out["duration_min"].between(dur_lo, dur_hi)]
    audit["after_duration"] = len(out)

    # (c) on-road distance window.
    dk_lo, dk_hi = drive_km_range
    out = out[out["drive_km"].between(dk_lo, dk_hi)]
    audit["after_drive_km"] = len(out)

    # (d) drop near-zero ODs.
    out = out[out["geo_dist_km"] >= geo_dist_km_min]
    audit["after_geo_dist"] = len(out)

    # (e) GPS sanity: drive distance must dominate geo distance.
    out = out[(out["geo_dist_km"] / out["drive_km"]) < geo_drive_ratio_max]
    audit["after_geo_drive_ratio"] = len(out)

    # (f) fare window: 0 < fare <= upper.
    fare_lo, fare_hi = fare_yuan_range
    out = out[(out["fare_yuan"] > fare_lo) & (out["fare_yuan"] <= fare_hi)]
    audit["after_fare"] = len(out)

    # (g) wait_min window (plan extension; see docs/decisions.md 2026-05-14).
    wait_lo, wait_hi = wait_min_range
    out = out[out["wait_min"].between(wait_lo, wait_hi)]
    audit["after_wait"] = len(out)

    return out.copy(), audit


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast dtypes and reorder per OUTPUT_COLUMNS.

    Contract: ``df`` must already be free of NA in any output column.
    Upstream callers are responsible for handling NA (e.g., dropping
    NA-age rows before invocation). Raises ``ValueError`` if any NA
    is present, surfacing data-quality assumptions instead of
    silently dropping rows. The full-data audit (2026-05-14) showed
    zero NA in raw data, so the raise should never fire in practice;
    if it does, that's a signal that the upstream contract changed.

    Assumes :func:`apply_simple_renames` has been called.
    """
    cols_present = [c for c in OUTPUT_COLUMNS if c in df.columns]
    na_per_col = df[cols_present].isna().sum()
    if int(na_per_col.sum()) > 0:
        bad = na_per_col[na_per_col > 0].to_dict()
        raise ValueError(f"select_output_columns received NA values: {bad}")

    out = df.copy()
    age_lo, age_hi = FILTER_THRESHOLDS["age_clip"]  # type: ignore[misc]
    # Two-step cast: nullable Int16 -> numpy int16 (no NA by contract)
    # -> numpy int8 (safe after clip into [0, 127]).
    out["age"] = out["age"].astype("int16").clip(age_lo, age_hi).astype("int8")

    out = out.astype({
        "order_id":  "string",
        "wait_min":  "float32",
        "fare_yuan": "float32",
        "drive_km":  "float32",
        "area_name": "string",
        "gender":    "string",
    })

    return out[OUTPUT_COLUMNS]
