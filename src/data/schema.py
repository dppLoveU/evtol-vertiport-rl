"""Raw CSV schema, dtype map, simple renames, and filter thresholds.

The raw file has 53 columns; only the 14 below are read via ``usecols``.
Dtypes target a < 2 GB in-memory footprint for ~4M rows: int32 for
coords, float32 for monetary/distance, nullable Int16 for age (so NaNs
survive read; cast to int8 happens at write time, after dropna+clip).
"""
from __future__ import annotations

# Columns kept from the raw CSV. Order is the read order; rename happens
# afterwards.
KEEP_COLUMNS: list[str] = [
    "fGuid",
    "fDepLongitude",
    "fDepLatitude",
    "fDestLongitude",
    "fDestLatitude",
    "fDepTime",
    "fDestTime",
    "fDriveMile",
    "fDriveTime",
    "fWaitTime",
    "fFactPrice",
    "AreaName",
    "性别",
    "年龄",
]

# Dtypes for ``pd.read_csv``, keyed by RAW column name.
# - Coords are nullable Int32 (placeholder zeros are valid; NA-safe in
#   case the source ever uses NA instead of 0).
# - fDriveTime is seconds, kept as Int32; converted to drive_min later.
# - fWaitTime is float in samples ("0.0"), so float32 not int.
# - 性别 / 年龄 read as Chinese names (UTF-8). Renamed downstream.
RAW_DTYPES: dict[str, str] = {
    "fGuid":          "string",
    "fDepLongitude":  "Int32",
    "fDepLatitude":   "Int32",
    "fDestLongitude": "Int32",
    "fDestLatitude":  "Int32",
    "fDepTime":       "string",
    "fDestTime":      "string",
    "fDriveMile":     "float32",
    "fDriveTime":     "Int32",
    "fWaitTime":      "float32",
    "fFactPrice":     "float32",
    "AreaName":       "string",
    "性别":           "string",
    "年龄":           "Int16",
}

# Pass-through renames (no value transform). Coord / time / drive_time
# columns are renamed by their respective transformer functions.
SIMPLE_RENAME: dict[str, str] = {
    "fGuid":      "order_id",
    "fDriveMile": "drive_km",
    "fWaitTime":  "wait_min",
    "fFactPrice": "fare_yuan",
    "AreaName":   "area_name",
    "性别":       "gender",
    "年龄":       "age",
}

# Final output column order, per docs/plan/stage1_data_cleaning.md.
OUTPUT_COLUMNS: list[str] = [
    "order_id",
    "o_lon", "o_lat", "d_lon", "d_lat",
    "dep_time", "dest_time",
    "duration_min", "drive_km", "drive_min", "geo_dist_km",
    "wait_min", "fare_yuan",
    "area_name", "gender", "age",
]

# Filter thresholds (consumed by task 5; centralized for testability).
FILTER_THRESHOLDS: dict[str, object] = {
    "duration_min_range":  (2.0, 180.0),     # inclusive on both ends
    "drive_km_range":      (0.5, 100.0),
    "geo_dist_km_min":     0.3,
    "geo_drive_ratio_max": 3.0,              # geo / drive must be < this
    "fare_yuan_range":     (0.0, 1000.0),    # 0 < fare <= 1000
    "age_clip":            (0, 127),         # for cast to int8 at write
}
