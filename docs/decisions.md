# Decision Log

This file records deviations from the plan documents and major
technical choices made during implementation.
Format: `## YYYY-MM-DD — <topic>` followed by what changed and why.

---

## 2026-05-14 — Stage 1: datetime format includes seconds

**Plan said**: parse `fDepTime` / `fDestTime` with
`format="%Y/%m/%d %H:%M"` (see `docs/plan/stage1_data_cleaning.md`,
Common Pitfalls).

**Reality**: the raw CSV's first data row shows
`fDepTime = "2023/7/16 10:43:05"` — month/day are not zero-padded
(matches plan) **but seconds are present** (does not match plan).

**Change**: parse with `format="%Y/%m/%d %H:%M:%S"`.
`DEFAULT_DATETIME_FORMAT` lives in `src/data/clean.py`; the same string
is exposed in `configs/preprocess.yaml`.

**Impact**: gain second-level precision in `dep_time` / `dest_time`
and downstream `duration_min`. No loss versus the planned format.

---

## 2026-05-14 — Stage 1: Python 3.10 instead of 3.11

**Plan said**: Python 3.11 (CLAUDE.md §5).

**Reality**: target machine has only system Python 3.10.12, no conda /
uv / poetry, no pre-installed Python 3.11. Installing 3.11 would
require either a heavy toolchain (miniconda) or a third-party binary
(uv); neither was available without extra setup.

**Change**: created `.venv` from system Python 3.10.12
(`python3 -m venv .venv` after `apt install python3.10-venv`). All
stage-1 code is type-hinted under PEP 604 (`X | Y`) which 3.10
supports; nothing in this stage needs 3.11-only features.

**Impact**: minimal. To revisit before stage 4 (PyTorch is fine on
3.10, so no near-term upgrade pressure).
