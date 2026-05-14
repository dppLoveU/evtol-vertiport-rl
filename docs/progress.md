# Progress Log

This file is appended to after every completed sub-task by Claude Code.
Format: `- YYYY-MM-DD [stageN] description (any caveats)`

- 2026-05-14 [stage1] bootstrapped python env: `.venv` from system Python 3.10.12 (apt installed `python3.10-venv` + `python3-pip`); pip-installed pandas 2.3.3, pyarrow 24.0.0, numpy 2.2.6, pytest 9.0.3, pyyaml 6.0.3 (caveat: 3.10 not 3.11 — see decisions log).
- 2026-05-14 [stage1] scaffolded package: `src/{constants,data,utils}`, `tests/`, `tests/fixtures/` with `__init__.py`; filled `pyproject.toml` with deps + pytest config.
- 2026-05-14 [stage1] task 1 (load_raw): implemented `src/data/clean.py::load_raw` with explicit `RAW_DTYPES` from `src/data/schema.py`; supports chunked read (default 500k) for the 4M-row file.
- 2026-05-14 [stage1] task 2 (coord fix): `src/data/clean.py::fix_coordinates` divides Int32 coords by `COORD_SCALE` in float64 then casts float32 (avoids precision loss on 9-digit ints); drops raw `fDep*` / `fDest*` columns.
- 2026-05-14 [stage1] task 3 (time parse): `src/data/clean.py::parse_times` uses `%Y/%m/%d %H:%M:%S` (deviation from plan — see decisions log), `errors="coerce"`, drops NaT rows and returns the dropped count for audit.
- 2026-05-14 [stage1] tests: `tests/fixtures/orders_sample.csv` (30 hand-crafted rows covering bbox / time / fare / age edge cases); `tests/test_clean.py` covers haversine, load_raw, fix_coordinates, parse_times, and an end-to-end pipeline check on the fixture.
- 2026-05-14 [meta] doc sync: updated CLAUDE.md §2/§3 (stage 1 in-progress), §5 (Python 3.10 reality, Hydra deferred to PyYAML for now, new "Test scope = current task" rule); fixed `docs/plan/stage1_data_cleaning.md` schema/pitfalls (datetime `%H:%M:%S`, fWaitTime float, encoding confirmed UTF-8); wrote setup section in `README.md`.
