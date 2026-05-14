# evtol-vertiport-rl

[苏州网约车]-扩散模型 强化学习-个人项目

eVTOL vertiport location-allocation in Suzhou, optimized via a
diffusion model (counterfactual demand) + MaskablePPO (site selection
under CVaR robustness). Master's thesis project.

See `CLAUDE.md` for the project's full ground rules and `docs/plan/`
for per-stage specs.

## Setup (one-time per machine)

Tested on WSL2 Ubuntu with system Python 3.10.12. See
`docs/decisions.md` 2026-05-14 for why 3.10 instead of the
CLAUDE.md-targeted 3.11.

```bash
# OS-level prerequisites for venv + pip (system-wide, needs sudo):
sudo apt install -y python3.10-venv python3-pip

# Project virtualenv:
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# Runtime deps (pyproject.toml is the source of truth):
.venv/bin/pip install pandas pyarrow numpy pyyaml

# Dev tooling — install when you start using each:
.venv/bin/pip install pytest        # tests
# .venv/bin/pip install ruff mypy   # lint + types (later stages)
```

The 4M-row raw CSV lives outside the repo and is symlinked at
`data/raw/suzhou_orders_7days.csv`. `data/`, `results/`, `models/`,
`wandb/`, `.venv/` are all gitignored.

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```
