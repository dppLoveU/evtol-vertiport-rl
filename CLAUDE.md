# Project: eVTOL Vertiport Location-Allocation via Diffusion + RL

> This file is automatically read by Claude Code at session start.
> Keep it concise and up to date. When in doubt, read `docs/progress.md`
> for current status and `docs/plan/stageN_*.md` for the active stage.

## 1. Goal

A master's thesis project. Optimize the placement of eVTOL vertiports in
Suzhou by:
1. Aggregating ~4M ride-hailing orders into an OD demand tensor.
2. Training a diffusion model on the OD tensor to sample plausible
   counterfactual demand scenarios.
3. Training an RL agent (MaskablePPO) to select K vertiport locations
   that maximize bilateral OD coverage, robust across diffusion-sampled
   demand scenarios.

Target venue: SCI Q2 (IEEE T-ITS, Transp. Res. Part C, Sust. Cities Soc.)
or strong EI journal. Manuscript expected in 8 weeks.

## 2. Current Stage

**Stage: 1 — Data Cleaning (not started)**

Update this line whenever a stage starts or completes. See
`docs/progress.md` for fine-grained log entries.

## 3. Plan Documents

All 7 stages have detailed specs in `docs/plan/`:

| Stage | File | Status |
|-------|------|--------|
| 1 | `docs/plan/stage1_data_cleaning.md` | not started |
| 2 | `docs/plan/stage2_spatial_discretization.md` | not started |
| 3 | `docs/plan/stage3_od_construction.md` | not started |
| 4 | `docs/plan/stage4_diffusion.md` | not started |
| 5 | `docs/plan/stage5_rl_env.md` | not started |
| 6 | `docs/plan/stage6_rl_training.md` | not started |
| 7 | `docs/plan/stage7_evaluation.md` | not started |

ALWAYS read the active stage's plan document end-to-end before writing
code for it.

## 4. Repository Layout

```
evtol-vertiport-rl/
├── CLAUDE.md                  this file
├── README.md
├── pyproject.toml             dependencies (uv or poetry)
├── configs/                   Hydra configs, one per pipeline step
├── data/                      gitignored
│   ├── raw/                   the original CSV from advisor
│   ├── processed/             cleaned + aggregated intermediates
│   └── synthetic/             diffusion-generated OD samples
├── docs/
│   ├── plan/                  stage specs (read these!)
│   ├── progress.md            running log (update after every sub-task)
│   └── decisions.md           record of plan deviations
├── src/
│   ├── data/                  cleaning, discretization, OD construction
│   ├── models/                diffusion U-Net, RL policy network
│   ├── envs/                  VertiportEnv (gymnasium)
│   ├── agents/                PPO wrappers, training callbacks
│   └── utils/                 seed, logging, checkpointing, geo helpers
├── experiments/               CLI entrypoints (python -m experiments.X)
├── notebooks/                 EDA only, never production code
├── results/                   gitignored; per-run artifacts
│   └── {stage}/{run_name}/    logs, ckpts, plots
└── tests/                     pytest, mirrors src/ layout
```

## 5. Engineering Conventions

- **Python**: 3.11. Type hints required on all public functions.
- **DL framework**: PyTorch 2.x. No TensorFlow.
- **Config**: Hydra. Never hardcode paths or hyperparameters; read from
  `configs/<step>.yaml`. Use `${oc.env:VAR}` for environment-dependent
  paths so the same config works on laptop and GPU server.
- **Seeding**: every script that uses randomness must call
  `src.utils.seed.set_seed(cfg.seed)` immediately after config load.
- **Logging**: WandB by default (`wandb.init(project="evtol-vertiport",
  name=run_name, config=OmegaConf.to_container(cfg))`). Fall back to
  TensorBoard if `cfg.logging.backend == "tb"`.
- **Checkpoints**: save to `results/{stage}/{run_name}/ckpt/`, with
  `best.pt`, `last.pt`, and `step_{N}.pt` for milestones.
- **Early stopping**: if eval metric does not improve for `patience`
  evals, stop. Configurable in YAML.
- **Comments and variable names**: English. Docstrings English.
  Only files under `docs/` may contain Chinese, and only for narrative
  prose, never for keys or identifiers.
- **Code style**: ruff for lint+format, line length 100. mypy strict on
  `src/`, lenient on `experiments/` and `notebooks/`.
- **Tests**: pytest. Aim for ~70% coverage on `src/data/` and
  `src/envs/` since those are the failure-prone glue layers.
- **Data files**: gitignored. Use `dvc` or just leave out of git.
  Commit small fixtures (<1 MB) under `tests/fixtures/`.
- **Math**: vectorize with numpy/torch. Avoid python loops over
  millions of rows; pandas + numpy broadcasting only.

## 6. Hard Rules for Claude Code

These rules are non-negotiable. Violating any of them requires explicit
user permission first.

1. **Read before write**. Before starting any stage, read the relevant
   `docs/plan/stageN_*.md` end-to-end. Before modifying an existing
   file, read the current version.
2. **Small steps**. Each turn, implement at most 3 sub-tasks from the
   active plan. After that, stop and let the user review.
3. **Log progress**. After completing each sub-task, append an entry to
   `docs/progress.md` with date, what was done, and any caveat.
4. **Record deviations**. If a technical decision differs from the
   plan, write a short entry to `docs/decisions.md` (date, what
   changed, why) and ASK the user before proceeding.
5. **Ask, don't guess**. When the plan is ambiguous, ask. Especially
   for: hyperparameters not pinned in the plan, choice of baseline,
   evaluation metric definitions, file format choices.
6. **Never delete data or git-tracked files** without explicit user
   confirmation in the chat.
7. **Don't start long-running training inside the session**. Anything
   expected to run > 10 minutes should be launched in the background
   (`nohup ... &` or `tmux`) and the session only checks logs.
8. **Don't auto-commit**. Stage files with `git add`, propose a commit
   message, but let the user run `git commit` themselves.
9. **No silent dependency additions**. Adding a package requires
   editing `pyproject.toml` and noting the addition in the response.

## 7. Project Constants

These values are referenced throughout the codebase. Keep them in
`src/constants.py` and never duplicate.

```python
# Suzhou bounding box (approximate, refined in Stage 1 EDA)
SUZHOU_BBOX = {"lon_min": 120.45, "lon_max": 120.95,
               "lat_min": 31.20,  "lat_max": 31.50}

# Raw coordinates are int (e.g. 120557806 = 120.557806°)
COORD_SCALE = 1e6

# Time discretization
TIME_BIN_MIN = 30           # 30-minute slots
NUM_TIME_BINS = 7 * 24 * 60 // TIME_BIN_MIN  # = 336

# eVTOL trip eligibility (refined in Stage 3 sensitivity analysis)
EVTOL_MIN_DIST_KM = 15.0
EVTOL_MIN_DURATION_MIN = 25.0

# Spatial discretization
H3_RESOLUTION = 7           # ~1.2 km hex edge

# Vertiport placement
DEFAULT_K = 10              # sweep 5..20 in sensitivity analysis
WALK_RADIUS_KM = 5.0
```

## 8. Key Design Decisions

For the full rationale, see `docs/decisions.md`. Headline choices:

- **OD matrix, not grid density**. Vertiports serve OD pairs; origin
  density alone cannot identify long enough trips for eVTOL.
- **Demand zones via H3 level 7**. Cleaner than admin boundaries, ~250
  hexagons for Suzhou. Edge length ~1.2 km matches walking radius.
- **Candidate vertiports via POI + uniform grid hybrid**. Start with
  OSM transit hubs, malls, industrial parks; pad with uniform grid
  centroids to ensure spatial coverage. Target |C| ≈ 300.
- **Diffusion target: OD slice images**. Treat each `[T_window, |Z|,
  |Z|]` tensor as a multi-channel image; DDPM with U-Net.
- **RL: MaskablePPO**. Action mask prevents revisiting placed sites.
- **Robustness: CVaR objective**. Optimize CVaR_α(reward) over
  diffusion-sampled scenarios, not just expectation.

## 9. Reference Repos (read-only, do not clone into project)

When you need to understand baseline behavior:

- `HIGISX/SpoNet` — MCLP / p-median DRL baseline, will be adapted in
  Stage 7 as a comparison method.
- `lucidrains/denoising-diffusion-pytorch` — clean DDPM reference,
  Stage 4 will adopt its U-Net structure.
- `Stable-Baselines-Team/sb3-contrib` — MaskablePPO source of truth.

## 10. Communication Protocol

When starting a session, the user will typically open with something
like "what's next". Default response:

1. Read `CLAUDE.md`, `docs/progress.md`, and the active stage plan.
2. State current stage and last completed sub-task.
3. Propose the next 1–3 sub-tasks with brief rationale.
4. Wait for user confirmation before writing code.

When ending a session:
1. Update `docs/progress.md` with what was done.
2. Stage relevant files with `git add` and propose a commit message.
3. Note any open questions or follow-ups.