# Stage 7: Evaluation & Paper Artifacts

## Purpose

Convert the Stage 6 ablation results into publication-ready figures,
tables, and statistical analyses. Run the robustness and generalization
experiments that distinguish a strong paper from a weak one. Produce a
single bundle of artifacts that the manuscript draws from.

This stage does not introduce new models; it polishes results.

## Inputs

- `results/stage6/ablation_summary.csv` from Stage 6.
- Every `selected.json` and `best.pt` from Stage 6.
- Stage 2 outputs (zones, candidates) for map plotting.
- Stage 3 outputs (od_evtol) for ground-truth evaluation.
- Stage 4 outputs (od_samples) for robustness scenarios.

## Outputs

### Tables (CSV, also rendered as LaTeX)

- `results/stage7/table_main.csv` — main ablation table:
  rows = method × K, columns = mean coverage, std, worst-case
  coverage, mean time. This is paper Table 1 or 2.
- `results/stage7/table_sensitivity_K.csv`
- `results/stage7/table_sensitivity_R.csv`
- `results/stage7/table_generalization.csv` — OOD experiments below.

### Figures (PDF for LaTeX, PNG for slides)

- `fig_method_comparison.pdf` — bar chart of mean coverage with
  error bars across methods (paper main result).
- `fig_pareto_K.pdf` — coverage vs K curve for each method, showing
  diminishing returns.
- `fig_selected_locations.pdf` — Suzhou map with selected vertiports
  for our method vs SpoNet vs K-means, side-by-side.
- `fig_worst_case.pdf` — for A6 vs A7, distribution of coverage
  across 64 diffusion scenarios; A7 should have tighter left tail.
- `fig_temporal_coverage.pdf` — line chart, coverage rate by hour of
  day under our method.
- `fig_diffusion_quality.pdf` — copy from Stage 4: real vs generated
  OD marginals.
- `fig_training_curves.pdf` — A5 vs A6 vs A7 training reward curves.
- `fig_ablation_breakdown.pdf` — bar chart isolating each component:
  RL alone, +diffusion, +CVaR.

### Statistical analyses

- `results/stage7/stat_tests.json` — paired t-test or Wilcoxon
  between our method (A7) and each baseline, across seeds. Report
  p-values.
- `results/stage7/effect_sizes.json` — Cohen's d for the same
  comparisons.

### Generalization experiments

These are the experiments most likely to win over a reviewer.

1. **Time-split generalization**:
   - Retrain A5/A6/A7 using only days 1–5 of training data.
   - Evaluate on days 6–7.
   - Hypothesis: A6/A7 generalize better than A5 because diffusion
     samples cover unseen demand patterns.
   - Output: `table_time_split.csv`.

2. **Demand shift**:
   - Synthetically modify the eval OD to simulate a new commercial
     district (multiply demand to a chosen zone by 2× or 3×).
   - Re-evaluate all methods without retraining.
   - Hypothesis: A7 degrades least.
   - Output: `table_demand_shift.csv`.

3. **Cross-K transfer**:
   - Train at K=10, evaluate at K=5 and K=15 (early stopping or
     extension of the placement sequence).
   - Output: `table_cross_K.csv`.

## Tasks

1. **Evaluator class** (`src/eval/evaluator.py`):
   - `Evaluator(env_factory, policy)` — runs N episodes deterministically
     and computes coverage statistics.
   - Supports: real eval set, single diffusion scenario, multi-scenario
     (returns array of coverage values).

2. **Main table generator** (`experiments/run_stage7_main_table.py`):
   - Iterates every `results/stage6/{run_name}/` directory.
   - Loads best policy, runs evaluator, fills the CSV.
   - Also formats as LaTeX with proper rounding (2 decimals for
     coverage, 1 for time).

3. **Figure scripts** — one Python script per figure for reproducibility:
   - `experiments/fig_method_comparison.py`
   - `experiments/fig_pareto_K.py`
   - `experiments/fig_selected_locations.py`
   - ... etc.
   Each script reads only from CSV/JSON outputs above, never re-runs
   training. This means figures can be regenerated quickly without
   re-running expensive experiments.

4. **Statistical tests** (`experiments/run_stage7_stats.py`):
   For each pair (A7 vs A_i for i ∈ {0,1,2,3,4,5,6}), compute paired
   Wilcoxon test on the seed-wise coverage values. Save to JSON.

5. **Generalization experiments**:
   - `experiments/run_stage7_time_split.py` — orchestrates the
     time-split retrain + eval.
   - `experiments/run_stage7_demand_shift.py` — perturbs OD and
     evaluates.
   - `experiments/run_stage7_cross_K.py` — varies K at eval time.

6. **Selected location maps** (`experiments/fig_selected_locations.py`):
   - Plot Suzhou base map (`folium` or `contextily` for static).
   - Overlay zones as light hexagons.
   - Plot selected candidates as colored markers, one color per method.
   - Add a coverage halo (circle of `WALK_RADIUS_KM`) around each
     vertiport.
   - For the paper, render as a 3-panel figure: ours vs K-means vs
     SpoNet.

7. **Compile paper artifacts manifest**
   (`results/stage7/manifest.md`): a markdown file mapping each
   manuscript figure number to its source script and output file. This
   is invaluable when revisions ask "regenerate Figure 3 with X
   change".

## Acceptance Criteria

- [ ] All 8 figures exist as both PDF and PNG.
- [ ] All 4 main CSV tables exist and are non-empty.
- [ ] Wilcoxon p-value for (A7 vs A0/A1/A2) is < 0.01.
- [ ] Wilcoxon p-value for (A7 vs A4 SpoNet) is < 0.05 (otherwise
  the paper's novelty story is weaker; consider revising or adding
  more seeds).
- [ ] Generalization tables show A7 ≥ A5 on time-split (otherwise C1
  innovation is unsupported; revisit).
- [ ] `manifest.md` exists and is up to date.
- [ ] Every figure script can be re-run end-to-end without errors.

## Files to Create

- `src/eval/__init__.py`
- `src/eval/evaluator.py`
- `src/eval/perturbations.py` — demand shift utilities.
- `experiments/run_stage7_main_table.py`
- `experiments/run_stage7_stats.py`
- `experiments/run_stage7_time_split.py`
- `experiments/run_stage7_demand_shift.py`
- `experiments/run_stage7_cross_K.py`
- `experiments/fig_*.py` (8 figure scripts)
- `tests/test_evaluator.py`

## Common Pitfalls

- **Cherry-picking seeds**: don't quietly drop bad seeds. If a seed
  fails, document why in `decisions.md` and report results either
  with all seeds or with a clear exclusion criterion (e.g. "training
  diverged in 1 of 5 seeds, excluded").
- **Figure consistency**: use a fixed color palette across all figures
  (method → color). Define in `src/utils/plot_style.py`.
- **Statistical test choice**: paired Wilcoxon is correct when
  comparing methods on the same seeds; unpaired t-test is wrong here.
- **Coverage rate definition**: paper text must define exactly which
  coverage (count, fare-weighted, time-weighted) is being reported in
  each table. Pick one as "main" and put others in supplementary.
- **Time-split data leakage**: if you trained on all 7 days in Stage 6
  and then re-evaluate on days 6–7 in Stage 7 generalization, that's
  leakage. The time-split tables must come from a *retrained* model
  on days 1–5 only. Budget compute for this.
- **OOD demand shift not too crazy**: 10× demand in one zone is
  unrealistic. Stick to 2–3× max, and limit to commercially
  plausible zones.
- **Map plotting**: `folium` saves HTML, which is great for exploration
  but useless for LaTeX. Use `contextily` + `matplotlib` for static
  figures. Or use `osmnx.plot_graph_folium` then export as PNG via
  `selenium` (last resort).
- **LaTeX table formatting**: use `pandas.DataFrame.to_latex` with
  `float_format="%.3f"`, `escape=False`, `booktabs=True`. Pre-format
  numbers before exporting.

## Out of Scope

The actual manuscript writing is not in this plan. After this stage
completes, you should have everything needed to write Sections 4
(Methods), 5 (Case Study), and 6 (Discussion) of the paper. Section 1
(Intro), 2 (Related Work), and 7 (Conclusion) are pure writing tasks
done outside Claude Code.

## Dependencies

- Stages 1–6.
- Adds: `scipy.stats`, `contextily`, `seaborn`.

## Estimated effort

5–7 days, depending on how many generalization experiments are run
and how polished the figures need to be.