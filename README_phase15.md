# Phase 1.5 Calibration And Myriad Handoff

This document covers the **Phase 1.5 calibration** workflow for `cfpa2_plus_phase1`.

Scope:
- calibrate existing execution-aware weights
- compare normalization / clipping modes
- diagnose feature redundancy
- run locally or on **UCL Myriad** with manifest-driven CPU array jobs

Out of scope:
- Phase 2 failure-aware terms
- changes to `rh_cfpa2` or `physics_rh_cfpa2`
- learned score components

## Local Workflow

1. Create a Python environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run smoke tests.

```bash
PYTHONPATH=. pytest -q tests/test_execution_fidelity_service.py tests/test_cfpa2_plus_phase1_smoke.py
```

3. Generate a small sanity manifest.

```bash
MPLCONFIGDIR=/tmp/mpl_phase15 PYTHONPATH=. python experiments/generate_phase15_manifest.py \
  --profile small_sanity
```

4. Run one manifest row.

```bash
MPLCONFIGDIR=/tmp/mpl_phase15 PYTHONPATH=. python experiments/run_manifest_row.py \
  --manifest outputs/manifests/<manifest>.csv \
  --row-index 0 \
  --skip-existing
```

5. Summarize finished rows.

```bash
PYTHONPATH=. python experiments/summarize_phase15.py \
  --manifest outputs/manifests/<manifest>.csv
```

6. Plot results from the summarized per-run CSV.

```bash
MPLCONFIGDIR=/tmp/mpl_phase15 PYTHONPATH=. python experiments/plot_metrics.py \
  --input outputs/benchmarks/<phase15_subdir>/phase15_summary/results_csv/phase15_per_run_results.csv \
  --group-by map_family normalization_mode run_group
```

## Manifest Layout

`experiments/generate_phase15_manifest.py` writes a CSV under `outputs/manifests/`.

Each row contains:
- `planner_label`
- `env_config`
- `seed`
- `execution_weight`
- `w_clearance`
- `w_density`
- `w_turn`
- `w_narrow`
- `w_team`
- `w_slowdown`
- `normalization_mode`
- `run_group`
- `output_subdir`
- `run_id`

The default full manifest includes three experiment groups:
- `weight_sweep`
- `normalization_compare`
- `redundancy_diagnosis`

## Phase 1.5 Result Files

Manifest:
- `outputs/manifests/phase15_*.csv`

Per-row runs:
- `outputs/benchmarks/<output_subdir>/row_*/...`

Summaries:
- `outputs/benchmarks/<output_subdir>/phase15_summary/results_csv/phase15_per_run_results.csv`
- `outputs/benchmarks/<output_subdir>/phase15_summary/results_csv/phase15_grouped_summary.csv`
- `outputs/benchmarks/<output_subdir>/phase15_summary/results_csv/phase15_best_configs.csv`
- `outputs/benchmarks/<output_subdir>/phase15_summary/results_csv/phase15_feature_redundancy_summary.csv`

Plots:
- `outputs/benchmarks/<output_subdir>/phase15_summary/plots/`

## Myriad Workflow

Use the Myriad-specific instructions in `jobs/myriad/README.md`.

The intended sequence is:
1. clone or pull the repo on Myriad
2. edit Python environment activation lines in `jobs/myriad/*.sh`
3. submit `job_phase15_generate_manifest.sh`
4. submit `job_phase15_manifest_array.sh` with shard-style `NUM_TASKS`
5. submit `job_phase15_summarize.sh`

## GitHub Push Guidance

Safe to push:
- source code
- configs
- docs
- `jobs/myriad/`

Do not push:
- `outputs/`
- rendered videos
- large benchmark artifacts

`outputs/` is already ignored by `.gitignore`.
