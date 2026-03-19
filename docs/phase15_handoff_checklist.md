# Phase 1.5 Handoff Checklist

## Local Readiness

1. Smoke tests pass.
2. `experiments/generate_phase15_manifest.py` runs locally.
3. `experiments/run_manifest_row.py` runs at least one row locally.
4. `experiments/summarize_phase15.py` produces grouped summary CSVs.
5. `experiments/plot_metrics.py` can render plots from `phase15_per_run_results.csv`.

## Manifest Readiness

1. A manifest exists under `outputs/manifests/phase15_*.csv`.
2. The manifest has valid `planner_label`, `env_config`, `seed`, `run_id`, and `output_subdir`.
3. The manifest row count is known before submission.
4. The chosen shard count `NUM_TASKS` is recorded for the Myriad run.

## Myriad Job Readiness

1. `jobs/myriad/job_phase15_generate_manifest.sh` is configured with the correct Python activation block.
2. `jobs/myriad/job_phase15_manifest_array.sh` has the correct manifest path.
3. `NUM_TASKS` matches the `qsub -t 1-N -tc N` submission settings.
4. `jobs/myriad/job_phase15_summarize.sh` points to the same manifest.
5. `jobs/myriad/job_phase15_small_sanity.sh` has been reviewed for quick preflight runs.
6. `outputs/myriad_logs/` exists before `qsub`.

## Push Guidance

Recommended to push:
- `configs/`
- `core/`
- `planners/`
- `experiments/`
- `jobs/myriad/`
- `docs/`
- `README.md`
- `README_phase15.md`
- `tests/`

Do not push:
- `outputs/`
- large mp4 artifacts
- generated benchmark CSVs unless intentionally curated

## Minimal Myriad Order

1. `git clone` or `git pull`
2. edit Python environment lines in `jobs/myriad/*.sh`
3. `mkdir -p outputs/myriad_logs`
4. `qsub jobs/myriad/job_phase15_generate_manifest.sh`
5. inspect printed manifest path and row count
6. choose shard count, for example `NUM_TASKS=36`
7. `qsub -t 1-N -tc N -v NUM_TASKS=N,MANIFEST_PATH=<manifest_csv> jobs/myriad/job_phase15_manifest_array.sh`
8. `qsub -hold_jid <array_job_id> -v MANIFEST_PATH=<manifest_csv> jobs/myriad/job_phase15_summarize.sh`
9. inspect grouped CSVs and plots under `outputs/benchmarks/<phase15_subdir>/phase15_summary/`

## Final Sanity Before Cloud

1. `git status` contains only intended source/config/doc changes.
2. `.gitignore` still excludes `outputs/`.
3. The recommended Phase 1 calibration config is identifiable from `phase15_best_configs.csv`.
