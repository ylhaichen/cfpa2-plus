# Myriad Phase 1.5 Guide

This folder contains **UCL Myriad / SGE** scripts for Phase 1.5 calibration of `cfpa2_plus_phase1`.

## 1. Clone And Prepare

```bash
git clone <your-repo-url>
cd cfpa2-rh-physics-exploration
mkdir -p outputs/myriad_logs
```

## 2. Python Environment

Each script has two editable sections near the top:

- Option A: module-based Python
- Option B: user-managed conda or venv

You must edit one of these strategies before large runs.

Typical module-based path:
- keep `module purge`
- keep `module load python/...`
- leave `PHASE15_USE_MODULE_PYTHON=1`

Typical conda-based path:
- set `PHASE15_USE_MODULE_PYTHON=0`
- set `PHASE15_USE_CUSTOM_PYTHON=1`
- uncomment `source ~/miniconda3/etc/profile.d/conda.sh`
- set the correct env name

## 3. Generate A Manifest

Small sanity:

```bash
qsub jobs/myriad/job_phase15_small_sanity.sh
```

Full manifest generation:

```bash
qsub -v MANIFEST_PROFILE=full,MANIFEST_TAG=phase15_calib_a jobs/myriad/job_phase15_generate_manifest.sh
```

The job prints:
- manifest path
- row count
- output subdir

## 4. Submit The Array Job

After manifest generation, inspect the row count and submit an array job.

Example:

```bash
qsub -t 1-1053 -v MANIFEST_PATH=outputs/manifests/phase15_full_<tag>.csv jobs/myriad/job_phase15_manifest_array.sh
```

The script maps:
- `SGE_TASK_ID=1` -> manifest row `0`
- `SGE_TASK_ID=N` -> manifest row `N-1`

## 5. Summarize And Plot

After the array job completes:

```bash
qsub -hold_jid <array_job_id> -v MANIFEST_PATH=outputs/manifests/phase15_full_<tag>.csv jobs/myriad/job_phase15_summarize.sh
```

This writes:
- `phase15_per_run_results.csv`
- `phase15_grouped_summary.csv`
- `phase15_best_configs.csv`
- `phase15_feature_redundancy_summary.csv`
- plots under `outputs/benchmarks/<phase15_subdir>/phase15_summary/plots/`

## 6. Files Written By This Workflow

Manifest:
- `outputs/manifests/phase15_*.csv`

Array stdout/stderr:
- `outputs/myriad_logs/`

Benchmark outputs:
- `outputs/benchmarks/phase15_*/`

## 7. Minimal Myriad Order

1. clone or pull repo
2. edit Python activation lines in the scripts
3. `mkdir -p outputs/myriad_logs`
4. `qsub jobs/myriad/job_phase15_generate_manifest.sh`
5. submit `job_phase15_manifest_array.sh` with the correct `-t` range and `MANIFEST_PATH`
6. submit `job_phase15_summarize.sh` with `-hold_jid`
7. inspect grouped CSVs and plots
