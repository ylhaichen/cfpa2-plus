# Myriad Phase 1.5 Guide

This folder contains **UCL Myriad / SGE** scripts for Phase 1.5 calibration of `cfpa2_plus_phase1`.

These scripts now use the same stable **shard-style array pattern** as the existing working Myriad jobs in this repo:
- pass `REPO_DIR` explicitly
- pass `OUTPUT_ROOT` explicitly
- pass log paths with `qsub -o/-e`
- optionally pass `CONDA_SH` and `CONDA_ENV`
- or reuse an existing venv via `VENV_PATH`
- use `NUM_TASKS` shards rather than one SGE task per manifest row

## 1. Clone And Prepare

```bash
git clone <your-repo-url>
cd cfpa2-plus
export REPO_DIR="$PWD"
export OUTPUT_ROOT="$REPO_DIR/outputs"
mkdir -p "$OUTPUT_ROOT/myriad_logs"
```

## 2. Python Environment

Choose one:
- pass `CONDA_SH` and `CONDA_ENV`
- pass `VENV_PATH`
- or submit from an already activated environment and omit both

Conda example:

```bash
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV="cfpa2rh"
```

Venv example:

```bash
export VENV_PATH="$HOME/venvs/cfpa2rh"
```

## 3. Generate A Manifest

Small sanity:

```bash
qsub \
  -o "$OUTPUT_ROOT/myriad_logs" \
  -e "$OUTPUT_ROOT/myriad_logs" \
  -v REPO_DIR="$REPO_DIR",OUTPUT_ROOT="$OUTPUT_ROOT",CONDA_SH="${CONDA_SH:-}",CONDA_ENV="${CONDA_ENV:-}",VENV_PATH="${VENV_PATH:-}" \
  jobs/myriad/job_phase15_small_sanity.sh
```

Full manifest generation:

```bash
qsub \
  -o "$OUTPUT_ROOT/myriad_logs" \
  -e "$OUTPUT_ROOT/myriad_logs" \
  -v REPO_DIR="$REPO_DIR",OUTPUT_ROOT="$OUTPUT_ROOT",CONDA_SH="${CONDA_SH:-}",CONDA_ENV="${CONDA_ENV:-}",VENV_PATH="${VENV_PATH:-}",MANIFEST_PROFILE=full,MANIFEST_TAG=phase15_calib_a \
  jobs/myriad/job_phase15_generate_manifest.sh
```

The job prints:
- manifest path
- row count
- output subdir

## 4. Submit The Array Job

After manifest generation, inspect the real manifest filename and submit a **shard-style** array job.

Recommended starting point:
- `NUM_TASKS=36`
- `-t 1-36`
- `-tc 36`

Example:

```bash
qsub \
  -o "$OUTPUT_ROOT/myriad_logs" \
  -e "$OUTPUT_ROOT/myriad_logs" \
  -t 1-36 \
  -tc 36 \
  -v REPO_DIR="$REPO_DIR",OUTPUT_ROOT="$OUTPUT_ROOT",NUM_TASKS=36,CONDA_SH="${CONDA_SH:-}",CONDA_ENV="${CONDA_ENV:-}",VENV_PATH="${VENV_PATH:-}",MANIFEST_PATH="$OUTPUT_ROOT/manifests/phase15_full_<tag>.csv" \
  jobs/myriad/job_phase15_manifest_array.sh
```

More aggressive option:

```bash
qsub \
  -o "$OUTPUT_ROOT/myriad_logs" \
  -e "$OUTPUT_ROOT/myriad_logs" \
  -t 1-72 \
  -tc 72 \
  -v REPO_DIR="$REPO_DIR",OUTPUT_ROOT="$OUTPUT_ROOT",NUM_TASKS=72,CONDA_SH="${CONDA_SH:-}",CONDA_ENV="${CONDA_ENV:-}",VENV_PATH="${VENV_PATH:-}",MANIFEST_PATH="$OUTPUT_ROOT/manifests/phase15_full_<tag>.csv" \
  jobs/myriad/job_phase15_manifest_array.sh
```

The script maps work like this:
- task `0` runs rows `0, NUM_TASKS, 2*NUM_TASKS, ...`
- task `1` runs rows `1, NUM_TASKS+1, 2*NUM_TASKS+1, ...`
- and so on

This is usually much more scheduler-friendly on Myriad than submitting one array task per manifest row.

## 5. Summarize And Plot

After the array job completes:

```bash
qsub \
  -o "$OUTPUT_ROOT/myriad_logs" \
  -e "$OUTPUT_ROOT/myriad_logs" \
  -hold_jid <array_job_id> \
  -v REPO_DIR="$REPO_DIR",OUTPUT_ROOT="$OUTPUT_ROOT",CONDA_SH="${CONDA_SH:-}",CONDA_ENV="${CONDA_ENV:-}",VENV_PATH="${VENV_PATH:-}",MANIFEST_PATH="$OUTPUT_ROOT/manifests/phase15_full_<tag>.csv" \
  jobs/myriad/job_phase15_summarize.sh
```

This writes:
- `phase15_per_run_results.csv`
- `phase15_grouped_summary.csv`
- `phase15_best_configs.csv`
- `phase15_feature_redundancy_summary.csv`
- plots under `"$OUTPUT_ROOT/benchmarks/<phase15_subdir>/phase15_summary/plots/"`

## 6. Files Written By This Workflow

Manifest:
- `"$OUTPUT_ROOT/manifests/phase15_*.csv"`

Array stdout/stderr:
- `"$OUTPUT_ROOT/myriad_logs/"`

Benchmark outputs:
- `"$OUTPUT_ROOT/benchmarks/phase15_*/"`

## 7. Minimal Myriad Order

1. clone or pull repo
2. set `REPO_DIR` and `OUTPUT_ROOT`
3. optionally set `CONDA_SH/CONDA_ENV` or `VENV_PATH`
4. `mkdir -p "$OUTPUT_ROOT/myriad_logs"`
5. submit `job_phase15_generate_manifest.sh`
6. submit `job_phase15_manifest_array.sh` with the correct `MANIFEST_PATH`, `NUM_TASKS`, and matching `-t/-tc`
7. submit `job_phase15_summarize.sh` with `-hold_jid`
8. inspect grouped CSVs and plots under `"$OUTPUT_ROOT"`
