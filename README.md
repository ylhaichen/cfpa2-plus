# CFPA-2 Demo

Centralized Complementary Frontier Pair Allocation (CFPA-2) demo for multi-robot frontier exploration in unknown 2D occupancy-grid environments.

This repository focuses on **high-level task allocation and exploration strategy** for one or two robots. It intentionally excludes SLAM and low-level dynamics so you can evaluate exploration policies directly.

## Highlights

- Shared occupancy grid with `UNKNOWN=-1`, `FREE=0`, `OCCUPIED=1`
- Frontier detection and BFS clustering
- A* path planning on known-free cells
- Three modes:
  - `single` (single robot greedy)
  - `dual_greedy` (two robots, independent greedy baseline)
  - `dual_joint` (CFPA-2 joint complementary assignment)
- Event-driven replanning with hysteresis/reservation hooks
- Live Matplotlib visualization
- GIF/MP4 run recording
- Reproducible experiments by seed with CSV/figure outputs

## Method Context

The implementation is inspired by:

- **Yamauchi** frontier-based exploration
- **Burgard et al.** coordinated utility/cost-based multi-robot exploration
- **Keidar & Kaminka** stale-frontier / fast frontier-update motivation (WFD/FFD line)

## Core Formulation

Single-robot utility:

\[
U(r,f)=w_{ig}IG(f)-w_cC(r,f)-w_{sw}SwitchPenalty(r,f)
\]

Two-robot joint score (CFPA-2):

\[
J(f_i,f_j)=U(r_1,f_i)+U(r_2,f_j)-\lambda O(f_i,f_j)-\mu I(f_i,f_j)
\]

Implemented overlap term:

\[
O(f_i,f_j)=\exp\left(-\frac{\|p_i-p_j\|^2}{2\sigma^2}\right)
\]

In v1, path interference term `I` is a stub (`0.0`).

## What "Frontier Reps" Means

`frontier reps` in the plot are **frontier representatives**:

- Raw frontier cells are first clustered.
- Each cluster is reduced to one representative point.
- Assignment/planning uses these representatives as target candidates.

This keeps planning stable and avoids noisy per-cell targeting.

## Repository Layout

```text
.
├── main.py                        # Root launcher
├── experiments/
│   ├── run_compare.py             # Root comparison launcher
│   └── summarize_results.py       # Root summary launcher
├── cfpa2_demo/
│   ├── main.py
│   ├── config/
│   ├── core/
│   ├── maps/
│   ├── viz/
│   ├── experiments/
│   └── tests/
├── outputs/
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the default live demo:

```bash
MPLCONFIGDIR=/tmp/matplotlib python main.py --config config/default.yaml --mode dual_joint
```

Run without live window:

```bash
MPLCONFIGDIR=/tmp/matplotlib python main.py --config config/default.yaml --mode dual_joint --no-viz
```

Save GIF:

```bash
MPLCONFIGDIR=/tmp/matplotlib python main.py --mode dual_joint --save-animation --no-viz
```

Save video (MP4):

```bash
MPLCONFIGDIR=/tmp/matplotlib python main.py --mode dual_joint --save-video --no-viz
```

## Experiment Runs

Run mode comparison sweeps:

```bash
MPLCONFIGDIR=/tmp/matplotlib python experiments/run_compare.py
```

Aggregate existing comparison CSV:

```bash
MPLCONFIGDIR=/tmp/matplotlib python experiments/summarize_results.py --input outputs/results_csv/compare_results.csv
```

## Configuration

Primary config: `cfpa2_demo/config/default.yaml`

Key knobs:

- `environment`: map size/type/density/seed
- `robots`: starts, sensing radius, LOS
- `frontier`: clustering and frontier-rep count controls
- `allocator`: overlap penalty and stability settings
- `replanning`: event triggers and periodic backup
- `termination`: coverage threshold and max steps
- `visualization`: live plot, GIF/video saving, fps

Map presets:

- `config/map_open.yaml`
- `config/map_rooms.yaml`
- `config/map_maze.yaml`

## Output Artifacts

Generated under `outputs/`:

- `outputs/results_csv/*.csv`
- `outputs/figures/*.png`
- `outputs/animations/*.gif` or `*.mp4`

Metrics include:

- completion steps
- coverage curve
- replan count/reasons
- repeated-coverage ratio
- conflict count (dual greedy baseline)
- frontier count statistics

## Assumptions and Limitations

- Perfect localization/map fusion (no SLAM drift)
- Grid-world motion (1-cell step, no dynamics)
- Centralized planner
- Currently optimized for 1–2 robots

## Extension Hooks

- Non-zero path interference model
- N-robot assignment strategies (Hungarian/Auction/etc.)
- Richer sensor models and communication constraints

---

Implementation details and module-level docs are also in:
- [`cfpa2_demo/README.md`](cfpa2_demo/README.md)
