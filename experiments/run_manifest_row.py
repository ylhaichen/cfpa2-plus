from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_single_experiment import build_runtime_override, execute_single_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Phase 1.5 calibration task from a manifest row")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--row-index", type=int, required=True, help="Zero-based row index in manifest CSV")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--override-yaml", action="append", default=[])
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    return parser.parse_args()


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row.index:
        return default
    value = row[key]
    return default if _is_missing(value) else value


def _override_pairs_from_row(row: pd.Series) -> list[str]:
    pairs = [
        f"planning.planner_label={_row_value(row, 'planner_label')}",
        f"planning.cfpa2_plus.score_mode.execution_weight={float(_row_value(row, 'execution_weight', 0.75))}",
        f"planning.cfpa2_plus.execution.w_clearance={float(_row_value(row, 'w_clearance', 1.0))}",
        f"planning.cfpa2_plus.execution.w_density={float(_row_value(row, 'w_density', 0.7))}",
        f"planning.cfpa2_plus.execution.w_turn={float(_row_value(row, 'w_turn', 0.5))}",
        f"planning.cfpa2_plus.execution.w_narrow={float(_row_value(row, 'w_narrow', 0.75))}",
        f"planning.cfpa2_plus.execution.w_team={float(_row_value(row, 'w_team', 0.6))}",
        f"planning.cfpa2_plus.execution.w_slowdown={float(_row_value(row, 'w_slowdown', 0.4))}",
        f"planning.cfpa2_plus.execution.normalization_mode={_row_value(row, 'normalization_mode', 'linear')}",
        f"planning.cfpa2_plus.execution.feature_clip_max={float(_row_value(row, 'feature_clip_max', 0.85))}",
        f"planning.cfpa2_plus.execution.total_clip_max={float(_row_value(row, 'total_clip_max', 0.85))}",
        f"planning.cfpa2_plus.execution.soft_saturation_gamma={float(_row_value(row, 'soft_saturation_gamma', 1.35))}",
    ]
    return pairs


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if args.row_index < 0 or args.row_index >= len(manifest_df):
        raise IndexError(f"row_index={args.row_index} out of range for manifest with {len(manifest_df)} rows")

    row = manifest_df.iloc[int(args.row_index)]
    run_id = str(_row_value(row, "run_id"))
    summary_csv = Path(args.output_root) / "benchmarks" / run_id / "results_csv" / "single_run_summary.csv"
    if args.skip_existing and summary_csv.exists():
        print(f"skip_existing=1 summary_csv={summary_csv}")
        return

    extra_override = build_runtime_override(
        override_yaml_paths=args.override_yaml,
        set_values=_override_pairs_from_row(row) + list(args.set_values),
        max_steps=int(_row_value(row, "max_steps", 80)),
        disable_animation=True,
        physics_weight_file=None,
    )

    result_row = execute_single_run(
        base_config=args.base_config,
        planner_choice=str(_row_value(row, "planner_choice", "cfpa2_plus_phase1_calib_base")),
        planner_config=str(_row_value(row, "planner_config", "configs/planner_cfpa2_plus_phase1_calib_base.yaml")),
        env_config=str(_row_value(row, "env_config")),
        seed=int(_row_value(row, "seed", 0)),
        run_id=run_id,
        output_root=args.output_root,
        extra_override=extra_override,
        metadata_extra={
            "manifest_path": str(manifest_path),
            "manifest_row_index": int(args.row_index),
            "run_group": str(_row_value(row, "run_group", "")),
        },
        summary_extra={
            "manifest_path": str(manifest_path),
            "manifest_row_index": int(args.row_index),
            "run_group": str(_row_value(row, "run_group", "")),
            "output_subdir": str(_row_value(row, "output_subdir", "")),
            "normalization_mode": str(_row_value(row, "normalization_mode", "linear")),
            "execution_weight": float(_row_value(row, "execution_weight", 0.75)),
            "w_clearance": float(_row_value(row, "w_clearance", 1.0)),
            "w_density": float(_row_value(row, "w_density", 0.7)),
            "w_turn": float(_row_value(row, "w_turn", 0.5)),
            "w_narrow": float(_row_value(row, "w_narrow", 0.75)),
            "w_team": float(_row_value(row, "w_team", 0.6)),
            "w_slowdown": float(_row_value(row, "w_slowdown", 0.4)),
            "feature_clip_max": float(_row_value(row, "feature_clip_max", 0.85)),
            "total_clip_max": float(_row_value(row, "total_clip_max", 0.85)),
            "soft_saturation_gamma": float(_row_value(row, "soft_saturation_gamma", 1.35)),
        },
    )

    print("=== Manifest Row Summary ===")
    print(f"manifest_csv: {manifest_path}")
    print(f"row_index: {args.row_index}")
    print(f"planner_label: {result_row['planner_label']}")
    print(f"map_family: {result_row.get('map_family', 'unknown')}")
    print(f"summary_csv: {result_row['summary_csv']}")


if __name__ == "__main__":
    main()
