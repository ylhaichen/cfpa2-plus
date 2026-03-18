from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import build_override_from_pairs, combine_overrides, load_experiment_config, load_override_yaml, write_config_snapshot
from core.preset_registry import get_planner_preset, planner_preset_choices
from experiments.common import enforce_mp4_only, git_commit_hash, make_run_id, prepare_output_dirs, save_run_metadata
from simulators.grid_sim import GridSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one planner episode in unified framework")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--planner-config", type=str, default=None)
    parser.add_argument("--env-config", type=str, default="configs/env_maze.yaml")
    parser.add_argument("--planner", type=str, default="cfpa2", choices=planner_preset_choices())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--disable-animation", action="store_true")
    parser.add_argument("--physics-weight-file", type=str, default=None)
    parser.add_argument("--override-yaml", action="append", default=[], help="Extra override YAML merged after base/planner/env configs")
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="Dotted KEY=VALUE override, repeatable")
    return parser.parse_args()


def build_runtime_override(
    override_yaml_paths: list[str] | None = None,
    set_values: list[str] | None = None,
    max_steps: int | None = None,
    disable_animation: bool = False,
    physics_weight_file: str | None = None,
) -> dict[str, Any]:
    yaml_overrides = [load_override_yaml(path) for path in (override_yaml_paths or [])]
    cli_override = build_override_from_pairs(set_values)
    runtime_override: dict[str, Any] = {}

    if max_steps is not None:
        runtime_override = combine_overrides(runtime_override, {"termination": {"max_steps": int(max_steps)}})
    if disable_animation:
        runtime_override = combine_overrides(runtime_override, {"experiment": {"save_animation": False}})
    if physics_weight_file is not None:
        runtime_override = combine_overrides(
            runtime_override,
            {
                "predictor": {
                    "type": "physics_residual",
                    "physics_residual": {
                        "enabled": True,
                        "weight_file": physics_weight_file,
                    },
                }
            },
        )

    return combine_overrides(*yaml_overrides, cli_override, runtime_override)


def execute_single_run(
    *,
    base_config: str,
    planner_choice: str,
    planner_config: str | None,
    env_config: str,
    seed: int,
    run_id: str | None,
    output_root: str,
    extra_override: dict[str, Any] | None = None,
    physics_weight_file: str | None = None,
    metadata_extra: dict[str, Any] | None = None,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    planner_preset = get_planner_preset(planner_choice)
    planner_cfg = planner_config or planner_preset.config_path
    cfg = load_experiment_config(
        base_config,
        planner_cfg_path=planner_cfg,
        env_cfg_path=env_config,
        extra_override=extra_override,
    )
    cfg = enforce_mp4_only(cfg)

    planning_cfg = dict(cfg.get("planning", {}))
    planning_cfg["planner_name"] = planner_preset.planner_name
    planning_cfg["planner_label"] = str(planning_cfg.get("planner_label", planner_preset.planner_label))
    cfg["planning"] = planning_cfg
    planner_label = str(planning_cfg["planner_label"])

    if planner_preset.planner_name == "physics_rh_cfpa2" and physics_weight_file is not None:
        cfg["predictor"]["type"] = "physics_residual"
        cfg["predictor"]["physics_residual"]["enabled"] = True
        cfg["predictor"]["physics_residual"]["weight_file"] = physics_weight_file

    run_id = run_id or make_run_id("single")
    dirs = prepare_output_dirs(output_root, run_id)

    write_config_snapshot(dirs["configs"] / "resolved_config.yaml", cfg)
    save_run_metadata(
        dirs["metadata"] / "run_metadata.json",
        {
            "run_id": run_id,
            "planner": planner_label,
            "planner_base_name": planner_preset.planner_name,
            "planner_choice": planner_choice,
            "env_config": env_config,
            "planner_config": planner_cfg,
            "seed": seed,
            "git_commit": git_commit_hash(),
            **(metadata_extra or {}),
        },
    )

    sim = GridSimulation()
    map_name = cfg["environment"].get("map_name", cfg["environment"].get("map_type", "map"))
    map_family = cfg["environment"].get("map_family", map_name)
    stem = f"{planner_label}_{map_name}_seed{seed}"
    episode_dir = dirs["episode"] / stem

    result = sim.run_episode(
        cfg=cfg,
        planner_name=planner_preset.planner_name,
        seed=seed,
        output_dir=episode_dir,
        animation_stem=stem,
    )

    row = dict(result.summary)
    row.update(
        {
            "run_id": run_id,
            "planner_label": planner_label,
            "planner_base_name": planner_preset.planner_name,
            "planner_choice": planner_choice,
            "env_config": env_config,
            "planner_config": planner_cfg,
            "map_family": map_family,
            "coverage_csv": result.coverage_csv_path,
            "step_log_csv": result.step_log_csv_path,
            "animation_gif": result.animation_gif_path,
            "animation_mp4": result.animation_mp4_path,
            **(summary_extra or {}),
        }
    )

    summary_csv = dirs["results_csv"] / "single_run_summary.csv"
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    row["summary_csv"] = str(summary_csv)
    row["resolved_config"] = str(dirs["configs"] / "resolved_config.yaml")
    return row

def main() -> None:
    args = parse_args()
    extra_override = build_runtime_override(
        override_yaml_paths=args.override_yaml,
        set_values=args.set_values,
        max_steps=args.max_steps,
        disable_animation=args.disable_animation,
        physics_weight_file=args.physics_weight_file,
    )
    row = execute_single_run(
        base_config=args.base_config,
        planner_choice=args.planner,
        planner_config=args.planner_config,
        env_config=args.env_config,
        seed=args.seed,
        run_id=args.run_id,
        output_root=args.output_root,
        extra_override=extra_override,
        physics_weight_file=args.physics_weight_file,
        metadata_extra={
            "override_yaml": list(args.override_yaml),
            "set_values": list(args.set_values),
        },
    )

    print("=== Single Run Summary ===")
    for k, v in row.items():
        print(f"{k}: {v}")
    print(f"summary_csv: {row['summary_csv']}")


if __name__ == "__main__":
    main()
