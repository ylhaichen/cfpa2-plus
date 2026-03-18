from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml
from experiments.common import make_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manifest-driven Phase 1.5 calibration experiments")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--manifest-tag", type=str, default=None)
    parser.add_argument("--profile", choices=["full", "small_sanity"], default="full")
    parser.add_argument("--planner-choice", type=str, default="cfpa2_plus_phase1_calib_base")
    parser.add_argument("--planner-config", type=str, default="configs/planner_cfpa2_plus_phase1_calib_base.yaml")
    parser.add_argument(
        "--env-configs",
        nargs="+",
        default=[
            "configs/env_go2w_like.yaml",
            "configs/env_branching_deadend.yaml",
            "configs/env_narrow_t_branches.yaml",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--execution-weights", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--w-narrow", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--w-team", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--w-slowdown", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--normalization-modes", nargs="+", default=["linear", "feature_clipped", "total_clipped"])
    parser.add_argument("--max-steps", type=int, default=80)
    return parser.parse_args()


def _float_slug(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _env_meta(path: str) -> tuple[str, str]:
    env_cfg = load_yaml(path).get("environment", {})
    map_name = str(env_cfg.get("map_name", env_cfg.get("map_type", Path(path).stem)))
    map_family = str(env_cfg.get("map_family", map_name))
    return map_name, map_family


def _planner_label(run_group: str, normalization_mode: str, execution_weight: float, w_narrow: float, w_team: float, w_slowdown: float) -> str:
    abbrev = {
        "weight_sweep": "ws",
        "normalization_compare": "norm",
        "redundancy_diagnosis": "red",
    }.get(run_group, run_group)
    return (
        f"cfpa2p15_{abbrev}"
        f"__norm-{normalization_mode}"
        f"__ew{_float_slug(execution_weight)}"
        f"__nar{_float_slug(w_narrow)}"
        f"__team{_float_slug(w_team)}"
        f"__slow{_float_slug(w_slowdown)}"
    )


def _base_execution_defaults(planner_config: str) -> dict[str, float | str]:
    cfg = load_yaml(planner_config).get("planning", {}).get("cfpa2_plus", {}).get("execution", {})
    return {
        "w_clearance": float(cfg.get("w_clearance", 1.0)),
        "w_density": float(cfg.get("w_density", 0.7)),
        "w_turn": float(cfg.get("w_turn", 0.5)),
        "feature_clip_max": float(cfg.get("feature_clip_max", 0.85)),
        "total_clip_max": float(cfg.get("total_clip_max", 0.85)),
        "soft_saturation_gamma": float(cfg.get("soft_saturation_gamma", 1.35)),
        "normalization_mode": str(cfg.get("normalization_mode", "linear")),
    }


def _append_row(
    rows: list[dict],
    *,
    manifest_tag: str,
    output_subdir: str,
    planner_choice: str,
    planner_config: str,
    env_config: str,
    seed: int,
    max_steps: int,
    run_group: str,
    normalization_mode: str,
    execution_weight: float,
    w_clearance: float,
    w_density: float,
    w_turn: float,
    w_narrow: float,
    w_team: float,
    w_slowdown: float,
    feature_clip_max: float,
    total_clip_max: float,
    soft_saturation_gamma: float,
) -> None:
    row_index = len(rows)
    map_name, map_family = _env_meta(env_config)
    planner_label = _planner_label(run_group, normalization_mode, execution_weight, w_narrow, w_team, w_slowdown)
    run_id = f"{output_subdir}/row_{row_index:05d}_{Path(env_config).stem}_s{seed}_{planner_label}"

    rows.append(
        {
            "row_index": row_index,
            "manifest_tag": manifest_tag,
            "planner_choice": planner_choice,
            "planner_config": planner_config,
            "planner_label": planner_label,
            "env_config": env_config,
            "map_name": map_name,
            "map_family": map_family,
            "seed": int(seed),
            "max_steps": int(max_steps),
            "execution_weight": float(execution_weight),
            "w_clearance": float(w_clearance),
            "w_density": float(w_density),
            "w_turn": float(w_turn),
            "w_narrow": float(w_narrow),
            "w_team": float(w_team),
            "w_slowdown": float(w_slowdown),
            "normalization_mode": normalization_mode,
            "feature_clip_max": float(feature_clip_max),
            "total_clip_max": float(total_clip_max),
            "soft_saturation_gamma": float(soft_saturation_gamma),
            "run_group": run_group,
            "output_subdir": output_subdir,
            "run_id": run_id,
        }
    )


def _build_rows(args: argparse.Namespace) -> list[dict]:
    defaults = _base_execution_defaults(args.planner_config)
    manifest_tag = args.manifest_tag or make_run_id("phase15").replace("phase15_", "")
    output_subdir = f"phase15_{manifest_tag}"

    rows: list[dict] = []
    env_configs = list(args.env_configs)
    seeds = list(args.seeds)

    if args.profile == "small_sanity":
        env_configs = env_configs[:2]
        seeds = seeds[:1]
        small_rows = [
            ("weight_sweep", "linear", 0.75, 0.75, 0.60, 0.40),
            ("normalization_compare", "feature_clipped", 0.75, 0.75, 0.60, 0.40),
            ("redundancy_diagnosis", "linear", 0.75, 0.75, 0.60, 0.00),
            ("redundancy_diagnosis", "linear", 0.75, 0.00, 0.60, 0.40),
        ]
        for env_config, seed, spec in itertools.product(env_configs, seeds, small_rows):
            run_group, normalization_mode, execution_weight, w_narrow, w_team, w_slowdown = spec
            _append_row(
                rows,
                manifest_tag=manifest_tag,
                output_subdir=output_subdir,
                planner_choice=args.planner_choice,
                planner_config=args.planner_config,
                env_config=env_config,
                seed=seed,
                max_steps=min(args.max_steps, 40),
                run_group=run_group,
                normalization_mode=normalization_mode,
                execution_weight=execution_weight,
                w_clearance=float(defaults["w_clearance"]),
                w_density=float(defaults["w_density"]),
                w_turn=float(defaults["w_turn"]),
                w_narrow=w_narrow,
                w_team=w_team,
                w_slowdown=w_slowdown,
                feature_clip_max=float(defaults["feature_clip_max"]),
                total_clip_max=float(defaults["total_clip_max"]),
                soft_saturation_gamma=float(defaults["soft_saturation_gamma"]),
            )
        return rows

    for env_config, seed in itertools.product(env_configs, seeds):
        for execution_weight, w_narrow, w_team, w_slowdown in itertools.product(
            args.execution_weights,
            args.w_narrow,
            args.w_team,
            args.w_slowdown,
        ):
            _append_row(
                rows,
                manifest_tag=manifest_tag,
                output_subdir=output_subdir,
                planner_choice=args.planner_choice,
                planner_config=args.planner_config,
                env_config=env_config,
                seed=seed,
                max_steps=args.max_steps,
                run_group="weight_sweep",
                normalization_mode="linear",
                execution_weight=execution_weight,
                w_clearance=float(defaults["w_clearance"]),
                w_density=float(defaults["w_density"]),
                w_turn=float(defaults["w_turn"]),
                w_narrow=w_narrow,
                w_team=w_team,
                w_slowdown=w_slowdown,
                feature_clip_max=float(defaults["feature_clip_max"]),
                total_clip_max=float(defaults["total_clip_max"]),
                soft_saturation_gamma=float(defaults["soft_saturation_gamma"]),
            )

        for normalization_mode in list(dict.fromkeys(list(args.normalization_modes) + ["soft_saturation"])):
            _append_row(
                rows,
                manifest_tag=manifest_tag,
                output_subdir=output_subdir,
                planner_choice=args.planner_choice,
                planner_config=args.planner_config,
                env_config=env_config,
                seed=seed,
                max_steps=args.max_steps,
                run_group="normalization_compare",
                normalization_mode=normalization_mode,
                execution_weight=0.75,
                w_clearance=float(defaults["w_clearance"]),
                w_density=float(defaults["w_density"]),
                w_turn=float(defaults["w_turn"]),
                w_narrow=0.75,
                w_team=0.60,
                w_slowdown=0.40,
                feature_clip_max=float(defaults["feature_clip_max"]),
                total_clip_max=float(defaults["total_clip_max"]),
                soft_saturation_gamma=float(defaults["soft_saturation_gamma"]),
            )

        redundancy_specs = [
            ("full", 0.75, 0.75, 0.60, 0.40),
            ("no_slowdown", 0.75, 0.75, 0.60, 0.00),
            ("no_narrow", 0.75, 0.00, 0.60, 0.40),
            ("no_narrow_no_slowdown", 0.75, 0.00, 0.60, 0.00),
            ("no_team", 0.75, 0.75, 0.00, 0.40),
        ]
        for _, execution_weight, w_narrow, w_team, w_slowdown in redundancy_specs:
            _append_row(
                rows,
                manifest_tag=manifest_tag,
                output_subdir=output_subdir,
                planner_choice=args.planner_choice,
                planner_config=args.planner_config,
                env_config=env_config,
                seed=seed,
                max_steps=args.max_steps,
                run_group="redundancy_diagnosis",
                normalization_mode="linear",
                execution_weight=execution_weight,
                w_clearance=float(defaults["w_clearance"]),
                w_density=float(defaults["w_density"]),
                w_turn=float(defaults["w_turn"]),
                w_narrow=w_narrow,
                w_team=w_team,
                w_slowdown=w_slowdown,
                feature_clip_max=float(defaults["feature_clip_max"]),
                total_clip_max=float(defaults["total_clip_max"]),
                soft_saturation_gamma=float(defaults["soft_saturation_gamma"]),
            )

    return rows


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    manifest_dir = Path(args.output_root) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path) if args.manifest_path else manifest_dir / f"phase15_{args.profile}_{rows[0]['manifest_tag']}.csv"

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)

    print(df.head(12).to_string(index=False))
    print(f"manifest_csv: {manifest_path}")
    print(f"row_count: {len(df)}")
    print(f"output_subdir: {df['output_subdir'].iloc[0]}")


if __name__ == "__main__":
    main()
