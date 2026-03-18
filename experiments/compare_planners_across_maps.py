from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_experiment_config, write_config_snapshot
from core.preset_registry import get_planner_preset, planner_compare_choices
from experiments.common import enforce_mp4_only, git_commit_hash, make_run_id, prepare_output_dirs, save_run_metadata
from simulators.grid_sim import GridSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CFPA2 / RH-CFPA2 / Physics-RH-CFPA2 across maps")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["cfpa2", "rh_cfpa2", "physics_rh_cfpa2"],
        choices=planner_compare_choices(),
    )
    parser.add_argument(
        "--env-configs",
        nargs="+",
        default=[
            "configs/env_maze.yaml",
            "configs/env_narrow_t_branches.yaml",
            "configs/env_narrow_t_asymmetric_branches.yaml",
            "configs/env_narrow_t_loop_branches.yaml",
        ],
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--animate-first-seed-only", action="store_true")
    parser.add_argument("--physics-weight-file", type=str, default=None, help="Optional trained physics residual checkpoint (.pt/.npz)")
    return parser.parse_args()


def _plot_summary(df: pd.DataFrame, out_dir: Path, order: list[str]) -> None:
    if df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for map_name, sub in df.groupby("map_name"):
        plt.figure(figsize=(8.0, 4.5))
        sub = sub.set_index("planner_name").reindex(order).dropna(how="all").reset_index()
        colors = ["#6D4C41", "#1976D2", "#2E7D32", "#8E24AA", "#00897B"]
        label_col = "planner_label" if "planner_label" in sub.columns else "planner_name"
        plt.bar(sub[label_col], sub["completion_steps"], color=colors[: len(sub)])
        plt.title(f"Mean Completion Steps | {map_name}")
        plt.ylabel("steps")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"completion_steps_{map_name}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8.0, 4.5))
        plt.bar(sub[label_col], sub["final_coverage"], color=["#8D6E63", "#42A5F5", "#66BB6A", "#BA68C8", "#26A69A"][: len(sub)])
        plt.title(f"Final Coverage | {map_name}")
        plt.ylim(0.0, 1.01)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"final_coverage_{map_name}.png", dpi=160)
        plt.close()

        if "execution_penalty_mean" in sub.columns:
            plt.figure(figsize=(8.0, 4.5))
            plt.bar(sub[label_col], sub["execution_penalty_mean"], color=["#795548", "#00897B", "#3949AB", "#8E24AA", "#5E35B1"][: len(sub)])
            plt.title(f"Execution Penalty Mean | {map_name}")
            plt.ylabel("penalty")
            plt.ylim(0.0, max(1.0, float(sub["execution_penalty_mean"].max()) * 1.15))
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"execution_penalty_{map_name}.png", dpi=160)
            plt.close()


def _format_table_frame(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["planner_label"] if "planner_label" in df.columns else ["planner_name"]
    cols = label_cols + [
        "success_rate",
        "completion_steps",
        "completion_time",
        "final_coverage",
        "total_path_length",
        "reassignments",
        "switches",
        "conflicts",
        "congestion",
        "planner_compute_time_ms",
        "predictor_inference_time_ms",
        "execution_penalty_mean",
        "low_progress_steps",
        "blocked_or_slow_steps_proxy",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    if "success_rate" in out.columns:
        out["success_rate"] = (out["success_rate"] * 100.0).map(lambda v: f"{v:.1f}%")
    for name, fmt in [
        ("completion_steps", "{:.1f}"),
        ("completion_time", "{:.1f}"),
        ("final_coverage", "{:.3f}"),
        ("total_path_length", "{:.1f}"),
        ("reassignments", "{:.1f}"),
        ("switches", "{:.1f}"),
        ("conflicts", "{:.1f}"),
        ("congestion", "{:.1f}"),
        ("planner_compute_time_ms", "{:.2f}"),
        ("predictor_inference_time_ms", "{:.2f}"),
        ("execution_penalty_mean", "{:.3f}"),
        ("low_progress_steps", "{:.1f}"),
        ("blocked_or_slow_steps_proxy", "{:.1f}"),
    ]:
        if name in out.columns:
            out[name] = out[name].map(lambda v, f=fmt: f.format(float(v)))
    return out


def _save_metrics_table(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return

    n_rows = len(df)
    n_cols = len(df.columns)
    fig_h = max(2.8, 1.2 + 0.55 * (n_rows + 1))
    fig_w = max(11.0, 2.0 + 1.25 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_facecolor("#263238")
        cell.set_text_props(color="white", weight="bold")

    for r in range(1, n_rows + 1):
        row_color = "#F4F6F8" if r % 2 == 0 else "#FFFFFF"
        for c in range(n_cols):
            table[(r, c)].set_facecolor(row_color)

    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_metrics_tables(summary_df: pd.DataFrame, out_dir: Path, order: list[str]) -> None:
    if summary_df.empty:
        return

    for map_name, sub in summary_df.groupby("map_name"):
        sub = sub.set_index("planner_name").reindex(order).dropna(how="all").reset_index()
        table_df = _format_table_frame(sub)
        _save_metrics_table(
            table_df,
            out_dir / f"metrics_table_{map_name}.png",
            title=f"Planner Metrics Comparison | {map_name}",
        )

    if "map_family" in summary_df.columns:
        for map_family, sub in summary_df.groupby("map_family"):
            sub = sub.set_index("planner_name").reindex(order).dropna(how="all").reset_index()
            table_df = _format_table_frame(sub)
            _save_metrics_table(
                table_df,
                out_dir / f"metrics_table_family_{map_family}.png",
                title=f"Planner Metrics Comparison | family={map_family}",
            )

    overall = (
        summary_df.groupby("planner_name", as_index=False)[
            [
                "success_rate",
                "completion_steps",
                "completion_time",
                "final_coverage",
                "total_path_length",
                "reassignments",
                "switches",
                "conflicts",
                "congestion",
                "planner_compute_time_ms",
                "predictor_inference_time_ms",
                "execution_penalty_mean",
                "blocked_or_slow_steps_proxy",
            ]
        ]
        .mean()
        .set_index("planner_name")
        .reindex(order)
        .dropna(how="all")
        .reset_index()
    )
    _save_metrics_table(
        _format_table_frame(overall),
        out_dir / "metrics_table_overall.png",
        title="Planner Metrics Comparison | Overall Mean Across Maps",
    )


def main() -> None:
    args = parse_args()

    run_id = args.run_id or make_run_id("compare_planners")
    dirs = prepare_output_dirs(args.output_root, run_id)
    planner_order = [get_planner_preset(p).planner_label for p in args.planners]

    save_run_metadata(
        dirs["metadata"] / "run_metadata.json",
        {
            "run_id": run_id,
            "base_config": args.base_config,
            "planners": args.planners,
            "env_configs": args.env_configs,
            "seed_start": args.seed_start,
            "num_seeds": args.num_seeds,
            "git_commit": git_commit_hash(),
        },
    )

    sim = GridSimulation()
    rows: list[dict] = []

    for env_cfg_path in args.env_configs:
        env_label = Path(env_cfg_path).stem
        for planner_choice in args.planners:
            planner_preset = get_planner_preset(planner_choice)
            planner_cfg_path = planner_preset.config_path
            cfg = load_experiment_config(args.base_config, planner_cfg_path=planner_cfg_path, env_cfg_path=env_cfg_path)
            cfg = enforce_mp4_only(cfg)
            cfg["planning"]["planner_name"] = planner_preset.planner_name
            cfg["planning"]["planner_label"] = planner_preset.planner_label
            if planner_preset.planner_name == "physics_rh_cfpa2" and args.physics_weight_file is not None:
                cfg["predictor"]["type"] = "physics_residual"
                cfg["predictor"]["physics_residual"]["enabled"] = True
                cfg["predictor"]["physics_residual"]["weight_file"] = args.physics_weight_file
            if args.max_steps is not None:
                cfg["termination"]["max_steps"] = int(args.max_steps)

            config_snapshot_path = dirs["configs"] / f"resolved_{env_label}_{planner_preset.planner_label}.yaml"
            write_config_snapshot(config_snapshot_path, cfg)

            for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                cfg_local = dict(cfg)
                cfg_local["experiment"] = dict(cfg.get("experiment", {}))
                if args.animate_first_seed_only:
                    cfg_local["experiment"]["save_animation"] = bool(seed == args.seed_start)

                map_name = cfg_local["environment"].get("map_name", cfg_local["environment"].get("map_type", env_label))
                map_family = cfg_local["environment"].get("map_family", map_name)
                stem = f"{planner_preset.planner_label}_{map_name}_seed{seed}"
                episode_dir = dirs["episode"] / map_name / planner_preset.planner_label / f"seed_{seed}"

                result = sim.run_episode(
                    cfg=cfg_local,
                    planner_name=planner_preset.planner_name,
                    seed=seed,
                    output_dir=episode_dir,
                    animation_stem=stem,
                )

                row = dict(result.summary)
                row.update(
                    {
                        "run_id": run_id,
                        "planner_choice": planner_choice,
                        "planner_label": planner_preset.planner_label,
                        "planner_base_name": planner_preset.planner_name,
                        "env_config": env_cfg_path,
                        "planner_config": planner_cfg_path,
                        "map_family": map_family,
                        "coverage_csv": result.coverage_csv_path,
                        "step_log_csv": result.step_log_csv_path,
                        "animation_gif": result.animation_gif_path,
                        "animation_mp4": result.animation_mp4_path,
                    }
                )
                rows.append(row)

                print(
                    f"planner={planner_preset.planner_label} map={map_name} seed={seed} "
                    f"success={row['success']} steps={row['completion_steps']} "
                    f"coverage={row['final_coverage']:.3f} replans={row['replan_count']}",
                    flush=True,
                )

    raw_df = pd.DataFrame(rows)
    raw_csv = dirs["results_csv"] / "compare_planners_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    summary_df = (
        raw_df.groupby(["map_name", "map_family", "planner_name"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            completion_steps=("completion_steps", "mean"),
            completion_time=("completion_time", "mean"),
            final_coverage=("final_coverage", "mean"),
            total_path_length=("total_path_length", "mean"),
            reassignments=("reassignment_count", "mean"),
            switches=("switching_count", "mean"),
            conflicts=("conflict_count", "mean"),
            congestion=("congestion_count", "mean"),
            planner_compute_time_ms=("planner_compute_time_ms_mean", "mean"),
            predictor_inference_time_ms=("predictor_inference_time_ms_mean", "mean"),
            execution_penalty_mean=("execution_penalty_mean", "mean"),
            clearance_penalty_mean=("clearance_penalty_mean", "mean"),
            density_penalty_mean=("density_penalty_mean", "mean"),
            turn_penalty_mean=("turn_penalty_mean", "mean"),
            narrowness_penalty_mean=("narrowness_penalty_mean", "mean"),
            teammate_proximity_penalty_mean=("teammate_proximity_penalty_mean", "mean"),
            slowdown_exposure_penalty_mean=("slowdown_exposure_penalty_mean", "mean"),
            low_progress_steps=("low_progress_steps", "mean"),
            blocked_or_slow_steps_proxy=("blocked_or_slow_steps_proxy", "mean"),
        )
        .sort_values(["map_name", "completion_steps"])
    )
    summary_df["planner_label"] = summary_df["planner_name"]

    summary_csv = dirs["results_csv"] / "compare_planners_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    family_summary_df = (
        raw_df.groupby(["map_family", "planner_label"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            completion_steps=("completion_steps", "mean"),
            completion_time=("completion_time", "mean"),
            final_coverage=("final_coverage", "mean"),
            total_path_length=("total_path_length", "mean"),
            reassignments=("reassignment_count", "mean"),
            switches=("switching_count", "mean"),
            conflicts=("conflict_count", "mean"),
            congestion=("congestion_count", "mean"),
            planner_compute_time_ms=("planner_compute_time_ms_mean", "mean"),
            predictor_inference_time_ms=("predictor_inference_time_ms_mean", "mean"),
            execution_penalty_mean=("execution_penalty_mean", "mean"),
            clearance_penalty_mean=("clearance_penalty_mean", "mean"),
            density_penalty_mean=("density_penalty_mean", "mean"),
            turn_penalty_mean=("turn_penalty_mean", "mean"),
            narrowness_penalty_mean=("narrowness_penalty_mean", "mean"),
            teammate_proximity_penalty_mean=("teammate_proximity_penalty_mean", "mean"),
            slowdown_exposure_penalty_mean=("slowdown_exposure_penalty_mean", "mean"),
            low_progress_steps=("low_progress_steps", "mean"),
            blocked_or_slow_steps_proxy=("blocked_or_slow_steps_proxy", "mean"),
        )
        .sort_values(["map_family", "completion_steps"])
    )
    family_summary_df["planner_name"] = family_summary_df["planner_label"]
    family_summary_csv = dirs["results_csv"] / "compare_planners_map_family_summary.csv"
    family_summary_df.to_csv(family_summary_csv, index=False)

    _plot_summary(summary_df, dirs["plots"], order=planner_order)
    _plot_metrics_tables(summary_df, dirs["plots"], order=planner_order)

    print("\n=== Aggregate Summary ===")
    print(summary_df.to_string(index=False))
    print(f"raw_csv: {raw_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"family_summary_csv: {family_summary_csv}")
    print(f"plots_dir: {dirs['plots']}")


if __name__ == "__main__":
    main()
