from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manifest-driven Phase 1.5 calibration runs")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--output-subdir", type=str, default=None)
    return parser.parse_args()


def _safe_corr(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    if col_a not in df.columns or col_b not in df.columns:
        return float("nan")
    sub = df[[col_a, col_b]].dropna()
    if len(sub) < 2:
        return float("nan")
    if sub[col_a].nunique(dropna=True) <= 1 or sub[col_b].nunique(dropna=True) <= 1:
        return float("nan")
    return float(sub[col_a].corr(sub[col_b]))


def _summary_path(output_root: str, run_id: str) -> Path:
    return Path(output_root) / "benchmarks" / run_id / "results_csv" / "single_run_summary.csv"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    output_subdir = args.output_subdir or str(manifest_df["output_subdir"].iloc[0])
    summary_root = Path(args.output_root) / "benchmarks" / output_subdir / "phase15_summary"
    results_dir = summary_root / "results_csv"
    plots_dir = summary_root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []
    missing_rows: list[dict] = []
    for row in manifest_df.to_dict(orient="records"):
        summary_csv = _summary_path(args.output_root, str(row["run_id"]))
        merged = dict(row)
        merged["summary_csv"] = str(summary_csv)
        if not summary_csv.exists():
            merged["result_status"] = "missing"
            missing_rows.append(merged)
            continue

        summary_df = pd.read_csv(summary_csv)
        if summary_df.empty:
            merged["result_status"] = "empty"
            missing_rows.append(merged)
            continue

        summary = summary_df.iloc[0].to_dict()
        merged.update(summary)
        merged["result_status"] = "ok"
        result_rows.append(merged)

    per_run_df = pd.DataFrame(result_rows)
    missing_df = pd.DataFrame(missing_rows)

    per_run_csv = results_dir / "phase15_per_run_results.csv"
    per_run_df.to_csv(per_run_csv, index=False)

    if not missing_df.empty:
        missing_csv = results_dir / "phase15_missing_runs.csv"
        missing_df.to_csv(missing_csv, index=False)
    else:
        missing_csv = None

    if per_run_df.empty:
        raise RuntimeError("No completed manifest rows found; cannot summarize Phase 1.5 results.")

    group_keys = ["run_group", "normalization_mode", "map_family", "planner_label"]
    metric_cols = [
        "completion_steps",
        "final_coverage",
        "execution_penalty_mean",
        "low_progress_steps",
        "blocked_or_slow_steps_proxy",
        "avg_assigned_frontier_execution_penalty",
        "clearance_penalty_mean",
        "density_penalty_mean",
        "turn_penalty_mean",
        "narrowness_penalty_mean",
        "teammate_proximity_penalty_mean",
        "slowdown_exposure_penalty_mean",
        "conflict_count",
        "congestion_count",
        "duplicated_exploration_proxy",
    ]
    available_metric_cols = [col for col in metric_cols if col in per_run_df.columns]
    grouped_df = (
        per_run_df.groupby([key for key in group_keys if key in per_run_df.columns], as_index=False)
        .agg({col: "mean" for col in available_metric_cols} | {"success": "mean"})
        .rename(columns={"success": "success_rate"})
        .sort_values([key for key in group_keys if key in per_run_df.columns])
    )
    grouped_csv = results_dir / "phase15_grouped_summary.csv"
    grouped_df.to_csv(grouped_csv, index=False)

    config_cols = [
        "planner_label",
        "run_group",
        "normalization_mode",
        "execution_weight",
        "w_clearance",
        "w_density",
        "w_turn",
        "w_narrow",
        "w_team",
        "w_slowdown",
        "feature_clip_max",
        "total_clip_max",
        "soft_saturation_gamma",
    ]
    best_df = (
        per_run_df.groupby([col for col in config_cols if col in per_run_df.columns], as_index=False)
        .agg(
            final_coverage=("final_coverage", "mean"),
            completion_steps=("completion_steps", "mean"),
            execution_penalty_mean=("execution_penalty_mean", "mean"),
            low_progress_steps=("low_progress_steps", "mean"),
            blocked_or_slow_steps_proxy=("blocked_or_slow_steps_proxy", "mean"),
            avg_assigned_frontier_execution_penalty=("avg_assigned_frontier_execution_penalty", "mean"),
        )
        .copy()
    )
    best_df["rank_final_coverage"] = best_df["final_coverage"].rank(method="dense", ascending=False)
    best_df["rank_blocked_or_slow"] = best_df["blocked_or_slow_steps_proxy"].rank(method="dense", ascending=True)
    best_df["rank_low_progress"] = best_df["low_progress_steps"].rank(method="dense", ascending=True)
    best_df["rank_execution_penalty"] = best_df["execution_penalty_mean"].rank(method="dense", ascending=True)
    best_df["combined_rank"] = (
        best_df["rank_final_coverage"]
        + best_df["rank_blocked_or_slow"]
        + best_df["rank_low_progress"]
        + best_df["rank_execution_penalty"]
    )
    best_df = best_df.sort_values(["combined_rank", "final_coverage"], ascending=[True, False])
    best_csv = results_dir / "phase15_best_configs.csv"
    best_df.to_csv(best_csv, index=False)

    redundancy_rows: list[dict] = []
    redundancy_pairs = [
        ("narrow_vs_slowdown", "narrowness_penalty_mean", "slowdown_exposure_penalty_mean"),
        ("team_vs_conflict", "teammate_proximity_penalty_mean", "conflict_count"),
        ("team_vs_congestion", "teammate_proximity_penalty_mean", "congestion_count"),
        ("team_vs_duplicate", "teammate_proximity_penalty_mean", "duplicated_exploration_proxy"),
    ]
    scopes = [("overall", per_run_df)] + [
        (f"map_family:{map_family}", sub.copy()) for map_family, sub in per_run_df.groupby("map_family")
    ]
    for scope_name, scope_df in scopes:
        for pair_name, col_a, col_b in redundancy_pairs:
            corr = _safe_corr(scope_df, col_a, col_b)
            redundancy_rows.append(
                {
                    "scope": scope_name,
                    "pair_name": pair_name,
                    "col_a": col_a,
                    "col_b": col_b,
                    "pearson_r": corr,
                    "abs_pearson_r": abs(corr) if not math.isnan(corr) else float("nan"),
                }
            )
    redundancy_df = pd.DataFrame(redundancy_rows).sort_values(["scope", "pair_name"])
    redundancy_csv = results_dir / "phase15_feature_redundancy_summary.csv"
    redundancy_df.to_csv(redundancy_csv, index=False)

    manifest_enriched = manifest_df.merge(
        per_run_df[["row_index", "result_status", "summary_csv"] + [col for col in available_metric_cols if col in per_run_df.columns]],
        on="row_index",
        how="left",
    )
    manifest_enriched["result_status"] = manifest_enriched["result_status"].fillna("missing")
    manifest_enriched_csv = results_dir / "phase15_manifest_enriched.csv"
    manifest_enriched.to_csv(manifest_enriched_csv, index=False)

    print(grouped_df.head(20).to_string(index=False))
    print(f"per_run_csv: {per_run_csv}")
    print(f"grouped_csv: {grouped_csv}")
    print(f"best_configs_csv: {best_csv}")
    print(f"redundancy_csv: {redundancy_csv}")
    print(f"manifest_enriched_csv: {manifest_enriched_csv}")
    if missing_csv is not None:
        print(f"missing_csv: {missing_csv}")
    print(f"plots_dir: {plots_dir}")


if __name__ == "__main__":
    main()
