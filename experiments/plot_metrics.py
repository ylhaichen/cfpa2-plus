from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark metrics from raw comparison CSV")
    parser.add_argument("--input", type=str, required=True, help="Path to raw results CSV")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["map_name", "map_family"],
        choices=["map_name", "map_family", "normalization_mode", "run_group"],
    )
    return parser.parse_args()


def _planner_col(df: pd.DataFrame) -> str:
    if "planner_label" in df.columns:
        return "planner_label"
    return "planner_name"


def _group_values(df: pd.DataFrame, group_key: str) -> list[tuple[str, pd.DataFrame]]:
    if group_key not in df.columns:
        return []
    return [(str(name), sub.copy()) for name, sub in df.groupby(group_key)]


def _boxplot_completion(df: pd.DataFrame, out_dir: Path, group_key: str) -> None:
    planner_col = _planner_col(df)
    for group_name, sub in _group_values(df, group_key):
        groups = []
        labels = []
        for planner_name, sub2 in sub.groupby(planner_col):
            groups.append(sub2["completion_steps"].to_list())
            labels.append(planner_name)

        if not groups:
            continue
        plt.figure(figsize=(8.0, 4.5))
        plt.boxplot(groups, tick_labels=labels)
        plt.title(f"Completion Steps Distribution | {group_key}={group_name}")
        plt.ylabel("steps")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"box_completion_steps_{group_key}_{group_name}.png", dpi=160)
        plt.close()


def _plot_compute_vs_coverage(df: pd.DataFrame, out_dir: Path) -> None:
    planner_col = _planner_col(df)
    compute_col = "planner_compute_time_ms_mean" if "planner_compute_time_ms_mean" in df.columns else "planner_compute_time_ms"
    if compute_col not in df.columns or "final_coverage" not in df.columns:
        return
    plt.figure(figsize=(7.6, 4.8))
    for planner_name, sub in df.groupby(planner_col):
        plt.scatter(
            sub[compute_col],
            sub["final_coverage"],
            label=planner_name,
            alpha=0.8,
        )
    plt.xlabel("planner_compute_time_ms_mean")
    plt.ylabel("final_coverage")
    plt.ylim(0.0, 1.01)
    plt.title("Planner Compute Time vs Coverage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_compute_vs_coverage.png", dpi=160)
    plt.close()


def _aggregate(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    planner_col = _planner_col(df)
    keys = [group_key, planner_col]
    metrics = [
        "completion_steps",
        "final_coverage",
        "planner_compute_time_ms_mean",
        "planner_compute_time_ms",
        "execution_penalty_mean",
        "low_progress_steps",
        "blocked_or_slow_steps_proxy",
        "avg_assigned_frontier_execution_penalty",
    ]
    available = [m for m in metrics if m in df.columns]
    if not available or group_key not in df.columns:
        return pd.DataFrame()
    agg_map = {metric: "mean" for metric in available}
    summary = df.groupby(keys, as_index=False).agg(agg_map).sort_values(keys)
    summary.rename(columns={planner_col: "planner_label"}, inplace=True)
    return summary


def _plot_metric_bars(summary: pd.DataFrame, out_dir: Path, group_key: str, metric: str, ylabel: str) -> None:
    if summary.empty or metric not in summary.columns:
        return
    for group_name, sub in summary.groupby(group_key):
        sub = sub.copy()
        plt.figure(figsize=(8.2, 4.8))
        plt.bar(sub["planner_label"], sub[metric], color=["#6D4C41", "#1976D2", "#2E7D32", "#8E24AA", "#00897B"][: len(sub)])
        plt.title(f"{metric} | {group_key}={group_name}")
        plt.ylabel(ylabel)
        if metric.endswith("coverage"):
            plt.ylim(0.0, 1.01)
        elif metric.endswith("penalty"):
            plt.ylim(0.0, max(1.0, float(sub[metric].max()) * 1.15))
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_{group_key}_{group_name}.png", dpi=160)
        plt.close()


def _save_summary_table(summary: pd.DataFrame, out_dir: Path, name: str) -> None:
    if summary.empty:
        return
    fig_h = max(3.2, 1.1 + 0.38 * (len(summary) + 1))
    fig_w = max(10.0, 1.5 + 1.15 * len(summary.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    rendered = summary.copy()
    for col in rendered.columns:
        if col in ("planner_label", "map_name", "map_family", "normalization_mode", "run_group"):
            continue
        rendered[col] = rendered[col].map(lambda v: f"{float(v):.3f}" if isinstance(v, (int, float)) and not math.isnan(float(v)) else v)
    table = ax.table(cellText=rendered.values.tolist(), colLabels=list(rendered.columns), cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    df = pd.read_csv(input_path)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_key in args.group_by:
        _boxplot_completion(df, out_dir, group_key)
    _plot_compute_vs_coverage(df, out_dir)

    for group_key in args.group_by:
        summary = _aggregate(df, group_key)
        if summary.empty:
            continue
        summary_csv = out_dir / f"summary_by_{group_key}.csv"
        summary.to_csv(summary_csv, index=False)
        _save_summary_table(summary, out_dir, name=f"summary_by_{group_key}")
        _plot_metric_bars(summary, out_dir, group_key, "execution_penalty_mean", "mean penalty")
        _plot_metric_bars(summary, out_dir, group_key, "low_progress_steps", "mean steps")
        _plot_metric_bars(summary, out_dir, group_key, "blocked_or_slow_steps_proxy", "mean steps")
        _plot_metric_bars(summary, out_dir, group_key, "avg_assigned_frontier_execution_penalty", "mean penalty")

    print(f"plots_dir: {out_dir}")


if __name__ == "__main__":
    main()
