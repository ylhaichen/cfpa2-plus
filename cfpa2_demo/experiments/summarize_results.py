from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CFPA-2 comparison CSV")
    parser.add_argument("--input", type=str, default="outputs/results_csv/compare_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Missing results csv: {path}")

    df = pd.read_csv(path)
    summary = (
        df.groupby(["map_type", "mode"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_steps=("steps", "mean"),
            median_steps=("steps", "median"),
            mean_replan_count=("replan_count", "mean"),
            mean_repeat_ratio=("repeated_coverage_ratio", "mean"),
        )
        .sort_values(["map_type", "mean_steps"])
    )

    out_dir = path.parent
    summary_path = out_dir / "compare_summary_from_script.csv"
    summary.to_csv(summary_path, index=False)

    for map_name, sub in summary.groupby("map_type"):
        plt.figure(figsize=(6.5, 4.0))
        plt.bar(sub["mode"], sub["mean_steps"], color=["#9E9D24", "#0288D1", "#388E3C"])
        plt.title(f"Mean Completion Steps | {map_name}")
        plt.ylabel("Steps")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig_path = Path("outputs/figures") / f"summary_steps_{map_name}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150)
        plt.close()

    print(summary.to_string(index=False))
    print(f"\nsummary_csv: {summary_path}")


if __name__ == "__main__":
    main()
