from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..core.metrics import save_coverage_curve_csv
from ..core.simulator import run_simulation


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg(default_path: Path, override_path: Path) -> dict[str, Any]:
    with open(default_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if override_path.resolve() == default_path.resolve():
        return base
    with open(override_path, "r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return deep_merge(base, override)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CFPA-2 mode comparisons")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--maps", nargs="+", default=["open", "rooms", "maze"], choices=["open", "rooms", "maze"])
    parser.add_argument("--max-steps", type=int, default=None, help="Optional termination step override for faster sweeps")
    parser.add_argument("--save-animation", action="store_true", help="Save one example animation per map for dual_joint")
    return parser.parse_args()


def _coverage_mean_std(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
        if len(c) < max_len:
            arr[i, len(c) :] = c[-1]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    default_cfg_path = root / "config" / "default.yaml"
    map_cfg_paths = {
        "open": root / "config" / "map_open.yaml",
        "rooms": root / "config" / "map_rooms.yaml",
        "maze": root / "config" / "map_maze.yaml",
    }

    modes = ["single", "dual_greedy", "dual_joint"]
    rows: list[dict[str, Any]] = []
    coverage_bank: dict[tuple[str, str], list[list[float]]] = {}

    for map_name in args.maps:
        cfg_base = load_cfg(default_cfg_path, map_cfg_paths[map_name])
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            for mode in modes:
                cfg = deep_merge(cfg_base, {})
                cfg["visualization"]["enable_live_plot"] = False
                cfg["visualization"]["save_animation"] = bool(
                    args.save_animation and mode == "dual_joint" and seed == args.seed_start
                )
                if args.max_steps is not None:
                    cfg["termination"]["max_steps"] = int(args.max_steps)

                result = run_simulation(
                    cfg=cfg,
                    mode=mode,
                    seed=seed,
                    enable_viz=False,
                    animation_filename=f"demo_{map_name}_{mode}_seed{seed}.gif",
                )

                row = result.metrics.to_summary_row()
                rows.append(row)

                key = (map_name, mode)
                coverage_bank.setdefault(key, []).append(result.coverage_curve)

                output_base = Path(cfg.get("outputs", {}).get("base_dir", "outputs"))
                cov_path = output_base / "results_csv" / f"coverage_{map_name}_{mode}_seed{seed}.csv"
                save_coverage_curve_csv(cov_path, result.metrics)

                print(
                    f"map={map_name} seed={seed} mode={mode} "
                    f"success={row['success']} steps={row['steps']} "
                    f"coverage={row['final_coverage']:.3f} replans={row['replan_count']}"
                , flush=True)

    df = pd.DataFrame(rows)
    output_base = Path("outputs")
    output_base.mkdir(parents=True, exist_ok=True)
    results_path = output_base / "results_csv" / "compare_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    # Coverage comparison curves per map.
    for map_name in args.maps:
        plt.figure(figsize=(7.5, 4.5))
        for mode in modes:
            curves = coverage_bank.get((map_name, mode), [])
            if not curves:
                continue
            mean, std = _coverage_mean_std(curves)
            x = np.arange(len(mean))
            plt.plot(x, mean, label=mode, linewidth=2)
            plt.fill_between(x, np.maximum(0.0, mean - std), np.minimum(1.0, mean + std), alpha=0.2)

        plt.title(f"Coverage Curves | {map_name}")
        plt.xlabel("Step")
        plt.ylabel("Explored Free-Space Ratio")
        plt.ylim(0.0, 1.01)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = output_base / "figures" / f"coverage_compare_{map_name}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=160)
        plt.close()

    summary = (
        df.groupby(["map_type", "mode"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_steps=("steps", "mean"),
            mean_repeated_ratio=("repeated_coverage_ratio", "mean"),
            mean_replans=("replan_count", "mean"),
        )
        .sort_values(["map_type", "mean_steps"])  # Lower steps is better.
    )

    summary_path = output_base / "results_csv" / "compare_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Aggregate Summary ===")
    print(summary.to_string(index=False))
    print(f"\nresults_csv: {results_path}")
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
