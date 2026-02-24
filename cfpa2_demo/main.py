from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from .core.metrics import save_coverage_curve_csv
from .core.simulator import run_simulation


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in updates.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str) -> dict[str, Any]:
    config_path = str(config_path)
    root = Path(__file__).resolve().parent
    default_path = root / "config" / "default.yaml"

    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = Path(config_path)
    if not p.is_absolute():
        p = root / config_path

    if p.resolve() != default_path.resolve():
        with open(p, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        cfg = deep_merge(cfg, override)

    return cfg


def save_coverage_plot(path: Path, coverage: list[float], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(coverage)), coverage, linewidth=2)
    plt.ylim(0, 1.01)
    plt.xlabel("Step")
    plt.ylabel("Explored Free-Space Ratio")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFPA-2 Demo")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="YAML config path")
    parser.add_argument("--mode", type=str, default="dual_joint", choices=["single", "dual_greedy", "dual_joint"])
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--no-viz", action="store_true", help="Disable live plotting")
    parser.add_argument("--save-animation", action="store_true", help="Force-save gif animation")
    parser.add_argument("--save-video", action="store_true", help="Save run as video (mp4)")
    parser.add_argument("--video-format", type=str, default=None, choices=["mp4", "mov", "mkv", "avi"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.save_animation:
        cfg["visualization"]["save_animation"] = True
    if args.save_video:
        cfg["visualization"]["save_video"] = True
    if args.video_format is not None:
        cfg["visualization"]["video_format"] = args.video_format
    if args.no_viz:
        cfg["visualization"]["enable_live_plot"] = False

    result = run_simulation(
        cfg=cfg,
        mode=args.mode,
        seed=args.seed,
        enable_viz=not args.no_viz,
    )

    output_base = Path(cfg.get("outputs", {}).get("base_dir", "outputs"))
    output_base.mkdir(parents=True, exist_ok=True)

    cov_csv = output_base / "results_csv" / f"coverage_{args.mode}_{result.metrics.map_type}_seed{result.metrics.seed}.csv"
    save_coverage_curve_csv(cov_csv, result.metrics)

    cov_png = output_base / "figures" / f"coverage_{args.mode}_{result.metrics.map_type}_seed{result.metrics.seed}.png"
    save_coverage_plot(cov_png, result.coverage_curve, f"{args.mode} | {result.metrics.map_type} | seed={result.metrics.seed}")

    summary = result.metrics.to_summary_row()
    print("=== Simulation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"coverage_csv: {cov_csv}")
    print(f"coverage_plot: {cov_png}")
    if result.animation_path:
        print(f"animation: {result.animation_path}")
    if result.video_path:
        print(f"video: {result.video_path}")


if __name__ == "__main__":
    main()
