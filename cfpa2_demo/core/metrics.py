from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .robot import RobotState


@dataclass
class SimulationMetrics:
    mode: str
    map_type: str
    seed: int
    coverage_curve: list[float] = field(default_factory=list)
    frontier_cell_curve: list[int] = field(default_factory=list)
    frontier_cluster_curve: list[int] = field(default_factory=list)
    steps: int = 0
    success: bool = False
    failure_reason: str = ""
    replan_count: int = 0
    replan_reasons: dict[str, int] = field(default_factory=dict)
    target_conflict_count: int = 0
    target_invalidation_events: int = 0
    idle_steps: int = 0
    total_move_steps: int = 0
    revisited_move_steps: int = 0

    def log_coverage(self, coverage: float) -> None:
        self.coverage_curve.append(float(coverage))

    def log_frontier_counts(self, frontier_cells: int, frontier_clusters: int) -> None:
        self.frontier_cell_curve.append(int(frontier_cells))
        self.frontier_cluster_curve.append(int(frontier_clusters))

    def log_replan(self, reason: str) -> None:
        self.replan_count += 1
        self.replan_reasons[reason] = self.replan_reasons.get(reason, 0) + 1

    def finalize(self, robots: Sequence[RobotState], steps: int, success: bool, reason: str) -> None:
        self.steps = int(steps)
        self.success = bool(success)
        self.failure_reason = reason
        self.idle_steps = int(sum(r.idle_steps for r in robots))
        self.total_move_steps = int(sum(r.total_move_steps for r in robots))
        self.revisited_move_steps = int(sum(r.revisited_move_steps for r in robots))

    @property
    def final_coverage(self) -> float:
        if not self.coverage_curve:
            return 0.0
        return float(self.coverage_curve[-1])

    @property
    def repeated_coverage_ratio(self) -> float:
        if self.total_move_steps <= 0:
            return 0.0
        return self.revisited_move_steps / float(self.total_move_steps)

    def to_summary_row(self) -> dict[str, Any]:
        avg_frontier_cells = (
            sum(self.frontier_cell_curve) / float(len(self.frontier_cell_curve))
            if self.frontier_cell_curve
            else 0.0
        )
        avg_frontier_clusters = (
            sum(self.frontier_cluster_curve) / float(len(self.frontier_cluster_curve))
            if self.frontier_cluster_curve
            else 0.0
        )
        max_frontier_clusters = max(self.frontier_cluster_curve) if self.frontier_cluster_curve else 0
        return {
            "mode": self.mode,
            "map_type": self.map_type,
            "seed": self.seed,
            "success": self.success,
            "steps": self.steps,
            "final_coverage": self.final_coverage,
            "replan_count": self.replan_count,
            "target_conflict_count": self.target_conflict_count,
            "target_invalidation_events": self.target_invalidation_events,
            "idle_steps": self.idle_steps,
            "total_move_steps": self.total_move_steps,
            "repeated_coverage_ratio": self.repeated_coverage_ratio,
            "avg_frontier_cells": avg_frontier_cells,
            "avg_frontier_clusters": avg_frontier_clusters,
            "max_frontier_clusters": max_frontier_clusters,
            "failure_reason": self.failure_reason,
        }


def save_coverage_curve_csv(path: str | Path, metrics: SimulationMetrics) -> None:
    rows = [{"step": i, "coverage": c} for i, c in enumerate(metrics.coverage_curve)]
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
