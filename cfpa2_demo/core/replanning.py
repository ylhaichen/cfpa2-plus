from __future__ import annotations

from typing import Iterable, Sequence

from .allocator import TargetAssignment
from .frontier import is_frontier_cell
from .grid_map import Cell, OccupancyGrid
from .robot import RobotState


def should_replan(
    grid: OccupancyGrid,
    robots: Sequence[RobotState],
    frontier_reps: set[Cell],
    step: int,
    prev_frontier_count: int,
    current_frontier_count: int,
    cfg: dict,
) -> tuple[bool, str]:
    repl_cfg = cfg["replanning"]
    enable_event = bool(repl_cfg.get("enable_event_replan", True))

    interval = int(repl_cfg.get("periodic_replan_interval", 0))
    if interval > 0 and step % interval == 0:
        return True, "periodic"

    if not enable_event:
        return False, "none"

    frontier_change_thr = float(repl_cfg.get("frontier_change_threshold", 0.25))
    if prev_frontier_count >= 0:
        baseline = max(1, prev_frontier_count)
        ratio = abs(current_frontier_count - prev_frontier_count) / baseline
        if ratio > frontier_change_thr:
            return True, "frontier_count_change"

    stuck_thr = int(repl_cfg.get("stuck_threshold", 8))
    invalidation_path_threshold = int(repl_cfg.get("invalidation_path_threshold", 4))
    invalidation_distance_threshold = float(repl_cfg.get("invalidation_distance_threshold", 2.0))
    invalidation_distance_sq = invalidation_distance_threshold * invalidation_distance_threshold

    for robot in robots:
        if robot.at_target():
            return True, f"target_reached_r{robot.robot_id}"

        if robot.current_target is not None:
            if not is_frontier_cell(grid, robot.current_target, neighborhood=int(cfg["frontier"].get("neighborhood", 8))):
                dx = robot.current_target[0] - robot.pose[0]
                dy = robot.current_target[1] - robot.pose[1]
                dist_sq = dx * dx + dy * dy
                close_to_target = dist_sq <= invalidation_distance_sq
                near_completion = bool(robot.path) and len(robot.path) <= invalidation_path_threshold
                if not close_to_target and not near_completion:
                    return True, f"target_invalidated_r{robot.robot_id}"

        if robot.current_target is not None and not robot.path:
            return True, f"path_empty_r{robot.robot_id}"

        if robot.current_target is not None and robot.steps_since_progress > stuck_thr:
            return True, f"stuck_r{robot.robot_id}"

    return False, "none"


def apply_hysteresis(
    old_assignments: Sequence[TargetAssignment],
    new_assignments: Sequence[TargetAssignment],
    old_score: float | None,
    new_score: float,
    cfg: dict,
) -> tuple[Sequence[TargetAssignment], bool]:
    margin = float(cfg["allocator"].get("hysteresis_margin", 0.0))
    if old_score is None:
        return new_assignments, False
    if new_score > old_score + margin:
        return new_assignments, False
    return old_assignments, True


def tick_reservations(reservation_table: dict[Cell, dict[str, int]]) -> None:
    expired: list[Cell] = []
    for cell, entry in reservation_table.items():
        entry["ttl"] = int(entry.get("ttl", 0)) - 1
        if entry["ttl"] <= 0:
            expired.append(cell)
    for cell in expired:
        reservation_table.pop(cell, None)


def update_reservations(
    reservation_table: dict[Cell, dict[str, int]],
    assignments: Sequence[TargetAssignment],
    ttl: int,
) -> None:
    tick_reservations(reservation_table)
    for a in assignments:
        if not a.valid or a.target is None:
            continue
        reservation_table[a.target] = {"robot_id": int(a.robot_id), "ttl": int(ttl)}
