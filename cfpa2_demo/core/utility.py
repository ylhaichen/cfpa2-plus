from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_map import Cell, UNKNOWN, OccupancyGrid
from .planner_astar import astar_path, path_cost
from .robot import RobotState


@dataclass
class CandidateEvaluation:
    utility: float
    information_gain: float
    travel_cost: float
    switch_penalty: float
    path: list[Cell]


def information_gain(grid: OccupancyGrid, frontier_cell: Cell, radius: int) -> float:
    cx, cy = frontier_cell
    rr = radius * radius

    min_x = max(0, cx - radius)
    max_x = min(grid.width - 1, cx + radius)
    min_y = max(0, cy - radius)
    max_y = min(grid.height - 1, cy + radius)

    gain = 0
    for y in range(min_y, max_y + 1):
        dy = y - cy
        for x in range(min_x, max_x + 1):
            dx = x - cx
            if dx * dx + dy * dy > rr:
                continue
            if grid.grid[y, x] == UNKNOWN:
                gain += 1
    return float(gain)


def travel_cost_astar(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    neighborhood: int = 8,
) -> tuple[float, list[Cell] | None]:
    path = astar_path(grid, start, goal, neighborhood=neighborhood)
    if path is None:
        return float("inf"), None
    return path_cost(path, neighborhood=neighborhood), path


def switch_penalty(robot: RobotState, frontier: Cell) -> float:
    if robot.current_target is None:
        return 0.0
    return 0.0 if robot.current_target == frontier else 1.0


def utility(
    robot: RobotState,
    frontier: Cell,
    grid: OccupancyGrid,
    cfg: dict,
    neighborhood: int = 8,
) -> CandidateEvaluation | None:
    weights = cfg["frontier"]

    ig = information_gain(grid, frontier, int(weights["ig_radius"]))
    cost, path = travel_cost_astar(grid, robot.pose, frontier, neighborhood=neighborhood)
    if path is None:
        return None

    sw = switch_penalty(robot, frontier)
    score = float(weights["w_ig"]) * ig - float(weights["w_c"]) * cost - float(weights["w_sw"]) * sw
    return CandidateEvaluation(
        utility=score,
        information_gain=ig,
        travel_cost=cost,
        switch_penalty=sw,
        path=path,
    )
