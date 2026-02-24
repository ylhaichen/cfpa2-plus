from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .grid_map import Cell, FREE, OccupancyGrid


@dataclass
class RobotState:
    robot_id: int
    pose: Cell
    current_target: Cell | None = None
    path: list[Cell] = field(default_factory=list)
    trajectory_history: list[Cell] = field(default_factory=list)
    steps_since_progress: int = 0
    idle_steps: int = 0
    total_steps: int = 0
    total_move_steps: int = 0
    revisited_move_steps: int = 0

    def __post_init__(self) -> None:
        self.trajectory_history.append(self.pose)

    def clear_plan(self) -> None:
        self.current_target = None
        self.path = []

    def set_plan(self, target: Cell | None, path: list[Cell] | None) -> None:
        self.current_target = target
        self.steps_since_progress = 0
        if not path:
            self.path = []
            return
        if path and path[0] == self.pose:
            self.path = list(path[1:])
        else:
            self.path = list(path)

    def at_target(self) -> bool:
        return self.current_target is not None and self.pose == self.current_target

    def move_one_step(self, grid: OccupancyGrid) -> bool:
        self.total_steps += 1
        prev = self.pose

        if not self.path:
            self.idle_steps += 1
            self.steps_since_progress += 1
            self.trajectory_history.append(self.pose)
            return False

        nxt = self.path.pop(0)
        if not grid.in_bounds(nxt) or grid.get(nxt) != FREE:
            self.path = []
            self.idle_steps += 1
            self.steps_since_progress += 1
            self.trajectory_history.append(self.pose)
            return False

        self.pose = nxt
        self.total_move_steps += 1
        if nxt in self.trajectory_history:
            self.revisited_move_steps += 1

        if self.pose == prev:
            self.steps_since_progress += 1
        else:
            self.steps_since_progress = 0

        self.trajectory_history.append(self.pose)
        return True
