from __future__ import annotations

import heapq
import math
from typing import Iterable

from .grid_map import Cell, FREE, OccupancyGrid


def _heuristic(a: Cell, b: Cell, neighborhood: int) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if neighborhood == 4:
        return float(dx + dy)
    # Octile distance
    return float((dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy))


def _neighbors(cell: Cell, neighborhood: int) -> list[tuple[Cell, float]]:
    x, y = cell
    if neighborhood == 4:
        return [((x + 1, y), 1.0), ((x - 1, y), 1.0), ((x, y + 1), 1.0), ((x, y - 1), 1.0)]
    c = math.sqrt(2.0)
    return [
        ((x + 1, y), 1.0),
        ((x - 1, y), 1.0),
        ((x, y + 1), 1.0),
        ((x, y - 1), 1.0),
        ((x + 1, y + 1), c),
        ((x + 1, y - 1), c),
        ((x - 1, y + 1), c),
        ((x - 1, y - 1), c),
    ]


def astar_path(grid: OccupancyGrid, start: Cell, goal: Cell, neighborhood: int = 8) -> list[Cell] | None:
    if not grid.in_bounds(start) or not grid.in_bounds(goal):
        return None
    if grid.get(start) != FREE or grid.get(goal) != FREE:
        return None
    if start == goal:
        return [start]

    open_heap: list[tuple[float, Cell]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    f_score: dict[Cell, float] = {start: _heuristic(start, goal, neighborhood)}

    closed: set[Cell] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)

        closed.add(current)

        for nxt, step_cost in _neighbors(current, neighborhood):
            if not grid.in_bounds(nxt):
                continue
            if grid.get(nxt) != FREE:
                continue

            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + _heuristic(nxt, goal, neighborhood)
                f_score[nxt] = f
                heapq.heappush(open_heap, (f, nxt))

    return None


def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_cost(path: list[Cell] | None, neighborhood: int = 8) -> float:
    if not path or len(path) <= 1:
        return 0.0 if path else float("inf")
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            total += math.sqrt(2.0)
        else:
            total += 1.0
    return total
