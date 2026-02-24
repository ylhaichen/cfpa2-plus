from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FREE = 0
OCCUPIED = 1


@dataclass
class MapSpec:
    width: int
    height: int
    obstacle_density: float
    random_seed: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _add_boundaries(grid: np.ndarray) -> None:
    grid[0, :] = OCCUPIED
    grid[-1, :] = OCCUPIED
    grid[:, 0] = OCCUPIED
    grid[:, -1] = OCCUPIED


def generate_open_map(width: int, height: int, obstacle_density: float, seed: int) -> np.ndarray:
    rng = _rng(seed)
    grid = np.full((height, width), FREE, dtype=np.int8)
    _add_boundaries(grid)

    interior_h = height - 2
    interior_w = width - 2
    mask = rng.random((interior_h, interior_w)) < obstacle_density
    grid[1:-1, 1:-1][mask] = OCCUPIED
    return grid


def _carve_door(line: np.ndarray, rng: np.random.Generator, width: int = 3) -> None:
    if line.size <= width + 2:
        return
    center = int(rng.integers(1 + width // 2, line.size - 1 - width // 2))
    half = width // 2
    line[max(1, center - half) : min(line.size - 1, center + half + 1)] = FREE


def generate_rooms_map(width: int, height: int, obstacle_density: float, seed: int) -> np.ndarray:
    rng = _rng(seed)
    grid = np.full((height, width), FREE, dtype=np.int8)
    _add_boundaries(grid)

    room_w = max(10, width // 6)
    room_h = max(10, height // 6)

    v_walls = list(range(room_w, width - 1, room_w))
    h_walls = list(range(room_h, height - 1, room_h))

    for x in v_walls:
        grid[:, x] = OCCUPIED
    for y in h_walls:
        grid[y, :] = OCCUPIED

    # Carve at least one doorway per wall segment to keep room graph connected.
    y_bounds = [0] + h_walls + [height - 1]
    for x in v_walls:
        for y0, y1 in zip(y_bounds[:-1], y_bounds[1:]):
            if y1 - y0 <= 2:
                continue
            door_y = int(rng.integers(y0 + 1, y1))
            grid[door_y, x] = FREE

    x_bounds = [0] + v_walls + [width - 1]
    for y in h_walls:
        for x0, x1 in zip(x_bounds[:-1], x_bounds[1:]):
            if x1 - x0 <= 2:
                continue
            door_x = int(rng.integers(x0 + 1, x1))
            grid[y, door_x] = FREE

    if obstacle_density > 0:
        interior = grid[1:-1, 1:-1]
        free_cells = np.argwhere(interior == FREE)
        n_noise = int(obstacle_density * 0.12 * free_cells.shape[0])
        if n_noise > 0:
            chosen_idx = rng.choice(free_cells.shape[0], size=n_noise, replace=False)
            chosen = free_cells[chosen_idx]
            interior[chosen[:, 0], chosen[:, 1]] = OCCUPIED

    return grid


def generate_maze_map(width: int, height: int, seed: int) -> np.ndarray:
    # Maze carving works best on odd dimensions.
    if width % 2 == 0:
        width -= 1
    if height % 2 == 0:
        height -= 1

    rng = _rng(seed)
    grid = np.full((height, width), OCCUPIED, dtype=np.int8)

    def neighbors(cx: int, cy: int):
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                yield nx, ny, dx, dy

    start = (1, 1)
    stack = [start]
    grid[start[1], start[0]] = FREE

    visited = {start}
    while stack:
        cx, cy = stack[-1]
        candidates = [(nx, ny, dx, dy) for nx, ny, dx, dy in neighbors(cx, cy) if (nx, ny) not in visited]
        if not candidates:
            stack.pop()
            continue
        nx, ny, dx, dy = candidates[int(rng.integers(0, len(candidates)))]
        wx, wy = cx + dx // 2, cy + dy // 2
        grid[wy, wx] = FREE
        grid[ny, nx] = FREE
        visited.add((nx, ny))
        stack.append((nx, ny))

    _add_boundaries(grid)

    # Add sparse loops by opening a few random walls.
    walls = np.argwhere(grid == OCCUPIED)
    if walls.size:
        n_open = max(1, int(0.02 * len(walls)))
        picks = walls[rng.choice(len(walls), size=n_open, replace=False)]
        for y, x in picks:
            if x in (0, width - 1) or y in (0, height - 1):
                continue
            grid[y, x] = FREE

    return grid


def generate_ground_truth(
    map_type: str,
    width: int,
    height: int,
    obstacle_density: float,
    seed: int,
) -> np.ndarray:
    if map_type == "open":
        return generate_open_map(width, height, obstacle_density, seed)
    if map_type == "rooms":
        return generate_rooms_map(width, height, obstacle_density, seed)
    if map_type == "maze":
        return generate_maze_map(width, height, seed)
    raise ValueError(f"Unsupported map_type: {map_type}")
