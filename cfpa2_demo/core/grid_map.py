from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

Cell = tuple[int, int]


class OccupancyGrid:
    def __init__(self, truth_map: np.ndarray):
        if truth_map.ndim != 2:
            raise ValueError("truth_map must be 2D")
        self.truth = truth_map.astype(np.int8, copy=True)
        self.height, self.width = self.truth.shape
        self.grid = np.full_like(self.truth, UNKNOWN)

        self._free_truth_count = int(np.count_nonzero(self.truth == FREE))

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, cell: Cell) -> int:
        x, y = cell
        return int(self.grid[y, x])

    def get_truth(self, cell: Cell) -> int:
        x, y = cell
        return int(self.truth[y, x])

    def set_truth_free(self, cell: Cell) -> None:
        x, y = cell
        if self.truth[y, x] == OCCUPIED:
            self.truth[y, x] = FREE
            self._free_truth_count += 1

    def is_known_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.get(cell) == FREE

    def is_unknown(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.get(cell) == UNKNOWN

    def is_occupied(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.get(cell) == OCCUPIED

    def _bresenham_line(self, start: Cell, end: Cell) -> list[Cell]:
        x0, y0 = start
        x1, y1 = end
        line: list[Cell] = []

        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0
        while True:
            line.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return line

    def _is_line_visible(self, start: Cell, end: Cell) -> bool:
        line = self._bresenham_line(start, end)
        # Any occupied cell before the endpoint blocks visibility.
        for cell in line[1:-1]:
            if self.get_truth(cell) == OCCUPIED:
                return False
        return True

    def observe_from(self, center: Cell, sensor_range: int, use_line_of_sight: bool = True) -> int:
        cx, cy = center
        newly_observed = 0
        rr = sensor_range * sensor_range

        min_x = max(0, cx - sensor_range)
        max_x = min(self.width - 1, cx + sensor_range)
        min_y = max(0, cy - sensor_range)
        max_y = min(self.height - 1, cy + sensor_range)

        for y in range(min_y, max_y + 1):
            dy = y - cy
            for x in range(min_x, max_x + 1):
                dx = x - cx
                if dx * dx + dy * dy > rr:
                    continue
                if use_line_of_sight and not self._is_line_visible(center, (x, y)):
                    continue
                if self.grid[y, x] == UNKNOWN:
                    newly_observed += 1
                self.grid[y, x] = self.truth[y, x]
        return newly_observed

    def known_free_count(self) -> int:
        return int(np.count_nonzero(self.grid == FREE))

    def explored_free_ratio(self) -> float:
        if self._free_truth_count <= 0:
            return 1.0
        return self.known_free_count() / float(self._free_truth_count)

    def known_ratio(self) -> float:
        total = self.width * self.height
        known = int(np.count_nonzero(self.grid != UNKNOWN))
        return known / float(total)

    def neighbors4(self, cell: Cell) -> list[Cell]:
        x, y = cell
        out: list[Cell] = []
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < self.width and 0 <= ny < self.height:
                out.append((nx, ny))
        return out

    def neighbors8(self, cell: Cell) -> list[Cell]:
        x, y = cell
        out: list[Cell] = []
        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):
                if nx == x and ny == y:
                    continue
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    out.append((nx, ny))
        return out

    def free_cells(self) -> np.ndarray:
        ys, xs = np.where(self.grid == FREE)
        return np.stack([xs, ys], axis=1)

    def ensure_starts_free(self, starts: Sequence[Cell]) -> None:
        for cell in starts:
            if not self.in_bounds(cell):
                raise ValueError(f"Start out of bounds: {cell}")
            self.set_truth_free(cell)

    def nearest_known_free(self, cell: Cell, max_radius: int = 20) -> Cell | None:
        if self.is_known_free(cell):
            return cell
        cx, cy = cell
        for radius in range(1, max_radius + 1):
            min_x = max(0, cx - radius)
            max_x = min(self.width - 1, cx + radius)
            min_y = max(0, cy - radius)
            max_y = min(self.height - 1, cy + radius)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if self.grid[y, x] == FREE:
                        return (x, y)
        return None
