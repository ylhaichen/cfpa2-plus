from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .grid_map import Cell, FREE, UNKNOWN, OccupancyGrid


@dataclass
class FrontierCluster:
    cells: list[Cell]
    representative: Cell


def _cluster_sort_key(cluster: FrontierCluster) -> tuple[int, int, int]:
    rx, ry = cluster.representative
    return (-len(cluster.cells), ry, rx)


def _distance_sq(a: Cell, b: Cell) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return dx * dx + dy * dy


def reduce_frontier_clusters(
    clusters: list[FrontierCluster],
    target_frontier_count_min: int,
    target_frontier_count_max: int | None,
    representative_min_distance: float,
) -> list[FrontierCluster]:
    if not clusters:
        return []
    if target_frontier_count_max is None:
        return clusters

    max_count = max(1, int(target_frontier_count_max))
    min_count = max(0, int(target_frontier_count_min))
    min_count = min(min_count, max_count)

    ranked = sorted(clusters, key=_cluster_sort_key)
    selected: list[FrontierCluster] = []
    selected_reps: set[Cell] = set()
    min_dist_sq = float(representative_min_distance) ** 2

    # 1) Greedy spread-out selection with NMS-like distance suppression.
    for cluster in ranked:
        if len(selected) >= max_count:
            break
        rep = cluster.representative
        if representative_min_distance > 0.0:
            too_close = any(_distance_sq(rep, s.representative) < min_dist_sq for s in selected)
            if too_close:
                continue
        selected.append(cluster)
        selected_reps.add(rep)

    # 2) If we suppressed too aggressively, backfill by size until min_count.
    if len(selected) < min_count:
        for cluster in ranked:
            rep = cluster.representative
            if rep in selected_reps:
                continue
            selected.append(cluster)
            selected_reps.add(rep)
            if len(selected) >= min_count or len(selected) >= max_count:
                break

    return selected[:max_count]


def _neighbors(cell: Cell, neighborhood: int) -> list[Cell]:
    x, y = cell
    if neighborhood == 4:
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    out: list[Cell] = []
    for ny in range(y - 1, y + 2):
        for nx in range(x - 1, x + 2):
            if nx == x and ny == y:
                continue
            out.append((nx, ny))
    return out


def is_frontier_cell(grid: OccupancyGrid, cell: Cell, neighborhood: int = 8) -> bool:
    if not grid.in_bounds(cell):
        return False
    if grid.get(cell) != FREE:
        return False
    for n in _neighbors(cell, neighborhood):
        if grid.in_bounds(n) and grid.get(n) == UNKNOWN:
            return True
    return False


def detect_frontiers(grid: OccupancyGrid, neighborhood: int = 8) -> list[Cell]:
    frontiers: list[Cell] = []
    ys, xs = np.where(grid.grid == FREE)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cell = (x, y)
        if is_frontier_cell(grid, cell, neighborhood):
            frontiers.append(cell)
    return frontiers


def cluster_frontiers(
    frontier_cells: list[Cell],
    method: str = "bfs",
    neighborhood: int = 8,
    min_cluster_size: int = 1,
) -> list[list[Cell]]:
    if method != "bfs":
        raise ValueError("Only bfs clustering is implemented in v1")

    frontier_set = set(frontier_cells)
    visited: set[Cell] = set()
    clusters: list[list[Cell]] = []

    for seed in frontier_cells:
        if seed in visited:
            continue
        q: deque[Cell] = deque([seed])
        visited.add(seed)
        comp: list[Cell] = []

        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in _neighbors(cur, neighborhood):
                if nxt in frontier_set and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)

        if len(comp) >= min_cluster_size:
            clusters.append(comp)

    return clusters


def representative(cluster: list[Cell], grid: OccupancyGrid, neighborhood: int = 8) -> Cell | None:
    if not cluster:
        return None

    coords = np.array(cluster, dtype=float)
    centroid = coords.mean(axis=0)
    cx, cy = int(round(float(centroid[0]))), int(round(float(centroid[1])))

    if (cx, cy) in cluster and is_frontier_cell(grid, (cx, cy), neighborhood):
        return (cx, cy)

    # Medoid-like fallback: closest valid frontier cell in the cluster to centroid.
    best_cell: Cell | None = None
    best_dist = float("inf")
    for c in cluster:
        if not is_frontier_cell(grid, c, neighborhood):
            continue
        d = (c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2
        if d < best_dist:
            best_dist = d
            best_cell = c

    return best_cell


def build_frontier_clusters(
    grid: OccupancyGrid,
    neighborhood: int = 8,
    method: str = "bfs",
    min_cluster_size: int = 1,
    target_frontier_count_min: int = 0,
    target_frontier_count_max: int | None = None,
    representative_min_distance: float = 0.0,
) -> tuple[list[Cell], list[FrontierCluster]]:
    frontier_cells = detect_frontiers(grid, neighborhood=neighborhood)
    comps = cluster_frontiers(
        frontier_cells,
        method=method,
        neighborhood=neighborhood,
        min_cluster_size=min_cluster_size,
    )

    clusters: list[FrontierCluster] = []
    for comp in comps:
        rep = representative(comp, grid, neighborhood=neighborhood)
        if rep is None:
            continue
        clusters.append(FrontierCluster(cells=comp, representative=rep))

    clusters = reduce_frontier_clusters(
        clusters,
        target_frontier_count_min=target_frontier_count_min,
        target_frontier_count_max=target_frontier_count_max,
        representative_min_distance=representative_min_distance,
    )

    return frontier_cells, clusters
