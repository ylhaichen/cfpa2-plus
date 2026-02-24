from __future__ import annotations

import numpy as np

from cfpa2_demo.core.frontier import cluster_frontiers, detect_frontiers
from cfpa2_demo.core.grid_map import FREE, UNKNOWN, OccupancyGrid


def test_frontier_detection_basic() -> None:
    truth = np.zeros((5, 5), dtype=np.int8)
    grid = OccupancyGrid(truth)

    grid.grid[:, :] = FREE
    grid.grid[0, :] = UNKNOWN

    frontiers = detect_frontiers(grid, neighborhood=8)
    assert (2, 1) in frontiers
    assert (2, 2) not in frontiers


def test_frontier_clustering() -> None:
    frontiers = [(1, 1), (2, 1), (10, 10), (11, 10)]
    clusters = cluster_frontiers(frontiers, method="bfs", neighborhood=8, min_cluster_size=2)
    assert len(clusters) == 2
