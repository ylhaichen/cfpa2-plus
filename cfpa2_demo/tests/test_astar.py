from __future__ import annotations

import numpy as np

from cfpa2_demo.core.grid_map import FREE, OCCUPIED, OccupancyGrid
from cfpa2_demo.core.planner_astar import astar_path


def test_astar_returns_path() -> None:
    truth = np.zeros((7, 7), dtype=np.int8)
    truth[3, 1:6] = OCCUPIED
    truth[3, 3] = FREE

    grid = OccupancyGrid(truth)
    grid.grid[:, :] = truth

    path = astar_path(grid, start=(1, 1), goal=(5, 5), neighborhood=8)
    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (5, 5)
