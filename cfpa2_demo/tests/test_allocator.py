from __future__ import annotations

import numpy as np

from cfpa2_demo.core.allocator import assign_dual_joint_cfpa2
from cfpa2_demo.core.grid_map import FREE, UNKNOWN, OccupancyGrid
from cfpa2_demo.core.robot import RobotState


def _cfg() -> dict:
    return {
        "robots": {"sensor_range": 4},
        "frontier": {
            "ig_radius": 4,
            "w_ig": 1.0,
            "w_c": 0.4,
            "w_sw": 0.2,
        },
        "allocator": {
            "lambda_overlap": 0.5,
            "mu_interference": 0.0,
            "sigma_overlap": 8.0,
        },
    }


def test_dual_joint_assigns_distinct_targets() -> None:
    truth = np.zeros((15, 15), dtype=np.int8)
    grid = OccupancyGrid(truth)

    # Make most space known free, keep upper strip unknown to create IG gradients.
    grid.grid[:, :] = FREE
    grid.grid[0:3, :] = UNKNOWN

    r1 = RobotState(robot_id=1, pose=(2, 10))
    r2 = RobotState(robot_id=2, pose=(12, 10))

    candidates = [(2, 4), (12, 4), (7, 4)]
    a1, a2, score, _ = assign_dual_joint_cfpa2(
        robot1=r1,
        robot2=r2,
        candidates=candidates,
        grid=grid,
        cfg=_cfg(),
        reservation_table=None,
        neighborhood=8,
    )

    assert a1.valid and a2.valid
    assert a1.target != a2.target
    assert np.isfinite(score)
