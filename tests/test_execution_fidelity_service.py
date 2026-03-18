from __future__ import annotations

import numpy as np

from core.execution_fidelity_service import estimate_execution_features, estimate_execution_penalty
from core.map_manager import FREE, OCCUPIED, MapManager
from core.types import RobotState


def _cfg() -> dict:
    return {
        "planning": {
            "cfpa2_plus": {
                "execution": {
                    "enabled": True,
                    "clearance_ref": 4.0,
                    "density_radius": 2,
                    "narrow_clearance_threshold": 2.0,
                    "teammate_distance_threshold": 5.0,
                    "path_sample_stride": 1,
                    "normalize_features": True,
                    "w_clearance": 1.0,
                    "w_density": 1.0,
                    "w_turn": 1.0,
                    "w_narrow": 1.0,
                    "w_team": 1.0,
                    "w_slowdown": 1.0,
                }
            }
        }
    }


def _open_map(width: int = 30, height: int = 30) -> MapManager:
    truth = np.zeros((height, width), dtype=np.int8)
    truth[0, :] = OCCUPIED
    truth[-1, :] = OCCUPIED
    truth[:, 0] = OCCUPIED
    truth[:, -1] = OCCUPIED
    map_mgr = MapManager(truth)
    map_mgr.known[:, :] = truth
    return map_mgr


def test_clearance_penalty_increases_in_tighter_corridor() -> None:
    robot = RobotState(robot_id=1, pose=(5, 10), heading_deg=0.0)
    wide_map = _open_map()
    tight_map = _open_map()
    tight_map.known[8:13, 7] = OCCUPIED
    tight_map.known[8:13, 9] = OCCUPIED

    path = [(5, 10), (6, 10), (7, 10), (8, 10), (9, 10)]
    wide = estimate_execution_features(robot, goal=(9, 10), path=path, map_mgr=wide_map, cfg=_cfg(), teammate_states=[])
    tight = estimate_execution_features(robot, goal=(9, 10), path=path, map_mgr=tight_map, cfg=_cfg(), teammate_states=[])

    assert float(tight["clearance_penalty"]) > float(wide["clearance_penalty"])


def test_density_penalty_increases_with_more_local_obstacles() -> None:
    robot = RobotState(robot_id=1, pose=(6, 6), heading_deg=0.0)
    sparse_map = _open_map()
    dense_map = _open_map()
    dense_map.known[6:10, 10:14] = OCCUPIED
    dense_map.known[9:13, 8:12] = OCCUPIED

    path = [(6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    sparse = estimate_execution_features(robot, goal=(10, 10), path=path, map_mgr=sparse_map, cfg=_cfg(), teammate_states=[])
    dense = estimate_execution_features(robot, goal=(10, 10), path=path, map_mgr=dense_map, cfg=_cfg(), teammate_states=[])

    assert float(dense["obstacle_density_penalty"]) > float(sparse["obstacle_density_penalty"])


def test_turn_complexity_penalty_increases_with_more_turns() -> None:
    robot = RobotState(robot_id=1, pose=(4, 4), heading_deg=0.0)
    map_mgr = _open_map()

    straight_path = [(4, 4), (5, 4), (6, 4), (7, 4), (8, 4)]
    zigzag_path = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)]

    straight = estimate_execution_features(robot, goal=(8, 4), path=straight_path, map_mgr=map_mgr, cfg=_cfg(), teammate_states=[])
    zigzag = estimate_execution_features(robot, goal=(7, 6), path=zigzag_path, map_mgr=map_mgr, cfg=_cfg(), teammate_states=[])

    assert float(zigzag["turn_complexity_penalty"]) > float(straight["turn_complexity_penalty"])


def test_teammate_proximity_penalty_increases_when_teammate_is_close() -> None:
    robot = RobotState(robot_id=1, pose=(4, 10), heading_deg=0.0)
    map_mgr = _open_map()
    path = [(4, 10), (5, 10), (6, 10), (7, 10), (8, 10)]

    far_teammate = RobotState(robot_id=2, pose=(20, 20), heading_deg=0.0)
    far_teammate.set_plan((22, 20), [(21, 20), (22, 20)])

    close_teammate = RobotState(robot_id=2, pose=(6, 11), heading_deg=180.0)
    close_teammate.set_plan((8, 11), [(7, 11), (8, 11)])

    far = estimate_execution_features(robot, goal=(8, 10), path=path, map_mgr=map_mgr, cfg=_cfg(), teammate_states=[far_teammate])
    close = estimate_execution_features(robot, goal=(8, 10), path=path, map_mgr=map_mgr, cfg=_cfg(), teammate_states=[close_teammate])

    assert float(close["teammate_proximity_penalty"]) > float(far["teammate_proximity_penalty"])


def test_execution_penalty_combines_feature_breakdown() -> None:
    robot = RobotState(robot_id=1, pose=(4, 4), heading_deg=0.0)
    map_mgr = _open_map()
    map_mgr.known[5:9, 7] = OCCUPIED
    path = [(4, 4), (5, 4), (6, 5), (6, 6), (7, 6)]

    features = estimate_execution_features(robot, goal=(7, 6), path=path, map_mgr=map_mgr, cfg=_cfg(), teammate_states=[])
    penalty, breakdown = estimate_execution_penalty(features, _cfg())

    assert penalty >= 0.0
    assert "execution_penalty" in breakdown
    assert float(breakdown["execution_penalty"]) > 0.0


def test_feature_and_total_clipping_do_not_exceed_linear_penalty() -> None:
    robot = RobotState(robot_id=1, pose=(4, 4), heading_deg=0.0)
    map_mgr = _open_map()
    map_mgr.known[4:9, 6:9] = OCCUPIED
    path = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)]

    features = estimate_execution_features(robot, goal=(7, 6), path=path, map_mgr=map_mgr, cfg=_cfg(), teammate_states=[])

    linear_cfg = _cfg()
    clipped_cfg = _cfg()
    total_cfg = _cfg()
    clipped_cfg["planning"]["cfpa2_plus"]["execution"]["normalization_mode"] = "feature_clipped"
    clipped_cfg["planning"]["cfpa2_plus"]["execution"]["feature_clip_max"] = 0.6
    total_cfg["planning"]["cfpa2_plus"]["execution"]["normalization_mode"] = "total_clipped"
    total_cfg["planning"]["cfpa2_plus"]["execution"]["total_clip_max"] = 0.6

    linear_penalty, _ = estimate_execution_penalty(features, linear_cfg)
    clipped_penalty, _ = estimate_execution_penalty(features, clipped_cfg)
    total_penalty, _ = estimate_execution_penalty(features, total_cfg)

    assert clipped_penalty <= linear_penalty
    assert total_penalty <= linear_penalty
