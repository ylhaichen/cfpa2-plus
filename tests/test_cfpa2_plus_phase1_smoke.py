from __future__ import annotations

import numpy as np

from core.frontier_manager import build_frontier_candidates
from core.map_manager import OCCUPIED, MapManager
from core.types import GoalAssignment, PlannerInput, RobotState
from planners import build_planner


def _cfg() -> dict:
    return {
        "environment": {"map_name": "phase1_smoke"},
        "robots": {
            "num_robots": 2,
            "sensor_range": 6,
            "sensor_fov_deg": 360.0,
            "use_line_of_sight": True,
            "observation_miss_prob": 0.0,
            "clearance_cells": 0,
            "max_speed_cells_per_step": 1.0,
            "turn_rate_deg_per_step": 35.0,
        },
        "frontier": {
            "neighborhood": 8,
            "min_cluster_size": 1,
            "target_frontier_count_min": 1,
            "target_frontier_count_max": 8,
            "representative_min_distance": 0.0,
            "ig_radius": 4,
        },
        "planning": {
            "planner_name": "cfpa2_plus",
            "planner_label": "cfpa2_plus_phase1",
            "reservation_ttl": 8,
            "weights": {"w_ig": 1.0, "w_cost": 0.4, "w_switch": 0.2, "w_turn": 0.1},
            "penalties": {"lambda_overlap": 0.5, "sigma_overlap": 8.0, "mu_interference": 0.1, "interference_distance": 2.5},
            "rollout": {"horizon": 1},
            "cfpa2_plus": {
                "enabled_components": {"execution_aware": True},
                "score_mode": {"baseline_weight": 1.0, "execution_weight": 1.0, "lambda_exec": 4.0},
                "execution": {
                    "enabled": True,
                    "clearance_ref": 4.0,
                    "density_radius": 2,
                    "narrow_clearance_threshold": 2.0,
                    "teammate_distance_threshold": 5.0,
                    "path_sample_stride": 1,
                    "normalize_features": True,
                    "w_clearance": 1.0,
                    "w_density": 0.7,
                    "w_turn": 0.5,
                    "w_narrow": 1.0,
                    "w_team": 0.8,
                    "w_slowdown": 0.8,
                },
            },
        },
        "predictor": {"type": "path_follow"},
        "replanning": {
            "enable_event_replan": True,
            "periodic_replan_interval": 10,
            "frontier_change_threshold": 0.25,
            "stuck_threshold": 8,
            "invalidation_path_threshold": 3,
            "invalidation_distance_threshold": 2.0,
        },
        "termination": {"step_dt": 1.0},
    }


def _map_mgr() -> MapManager:
    truth = np.zeros((28, 28), dtype=np.int8)
    truth[0, :] = OCCUPIED
    truth[-1, :] = OCCUPIED
    truth[:, 0] = OCCUPIED
    truth[:, -1] = OCCUPIED
    truth[12, 5:23] = OCCUPIED
    truth[12, 13] = 0

    map_mgr = MapManager(truth)
    r1 = RobotState(robot_id=1, pose=(4, 4), heading_deg=0.0)
    r2 = RobotState(robot_id=2, pose=(8, 4), heading_deg=0.0)
    rng = np.random.default_rng(0)
    map_mgr.observe_from(r1.pose, r1.heading_deg, 6, 360.0, True, 0.0, rng)
    map_mgr.observe_from(r2.pose, r2.heading_deg, 6, 360.0, True, 0.0, rng)
    return map_mgr


def test_cfpa2_plus_phase1_smoke() -> None:
    cfg = _cfg()
    map_mgr = _map_mgr()
    robots = [RobotState(robot_id=1, pose=(4, 4), heading_deg=0.0), RobotState(robot_id=2, pose=(8, 4), heading_deg=0.0)]
    frontier_cells, candidates = build_frontier_candidates(map_mgr, cfg)
    assert frontier_cells
    assert candidates

    assignments = {
        1: GoalAssignment(1, None, [], float("-inf"), False, {}),
        2: GoalAssignment(2, None, [], float("-inf"), False, {}),
    }

    planner = build_planner(cfg)
    out = planner.plan(
        PlannerInput(
            shared_map=map_mgr,
            robot_states=robots,
            frontier_candidates=candidates,
            current_assignments=assignments,
            reservation_state={},
            step_idx=0,
            sim_time=0.0,
            config=cfg,
        )
    )

    assert out.assignments
    assert out.planner_name == "cfpa2_plus"
    assert "selected_execution_penalty_mean" in out.debug

    valid = [a for a in out.assignments.values() if a.valid]
    assert valid
    assert all("execution_penalty" in a.breakdown for a in valid)
