from __future__ import annotations

import math
from typing import Iterable

from .types import Cell, RobotState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _execution_cfg(cfg: dict) -> dict:
    return dict(cfg.get("planning", {}).get("cfpa2_plus", {}).get("execution", {}))


def _trimmed_path(robot: RobotState, path: list[Cell]) -> list[Cell]:
    if not path:
        return [robot.pose]
    if path[0] == robot.pose:
        trimmed = list(path[1:])
        return trimmed or [robot.pose]
    return list(path)


def _sample_path_cells(cells: list[Cell], stride: int, use_prefix_only: bool, prefix_steps: int | None) -> list[Cell]:
    if not cells:
        return []
    limit = len(cells)
    if use_prefix_only and prefix_steps is not None:
        limit = min(limit, max(1, int(prefix_steps)))
    sampled = list(cells[:limit:max(1, int(stride))])
    if cells[min(limit, len(cells)) - 1] not in sampled:
        sampled.append(cells[min(limit, len(cells)) - 1])
    return sampled


def _closest_known_obstacle_distance(map_mgr, cell: Cell, max_radius: int) -> float:
    cx, cy = cell
    best = float(max_radius + 1)

    for dy in range(-max_radius, max_radius + 1):
        for dx in range(-max_radius, max_radius + 1):
            gx = cx + dx
            gy = cy + dy
            if gx < 0 or gx >= map_mgr.width or gy < 0 or gy >= map_mgr.height:
                best = min(best, math.hypot(float(dx), float(dy)))
                continue
            if int(map_mgr.known[gy, gx]) == 1:
                best = min(best, math.hypot(float(dx), float(dy)))

    return best


def _occupied_density(map_mgr, cell: Cell, radius: int) -> float:
    cx, cy = cell
    occupied = 0
    total = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gx = cx + dx
            gy = cy + dy
            total += 1
            if gx < 0 or gx >= map_mgr.width or gy < 0 or gy >= map_mgr.height:
                occupied += 1
                continue
            if int(map_mgr.known[gy, gx]) == 1:
                occupied += 1

    return float(occupied) / float(max(1, total))


def _direction_sequence(path_cells: list[Cell]) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    for a, b in zip(path_cells[:-1], path_cells[1:]):
        dx = int(b[0] - a[0])
        dy = int(b[1] - a[1])
        if dx == 0 and dy == 0:
            continue
        dirs.append((0 if dx == 0 else int(dx / abs(dx)), 0 if dy == 0 else int(dy / abs(dy))))
    return dirs


def _turn_complexity(robot: RobotState, path_cells: list[Cell]) -> tuple[float, int, float]:
    dirs = _direction_sequence(path_cells)
    if not dirs:
        return 0.0, 0, 0.0

    headings = [math.degrees(math.atan2(float(dy), float(dx))) for dx, dy in dirs]
    initial_turn = abs(((headings[0] - float(robot.heading_deg) + 180.0) % 360.0) - 180.0)

    turn_count = 0
    accum_angle = initial_turn
    for prev, cur in zip(headings[:-1], headings[1:]):
        diff = abs(((cur - prev + 180.0) % 360.0) - 180.0)
        accum_angle += diff
        if diff > 1e-6:
            turn_count += 1

    seg_count = max(1, len(dirs))
    turn_ratio = float(turn_count + (1 if initial_turn > 1e-6 else 0)) / float(seg_count)
    angle_ratio = float(accum_angle) / float(180.0 * seg_count)
    penalty = _clip01(0.45 * turn_ratio + 0.55 * angle_ratio)
    return penalty, int(turn_count), float(accum_angle)


def _points_from_teammates(teammates: Iterable[RobotState]) -> list[Cell]:
    points: list[Cell] = []
    for teammate in teammates:
        points.append(teammate.pose)
        if teammate.current_target is not None:
            points.append(teammate.current_target)
        if teammate.path:
            points.extend(teammate.path)
    return points


def _min_distance_to_points(cells: list[Cell], points: list[Cell]) -> float:
    if not cells or not points:
        return float("inf")
    best = float("inf")
    for a in cells:
        for b in points:
            best = min(best, math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))
    return best


def estimate_execution_features(
    robot: RobotState,
    goal: Cell | None,
    path: list[Cell],
    map_mgr,
    cfg: dict,
    teammate_states: list[RobotState] | None = None,
) -> dict[str, float | int | bool]:
    execution_cfg = _execution_cfg(cfg)
    stride = max(1, int(execution_cfg.get("path_sample_stride", 2)))
    use_prefix_only = bool(execution_cfg.get("use_path_prefix_only", False))
    prefix_steps_raw = execution_cfg.get("path_prefix_steps")
    prefix_steps = None if prefix_steps_raw is None else int(prefix_steps_raw)
    sampled = _sample_path_cells(
        _trimmed_path(robot, path),
        stride=stride,
        use_prefix_only=use_prefix_only,
        prefix_steps=prefix_steps,
    )
    if goal is not None and goal not in sampled:
        sampled.append(goal)

    clearance_ref = float(execution_cfg.get("clearance_ref", 4.0))
    density_radius = max(1, int(execution_cfg.get("density_radius", 3)))
    narrow_clearance_threshold = float(execution_cfg.get("narrow_clearance_threshold", 2.5))
    teammate_distance_threshold = float(execution_cfg.get("teammate_distance_threshold", 6.0))
    max_clearance_scan_radius = max(
        1,
        int(
            execution_cfg.get(
                "max_clearance_scan_radius",
                max(math.ceil(clearance_ref * 2.0), math.ceil(narrow_clearance_threshold * 2.0)),
            )
        ),
    )

    clearances = [_closest_known_obstacle_distance(map_mgr, cell, max_clearance_scan_radius) for cell in sampled]
    mean_clearance = float(sum(clearances) / len(clearances)) if clearances else float(clearance_ref)
    min_clearance = float(min(clearances)) if clearances else float(clearance_ref)
    clearance_score = min(1.0, ((0.65 * min_clearance) + (0.35 * mean_clearance)) / max(1e-6, clearance_ref))
    clearance_penalty = _clip01(1.0 - clearance_score)

    densities = [_occupied_density(map_mgr, cell, density_radius) for cell in sampled]
    obstacle_density_penalty = _clip01(sum(densities) / len(densities)) if densities else 0.0

    turn_complexity_penalty, turn_count, accumulated_turn_deg = _turn_complexity(robot, _trimmed_path(robot, path))

    narrow_count = sum(1 for c in clearances if c <= narrow_clearance_threshold)
    corridor_narrowness_penalty = _clip01(float(narrow_count) / float(max(1, len(clearances))))

    teammate_states = list(teammate_states or [])
    teammate_points = _points_from_teammates(teammate_states)
    teammate_min_distance = _min_distance_to_points(sampled, teammate_points)
    if math.isinf(teammate_min_distance):
        teammate_proximity_penalty = 0.0
    else:
        teammate_proximity_penalty = _clip01(
            (teammate_distance_threshold - float(teammate_min_distance)) / max(1e-6, teammate_distance_threshold)
        )

    high_density_threshold = float(execution_cfg.get("slowdown_density_threshold", 0.20))
    high_density_ratio = float(sum(1 for d in densities if d >= high_density_threshold)) / float(max(1, len(densities)))
    slowdown_exposure_penalty = _clip01(
        0.45 * corridor_narrowness_penalty
        + 0.35 * high_density_ratio
        + 0.20 * teammate_proximity_penalty
    )

    return {
        "sample_count": len(sampled),
        "mean_clearance": float(mean_clearance),
        "min_clearance": float(min_clearance),
        "clearance_penalty": float(clearance_penalty),
        "obstacle_density_penalty": float(obstacle_density_penalty),
        "turn_complexity_penalty": float(turn_complexity_penalty),
        "turn_count": int(turn_count),
        "accumulated_turn_deg": float(accumulated_turn_deg),
        "corridor_narrowness_penalty": float(corridor_narrowness_penalty),
        "narrow_segment_ratio": float(corridor_narrowness_penalty),
        "teammate_min_distance": float(teammate_min_distance if not math.isinf(teammate_min_distance) else 999.0),
        "teammate_proximity_penalty": float(teammate_proximity_penalty),
        "slowdown_exposure_penalty": float(slowdown_exposure_penalty),
        "goal_is_sampled": bool(goal is not None and goal in sampled),
    }


def estimate_execution_penalty(features: dict[str, float | int | bool], cfg: dict) -> tuple[float, dict[str, float]]:
    execution_cfg = _execution_cfg(cfg)
    weights = {
        "clearance_penalty": float(execution_cfg.get("w_clearance", 1.0)),
        "obstacle_density_penalty": float(execution_cfg.get("w_density", 0.7)),
        "turn_complexity_penalty": float(execution_cfg.get("w_turn", 0.5)),
        "corridor_narrowness_penalty": float(execution_cfg.get("w_narrow", 1.0)),
        "teammate_proximity_penalty": float(execution_cfg.get("w_team", 0.8)),
        "slowdown_exposure_penalty": float(execution_cfg.get("w_slowdown", 0.8)),
    }

    normalization_mode = str(execution_cfg.get("normalization_mode", "linear"))
    feature_clip_max = float(execution_cfg.get("feature_clip_max", 0.85))
    total_clip_max = float(execution_cfg.get("total_clip_max", 0.85))
    soft_saturation_gamma = float(execution_cfg.get("soft_saturation_gamma", 1.35))

    weighted_sum = 0.0
    weight_total = 0.0
    breakdown: dict[str, float] = {}
    for key, weight in weights.items():
        raw_value = float(features.get(key, 0.0))
        value = float(raw_value)
        if normalization_mode == "feature_clipped":
            value = min(value, feature_clip_max)
        weighted_sum += weight * value
        weight_total += weight
        breakdown[key] = raw_value

    penalty_linear = weighted_sum / max(1e-6, weight_total)
    if normalization_mode == "total_clipped":
        penalty = min(penalty_linear, total_clip_max)
    elif normalization_mode == "soft_saturation":
        penalty = penalty_linear ** max(1.0, soft_saturation_gamma)
    else:
        penalty = penalty_linear

    penalty_out = penalty if bool(execution_cfg.get("normalize_features", True)) else weighted_sum
    breakdown["weighted_penalty_linear"] = float(_clip01(penalty_linear))
    breakdown["weighted_penalty_processed"] = float(_clip01(penalty))
    breakdown["normalization_mode_linear"] = 1.0 if normalization_mode == "linear" else 0.0
    breakdown["normalization_mode_feature_clipped"] = 1.0 if normalization_mode == "feature_clipped" else 0.0
    breakdown["normalization_mode_total_clipped"] = 1.0 if normalization_mode == "total_clipped" else 0.0
    breakdown["normalization_mode_soft_saturation"] = 1.0 if normalization_mode == "soft_saturation" else 0.0
    breakdown["execution_penalty"] = float(_clip01(penalty_out))
    return breakdown["execution_penalty"], breakdown
