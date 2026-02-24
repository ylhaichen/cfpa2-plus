from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .grid_map import Cell, OccupancyGrid
from .robot import RobotState
from .utility import CandidateEvaluation, utility


@dataclass
class TargetAssignment:
    robot_id: int
    target: Cell | None
    path: list[Cell]
    utility: float
    valid: bool


def _allowed_by_reservation(
    candidate: Cell,
    robot_id: int,
    reservation_table: dict[Cell, dict[str, Any]] | None,
) -> bool:
    if not reservation_table:
        return True
    entry = reservation_table.get(candidate)
    if entry is None:
        return True
    return int(entry.get("robot_id", -1)) == robot_id


def _compute_utilities(
    robot: RobotState,
    candidates: list[Cell],
    grid: OccupancyGrid,
    cfg: dict,
    reservation_table: dict[Cell, dict[str, Any]] | None,
    neighborhood: int,
) -> dict[Cell, CandidateEvaluation]:
    out: dict[Cell, CandidateEvaluation] = {}
    for c in candidates:
        if not _allowed_by_reservation(c, robot.robot_id, reservation_table):
            continue
        e = utility(robot, c, grid, cfg, neighborhood=neighborhood)
        if e is None:
            continue
        out[c] = e
    return out


def _idle_assignment(robot_id: int) -> TargetAssignment:
    return TargetAssignment(robot_id=robot_id, target=None, path=[], utility=float("-inf"), valid=False)


def assign_single_robot(
    robot: RobotState,
    candidates: list[Cell],
    grid: OccupancyGrid,
    cfg: dict,
    reservation_table: dict[Cell, dict[str, Any]] | None = None,
    neighborhood: int = 8,
) -> TargetAssignment:
    utilities = _compute_utilities(robot, candidates, grid, cfg, reservation_table, neighborhood)
    if not utilities:
        return _idle_assignment(robot.robot_id)

    best_target, best_eval = max(utilities.items(), key=lambda kv: kv[1].utility)
    return TargetAssignment(
        robot_id=robot.robot_id,
        target=best_target,
        path=best_eval.path,
        utility=best_eval.utility,
        valid=True,
    )


def assign_dual_greedy(
    robot1: RobotState,
    robot2: RobotState,
    candidates: list[Cell],
    grid: OccupancyGrid,
    cfg: dict,
    reservation_table: dict[Cell, dict[str, Any]] | None = None,
    neighborhood: int = 8,
) -> tuple[TargetAssignment, TargetAssignment, dict[str, Any]]:
    u1 = _compute_utilities(robot1, candidates, grid, cfg, reservation_table, neighborhood)
    u2 = _compute_utilities(robot2, candidates, grid, cfg, reservation_table, neighborhood)

    if not u1 and not u2:
        return _idle_assignment(robot1.robot_id), _idle_assignment(robot2.robot_id), {"conflict": False}

    ranked1 = sorted(u1.items(), key=lambda kv: kv[1].utility, reverse=True)
    ranked2 = sorted(u2.items(), key=lambda kv: kv[1].utility, reverse=True)

    def pick_first(ranked: list[tuple[Cell, CandidateEvaluation]], excluded: set[Cell]) -> tuple[Cell, CandidateEvaluation] | None:
        for cell, ev in ranked:
            if cell not in excluded:
                return cell, ev
        return None

    p1 = pick_first(ranked1, set())
    p2 = pick_first(ranked2, set())
    conflict = False

    if p1 and p2 and p1[0] == p2[0]:
        conflict = True
        if p1[1].utility >= p2[1].utility:
            p2 = pick_first(ranked2, {p1[0]})
        else:
            p1 = pick_first(ranked1, {p2[0]})

    a1 = _idle_assignment(robot1.robot_id)
    a2 = _idle_assignment(robot2.robot_id)

    if p1 is not None:
        a1 = TargetAssignment(
            robot_id=robot1.robot_id,
            target=p1[0],
            path=p1[1].path,
            utility=p1[1].utility,
            valid=True,
        )

    if p2 is not None:
        a2 = TargetAssignment(
            robot_id=robot2.robot_id,
            target=p2[0],
            path=p2[1].path,
            utility=p2[1].utility,
            valid=True,
        )

    return a1, a2, {"conflict": conflict}


def overlap_penalty(fi: Cell, fj: Cell, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    dx = fi[0] - fj[0]
    dy = fi[1] - fj[1]
    d2 = dx * dx + dy * dy
    return math.exp(-d2 / (2.0 * sigma * sigma))


def path_interference_penalty(*_args: object, **_kwargs: object) -> float:
    return 0.0


def assign_dual_joint_cfpa2(
    robot1: RobotState,
    robot2: RobotState,
    candidates: list[Cell],
    grid: OccupancyGrid,
    cfg: dict,
    reservation_table: dict[Cell, dict[str, Any]] | None = None,
    neighborhood: int = 8,
) -> tuple[TargetAssignment, TargetAssignment, float, dict[str, Any]]:
    allocator_cfg = cfg["allocator"]
    sensor_range = int(cfg["robots"]["sensor_range"])
    sigma = allocator_cfg.get("sigma_overlap")
    if sigma is None:
        sigma = 2.0 * sensor_range
    sigma = float(sigma)
    lambda_overlap = float(allocator_cfg["lambda_overlap"])
    mu_interference = float(allocator_cfg.get("mu_interference", 0.0))

    u1 = _compute_utilities(robot1, candidates, grid, cfg, reservation_table, neighborhood)
    u2 = _compute_utilities(robot2, candidates, grid, cfg, reservation_table, neighborhood)

    best_pair: tuple[Cell, Cell] | None = None
    best_score = float("-inf")
    best_dbg: dict[str, float] = {}

    top_pairs: list[tuple[float, Cell, Cell]] = []

    for fi, ev1 in u1.items():
        for fj, ev2 in u2.items():
            if fi == fj:
                continue
            overlap = overlap_penalty(fi, fj, sigma)
            interference = path_interference_penalty(fi, fj)
            joint = ev1.utility + ev2.utility
            score = joint - lambda_overlap * overlap - mu_interference * interference
            top_pairs.append((score, fi, fj))

            if score > best_score:
                best_score = score
                best_pair = (fi, fj)
                best_dbg = {
                    "joint_utility": joint,
                    "overlap_penalty": overlap,
                    "interference_penalty": interference,
                }

    if best_pair is not None:
        fi, fj = best_pair
        a1 = TargetAssignment(robot_id=robot1.robot_id, target=fi, path=u1[fi].path, utility=u1[fi].utility, valid=True)
        a2 = TargetAssignment(robot_id=robot2.robot_id, target=fj, path=u2[fj].path, utility=u2[fj].utility, valid=True)
        top_pairs.sort(key=lambda t: t[0], reverse=True)
        debug = {
            "top_pairs": [
                {
                    "score": float(s),
                    "target1": tuple(fi_),
                    "target2": tuple(fj_),
                }
                for s, fi_, fj_ in top_pairs[:5]
            ]
        }
        debug.update(best_dbg)
        return a1, a2, best_score, debug

    # Fallback strategy: allow one robot to continue if pair assignment impossible.
    single1 = _idle_assignment(robot1.robot_id)
    single2 = _idle_assignment(robot2.robot_id)

    if u1:
        t1, e1 = max(u1.items(), key=lambda kv: kv[1].utility)
        single1 = TargetAssignment(robot_id=robot1.robot_id, target=t1, path=e1.path, utility=e1.utility, valid=True)
    if u2:
        t2, e2 = max(u2.items(), key=lambda kv: kv[1].utility)
        single2 = TargetAssignment(robot_id=robot2.robot_id, target=t2, path=e2.path, utility=e2.utility, valid=True)

    # Select better single-assignment fallback.
    if single1.valid and (not single2.valid or single1.utility >= single2.utility):
        return single1, _idle_assignment(robot2.robot_id), single1.utility, {"fallback": "robot1_only"}
    if single2.valid:
        return _idle_assignment(robot1.robot_id), single2, single2.utility, {"fallback": "robot2_only"}

    return _idle_assignment(robot1.robot_id), _idle_assignment(robot2.robot_id), float("-inf"), {"fallback": "none"}
