from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from core.assignment_solver import compute_candidate_utilities, solve_joint_cfpa2
from core.execution_fidelity_service import estimate_execution_features, estimate_execution_penalty
from core.score_composer import compose_cfpa2_plus_utility
from core.types import GoalAssignment, PlannerInput, PlannerOutput
from core.utility_service import CandidateEvaluation

from .base_planner import BasePlanner


@dataclass
class _CandidateDebug:
    baseline_utility: float
    adjusted_utility: float
    execution_penalty: float
    execution_features: dict[str, float | int | bool]
    composition: dict[str, float | bool]
    path: list[tuple[int, int]]
    information_gain: float
    travel_cost: float
    switch_penalty: float
    turn_penalty: float


class CFPA2PlusPlanner(BasePlanner):
    name = "cfpa2_plus"

    def _score_candidates(
        self,
        robot,
        robots,
        candidates,
        map_mgr,
        cfg: dict,
        reservation_state,
    ) -> tuple[dict[tuple[int, int], CandidateEvaluation], dict[tuple[int, int], _CandidateDebug]]:
        baseline = compute_candidate_utilities(robot, candidates, cfg, map_mgr, reservation_state)
        teammates = [r for r in robots if r.robot_id != robot.robot_id]

        scored: dict[tuple[int, int], CandidateEvaluation] = {}
        debug: dict[tuple[int, int], _CandidateDebug] = {}

        for target, ev in baseline.items():
            execution_features = estimate_execution_features(
                robot=robot,
                goal=target,
                path=ev.path,
                map_mgr=map_mgr,
                cfg=cfg,
                teammate_states=teammates,
            )
            execution_penalty, execution_breakdown = estimate_execution_penalty(execution_features, cfg)
            adjusted_utility, composition = compose_cfpa2_plus_utility(
                baseline_utility=float(ev.utility),
                execution_penalty=float(execution_penalty),
                cfg=cfg,
            )

            scored[target] = CandidateEvaluation(
                utility=float(adjusted_utility),
                information_gain=float(ev.information_gain),
                travel_cost=float(ev.travel_cost),
                switch_penalty=float(ev.switch_penalty),
                turn_penalty=float(ev.turn_penalty),
                path=list(ev.path),
            )
            debug[target] = _CandidateDebug(
                baseline_utility=float(ev.utility),
                adjusted_utility=float(adjusted_utility),
                execution_penalty=float(execution_penalty),
                execution_features={**execution_features, **execution_breakdown},
                composition=composition,
                path=list(ev.path),
                information_gain=float(ev.information_gain),
                travel_cost=float(ev.travel_cost),
                switch_penalty=float(ev.switch_penalty),
                turn_penalty=float(ev.turn_penalty),
            )

        return scored, debug

    def _assignment_from_debug(self, robot_id: int, target, cand: _CandidateDebug) -> GoalAssignment:
        return GoalAssignment(
            robot_id=robot_id,
            target=target,
            path=list(cand.path),
            utility=float(cand.adjusted_utility),
            valid=True,
            breakdown={
                "ig": float(cand.information_gain),
                "travel_cost": float(cand.travel_cost),
                "switch_penalty": float(cand.switch_penalty),
                "turn_penalty": float(cand.turn_penalty),
                "baseline_utility": float(cand.baseline_utility),
                "execution_penalty": float(cand.execution_penalty),
                "utility_plus": float(cand.adjusted_utility),
                "clearance_penalty": float(cand.execution_features.get("clearance_penalty", 0.0)),
                "density_penalty": float(cand.execution_features.get("obstacle_density_penalty", 0.0)),
                "turn_complexity_penalty": float(cand.execution_features.get("turn_complexity_penalty", 0.0)),
                "narrowness_penalty": float(cand.execution_features.get("corridor_narrowness_penalty", 0.0)),
                "teammate_proximity_penalty": float(cand.execution_features.get("teammate_proximity_penalty", 0.0)),
                "slowdown_exposure_penalty": float(cand.execution_features.get("slowdown_exposure_penalty", 0.0)),
            },
        )

    def _selected_execution_debug(
        self,
        assignments: dict[int, GoalAssignment],
        debug_maps: dict[int, dict[tuple[int, int], _CandidateDebug]],
    ) -> dict:
        by_robot: dict[int, float] = {}
        features_by_robot: dict[int, dict[str, float]] = {}
        feature_keys = [
            "clearance_penalty",
            "obstacle_density_penalty",
            "turn_complexity_penalty",
            "corridor_narrowness_penalty",
            "teammate_proximity_penalty",
            "slowdown_exposure_penalty",
        ]

        for rid, assignment in assignments.items():
            if not assignment.valid or assignment.target is None:
                continue
            cand = debug_maps.get(rid, {}).get(assignment.target)
            if cand is None:
                continue
            by_robot[int(rid)] = float(cand.execution_penalty)
            features_by_robot[int(rid)] = {
                key: float(cand.execution_features.get(key, 0.0))
                for key in feature_keys
            }

        feature_means: dict[str, float] = {}
        if features_by_robot:
            for key in feature_keys:
                feature_means[key] = float(mean(v.get(key, 0.0) for v in features_by_robot.values()))

        return {
            "selected_execution_penalty_by_robot": by_robot,
            "selected_execution_penalty_mean": float(mean(by_robot.values())) if by_robot else 0.0,
            "selected_execution_feature_breakdown_by_robot": features_by_robot,
            "selected_execution_feature_means": feature_means,
            "execution_aware_enabled": True,
        }

    def plan(self, planner_input: PlannerInput) -> PlannerOutput:
        robots = planner_input.robot_states
        candidates = planner_input.frontier_candidates
        map_mgr = planner_input.shared_map
        cfg = planner_input.config
        reservations = planner_input.reservation_state

        if not robots:
            return PlannerOutput(planner_name=self.name, assignments={}, joint_score=float("-inf"), debug={"reason": "no_robot"})

        debug_maps: dict[int, dict[tuple[int, int], _CandidateDebug]] = {}

        if len(robots) == 1:
            robot = robots[0]
            scored, debug_map = self._score_candidates(robot, robots, candidates, map_mgr, cfg, reservations)
            debug_maps[robot.robot_id] = debug_map
            if not scored:
                return PlannerOutput(planner_name=self.name, assignments={robot.robot_id: GoalAssignment(robot.robot_id, None, [], float("-inf"), False, {})}, joint_score=float("-inf"), debug={"reason": "no_candidate"})

            target, ev = max(scored.items(), key=lambda kv: kv[1].utility)
            assignment = self._assignment_from_debug(robot.robot_id, target, debug_map[target])
            debug = {
                "candidate_count": len(candidates),
                "u_count": len(scored),
                **self._selected_execution_debug({robot.robot_id: assignment}, debug_maps),
            }
            return PlannerOutput(
                planner_name=self.name,
                assignments={robot.robot_id: assignment},
                joint_score=float(ev.utility),
                score_breakdown={
                    "baseline_utility": float(debug_map[target].baseline_utility),
                    "execution_penalty": float(debug_map[target].execution_penalty),
                    "utility_plus": float(debug_map[target].adjusted_utility),
                },
                predicted_paths={robot.robot_id: list(ev.path)},
                debug=debug,
            )

        r1, r2 = robots[0], robots[1]
        u1, d1 = self._score_candidates(r1, robots, candidates, map_mgr, cfg, reservations)
        u2, d2 = self._score_candidates(r2, robots, candidates, map_mgr, cfg, reservations)
        debug_maps[r1.robot_id] = d1
        debug_maps[r2.robot_id] = d2

        assignments, joint_score, breakdown, debug = solve_joint_cfpa2(r1, r2, u1, u2, cfg)
        selected_assignments: dict[int, GoalAssignment] = {}
        predicted_paths: dict[int, list[tuple[int, int]]] = {}

        for rid, assignment in assignments.items():
            if not assignment.valid or assignment.target is None:
                selected_assignments[rid] = assignment
                continue
            cand = debug_maps.get(rid, {}).get(assignment.target)
            if cand is None:
                selected_assignments[rid] = assignment
                continue
            selected_assignments[rid] = self._assignment_from_debug(rid, assignment.target, cand)
            predicted_paths[rid] = list(cand.path)

        selected_debug = self._selected_execution_debug(selected_assignments, debug_maps)
        baseline_sum = 0.0
        for rid, assignment in selected_assignments.items():
            if assignment.valid and assignment.target is not None:
                cand = debug_maps.get(rid, {}).get(assignment.target)
                if cand is not None:
                    baseline_sum += float(cand.baseline_utility)

        breakdown = dict(breakdown)
        breakdown["baseline_utility_sum"] = float(baseline_sum)
        breakdown["execution_penalty_mean"] = float(selected_debug.get("selected_execution_penalty_mean", 0.0))

        debug.update(selected_debug)
        debug.update(
            {
                "candidate_count": len(candidates),
                "u1_count": len(u1),
                "u2_count": len(u2),
            }
        )

        return PlannerOutput(
            planner_name=self.name,
            assignments=selected_assignments,
            joint_score=float(joint_score),
            score_breakdown=breakdown,
            predicted_paths=predicted_paths,
            debug=debug,
        )
