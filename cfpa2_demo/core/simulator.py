from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..maps.generators import generate_ground_truth
from ..viz.animator import Animator
from .allocator import (
    TargetAssignment,
    assign_dual_greedy,
    assign_dual_joint_cfpa2,
    assign_single_robot,
)
from .frontier import build_frontier_clusters
from .grid_map import Cell, OccupancyGrid
from .metrics import SimulationMetrics
from .replanning import apply_hysteresis, should_replan, tick_reservations, update_reservations
from .robot import RobotState


@dataclass
class SimulationResult:
    metrics: SimulationMetrics
    coverage_curve: list[float]
    animation_path: str | None
    video_path: str | None = None


def _coerce_start(start: Sequence[int]) -> Cell:
    return int(start[0]), int(start[1])


def _build_robots(cfg: dict, grid: OccupancyGrid, requested_robots: int | None = None) -> list[RobotState]:
    robots_cfg = cfg["robots"]
    num_robots = int(robots_cfg["num_robots"])
    if requested_robots is not None:
        num_robots = min(num_robots, int(requested_robots))
    starts = [_coerce_start(s) for s in robots_cfg["start_positions"]]

    if len(starts) < num_robots:
        raise ValueError("Not enough start_positions for num_robots")

    starts = starts[:num_robots]
    grid.ensure_starts_free(starts)

    robots: list[RobotState] = []
    for i, s in enumerate(starts, start=1):
        if not grid.in_bounds(s):
            raise ValueError(f"Robot start out of bounds: {s}")
        robots.append(RobotState(robot_id=i, pose=s))
    return robots


def _initial_observe(
    grid: OccupancyGrid,
    robots: Sequence[RobotState],
    sensor_range: int,
    use_line_of_sight: bool,
) -> None:
    for r in robots:
        grid.observe_from(r.pose, sensor_range, use_line_of_sight=use_line_of_sight)


def _valid_assignment(existing: TargetAssignment, frontier_set: set[Cell]) -> bool:
    return existing.valid and existing.target is not None and existing.target in frontier_set


def run_simulation(
    cfg: dict,
    mode: str,
    seed: int | None = None,
    enable_viz: bool | None = None,
    animation_filename: str | None = None,
) -> SimulationResult:
    env_cfg = cfg["environment"]
    if seed is not None:
        env_cfg = dict(env_cfg)
        env_cfg["random_seed"] = int(seed)

    truth = generate_ground_truth(
        map_type=str(env_cfg["map_type"]),
        width=int(env_cfg["map_width"]),
        height=int(env_cfg["map_height"]),
        obstacle_density=float(env_cfg.get("obstacle_density", 0.1)),
        seed=int(env_cfg["random_seed"]),
    )

    grid = OccupancyGrid(truth)
    requested_robots = 1 if mode == "single" else 2
    robots = _build_robots(cfg, grid, requested_robots=requested_robots)

    sensor_range = int(cfg["robots"]["sensor_range"])
    use_line_of_sight = bool(cfg["robots"].get("use_line_of_sight", True))
    frontier_cfg = cfg["frontier"]
    neighborhood = int(frontier_cfg.get("neighborhood", 8))
    target_frontier_count_max_raw = frontier_cfg.get("target_frontier_count_max")
    target_frontier_count_max = (
        int(target_frontier_count_max_raw) if target_frontier_count_max_raw is not None else None
    )

    _initial_observe(grid, robots, sensor_range, use_line_of_sight)

    viz_cfg = dict(cfg["visualization"])
    if enable_viz is not None:
        viz_cfg["enable_live_plot"] = bool(enable_viz)
    animator = Animator(viz_cfg)

    metrics = SimulationMetrics(
        mode=mode,
        map_type=str(env_cfg["map_type"]),
        seed=int(env_cfg["random_seed"]),
    )

    max_steps = int(cfg["termination"]["max_steps"])
    coverage_threshold = float(cfg["termination"]["coverage_threshold"])

    assignments = [
        TargetAssignment(robot_id=r.robot_id, target=None, path=[], utility=float("-inf"), valid=False)
        for r in robots
    ]
    reservation_table: dict[Cell, dict[str, int]] = {}

    old_joint_score: float | None = None
    current_joint_score: float | None = None
    prev_frontier_count = -1
    step = 0
    success = False
    reason = "max_steps"
    last_replan_reason = "init"

    while step < max_steps:
        if mode == "dual_joint":
            tick_reservations(reservation_table)

        # Observation update from both robots.
        for r in robots:
            grid.observe_from(r.pose, sensor_range, use_line_of_sight=use_line_of_sight)

        frontier_cells, clusters = build_frontier_clusters(
            grid,
            neighborhood=neighborhood,
            method=str(frontier_cfg.get("cluster_method", "bfs")),
            min_cluster_size=int(frontier_cfg.get("min_cluster_size", 1)),
            target_frontier_count_min=int(frontier_cfg.get("target_frontier_count_min", 0)),
            target_frontier_count_max=target_frontier_count_max,
            representative_min_distance=float(frontier_cfg.get("representative_min_distance", 0.0)),
        )
        candidates = [c.representative for c in clusters]

        coverage = grid.explored_free_ratio()
        metrics.log_coverage(coverage)
        metrics.log_frontier_counts(len(frontier_cells), len(candidates))

        if coverage >= coverage_threshold:
            success = True
            reason = "coverage_reached"
            break

        if not candidates:
            success = False
            reason = "no_frontier"
            break

        frontier_set = set(candidates)
        do_replan, replan_reason = should_replan(
            grid=grid,
            robots=robots,
            frontier_reps=frontier_set,
            step=step,
            prev_frontier_count=prev_frontier_count,
            current_frontier_count=len(candidates),
            cfg=cfg,
        )
        if step == 0:
            do_replan = True
            replan_reason = "initial"

        if do_replan:
            new_assignments = [
                TargetAssignment(robot_id=r.robot_id, target=None, path=[], utility=float("-inf"), valid=False)
                for r in robots
            ]
            new_joint_score = float("-inf")

            if mode == "single":
                single = assign_single_robot(
                    robot=robots[0],
                    candidates=candidates,
                    grid=grid,
                    cfg=cfg,
                    reservation_table=None,
                    neighborhood=neighborhood,
                )
                new_assignments[0] = single
                new_joint_score = single.utility if single.valid else float("-inf")

            elif mode == "dual_greedy":
                if len(robots) < 2:
                    single = assign_single_robot(robots[0], candidates, grid, cfg, None, neighborhood)
                    new_assignments[0] = single
                    new_joint_score = single.utility if single.valid else float("-inf")
                else:
                    a1, a2, dbg = assign_dual_greedy(
                        robot1=robots[0],
                        robot2=robots[1],
                        candidates=candidates,
                        grid=grid,
                        cfg=cfg,
                        reservation_table=None,
                        neighborhood=neighborhood,
                    )
                    new_assignments[0] = a1
                    new_assignments[1] = a2
                    new_joint_score = (a1.utility if a1.valid else 0.0) + (a2.utility if a2.valid else 0.0)
                    if bool(dbg.get("conflict", False)):
                        metrics.target_conflict_count += 1

            elif mode == "dual_joint":
                if len(robots) < 2:
                    single = assign_single_robot(robots[0], candidates, grid, cfg, None, neighborhood)
                    new_assignments[0] = single
                    new_joint_score = single.utility if single.valid else float("-inf")
                else:
                    a1, a2, score, _dbg = assign_dual_joint_cfpa2(
                        robot1=robots[0],
                        robot2=robots[1],
                        candidates=candidates,
                        grid=grid,
                        cfg=cfg,
                        reservation_table=reservation_table,
                        neighborhood=neighborhood,
                    )
                    new_assignments[0] = a1
                    new_assignments[1] = a2
                    new_joint_score = score
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            critical_reason = (
                "target_invalidated" in replan_reason
                or "target_reached" in replan_reason
                or "path_empty" in replan_reason
                or "stuck" in replan_reason
            )

            if mode == "dual_joint" and (not critical_reason) and any(a.valid for a in assignments):
                selected, kept_old = apply_hysteresis(
                    old_assignments=assignments,
                    new_assignments=new_assignments,
                    old_score=old_joint_score,
                    new_score=new_joint_score,
                    cfg=cfg,
                )
                if kept_old:
                    # Keep previous plan to avoid oscillation.
                    selected = assignments
                    new_joint_score = old_joint_score if old_joint_score is not None else new_joint_score
                assignments = list(selected)
            else:
                assignments = list(new_assignments)

            for robot, assignment in zip(robots, assignments):
                robot.set_plan(assignment.target, assignment.path)

            if mode == "dual_joint":
                update_reservations(
                    reservation_table=reservation_table,
                    assignments=assignments,
                    ttl=int(cfg["allocator"].get("reservation_ttl", 20)),
                )

            old_joint_score = new_joint_score
            current_joint_score = new_joint_score
            metrics.log_replan(replan_reason)
            last_replan_reason = replan_reason
            if "target_invalidated" in replan_reason:
                metrics.target_invalidation_events += 1

        # Move robots one cell along path.
        for robot in robots:
            robot.move_one_step(grid)

        # Immediate post-motion sensing keeps frontier updates responsive.
        for r in robots:
            grid.observe_from(r.pose, sensor_range, use_line_of_sight=use_line_of_sight)

        animator.update(
            step=step,
            grid=grid,
            robots=robots,
            frontier_cells=frontier_cells,
            frontier_reps=candidates,
            assignments=assignments,
            explored_ratio=coverage,
            frontier_cell_count=len(frontier_cells),
            frontier_cluster_count=len(candidates),
            joint_score=current_joint_score,
            replan_count=metrics.replan_count,
            mode=mode,
            last_replan_reason=last_replan_reason,
        )

        prev_frontier_count = len(candidates)
        step += 1

    metrics.finalize(robots=robots, steps=step, success=success, reason=reason)

    output_base = Path(cfg.get("outputs", {}).get("base_dir", "outputs"))
    stem = f"{mode}_{env_cfg['map_type']}_seed{env_cfg['random_seed']}"
    if animation_filename is None:
        animation_filename = f"{stem}.gif"
    video_filename = f"{stem}.{str(viz_cfg.get('video_format', 'mp4')).lower()}"

    gif_output = output_base / "animations" / animation_filename if viz_cfg.get("save_animation", False) else None
    video_output = output_base / "animations" / video_filename if viz_cfg.get("save_video", False) else None

    animation_path, video_path = animator.finalize(gif_output_path=gif_output, video_output_path=video_output)

    return SimulationResult(
        metrics=metrics,
        coverage_curve=metrics.coverage_curve,
        animation_path=animation_path,
        video_path=video_path,
    )
