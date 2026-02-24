from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..core.allocator import TargetAssignment
from ..core.grid_map import Cell, FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from ..core.robot import RobotState
from .colors import FRONTIER_COLOR, FREE_COLOR, OCCUPIED_COLOR, PATH_COLOR, ROBOT_COLORS, TRAJ_COLORS, UNKNOWN_COLOR


def make_grid_image(grid: OccupancyGrid) -> np.ndarray:
    # UNKNOWN -> 0, FREE -> 1, OCCUPIED -> 2
    img = np.zeros_like(grid.grid, dtype=np.int8)
    img[grid.grid == FREE] = 1
    img[grid.grid == OCCUPIED] = 2
    return img


def draw_state(
    ax: plt.Axes,
    grid: OccupancyGrid,
    robots: Sequence[RobotState],
    frontier_cells: Sequence[Cell],
    frontier_reps: Sequence[Cell],
    assignments: Sequence[TargetAssignment],
    step: int,
    explored_ratio: float,
    frontier_cell_count: int,
    frontier_cluster_count: int,
    joint_score: float | None,
    replan_count: int,
    mode: str,
    last_replan_reason: str,
    show_frontier_cells: bool = True,
) -> None:
    cmap = ListedColormap([UNKNOWN_COLOR, FREE_COLOR, OCCUPIED_COLOR])
    ax.imshow(make_grid_image(grid), cmap=cmap, origin="upper", interpolation="nearest")

    if show_frontier_cells and frontier_cells:
        fx = [c[0] for c in frontier_cells]
        fy = [c[1] for c in frontier_cells]
        ax.scatter(fx, fy, c=FRONTIER_COLOR, s=5, marker=".", alpha=0.5)

    if frontier_reps:
        rx = [c[0] for c in frontier_reps]
        ry = [c[1] for c in frontier_reps]
        ax.scatter(rx, ry, c=FRONTIER_COLOR, s=45, marker="x", linewidths=1.5)

    for robot in robots:
        rid = robot.robot_id
        color = ROBOT_COLORS.get(rid, "#FF9800")
        traj_color = TRAJ_COLORS.get(rid, "#FFE082")

        if len(robot.trajectory_history) >= 2:
            tx = [p[0] for p in robot.trajectory_history]
            ty = [p[1] for p in robot.trajectory_history]
            ax.plot(tx, ty, color=traj_color, linewidth=1.0, alpha=0.8)

        if robot.path:
            px = [p[0] for p in robot.path]
            py = [p[1] for p in robot.path]
            ax.plot(px, py, linestyle="--", color=PATH_COLOR.get(rid, color), linewidth=1.2)

        ax.scatter([robot.pose[0]], [robot.pose[1]], c=color, s=70, marker="o", edgecolors="black", linewidths=0.5)
        ax.text(robot.pose[0] + 0.4, robot.pose[1] + 0.4, f"R{rid}", color=color, fontsize=8)

    for assignment in assignments:
        if not assignment.valid or assignment.target is None:
            continue
        color = ROBOT_COLORS.get(assignment.robot_id, "#FF9800")
        ax.scatter([assignment.target[0]], [assignment.target[1]], c=color, s=140, marker="*")

    score_text = f"{joint_score:.2f}" if joint_score is not None and np.isfinite(joint_score) else "-"
    text = (
        f"mode={mode}\n"
        f"step={step}\n"
        f"coverage={explored_ratio * 100:.1f}%\n"
        f"frontier_cells={frontier_cell_count}\n"
        f"frontier_clusters={frontier_cluster_count}\n"
        f"replans={replan_count}\n"
        f"joint_score={score_text}\n"
        f"last_replan={last_replan_reason}"
    )

    ax.text(
        1.01,
        0.99,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    legend_handles = [
        Patch(facecolor=UNKNOWN_COLOR, edgecolor="black", label="Unknown"),
        Patch(facecolor=FREE_COLOR, edgecolor="black", label="Free"),
        Patch(facecolor=OCCUPIED_COLOR, edgecolor="black", label="Occupied"),
        Line2D([0], [0], marker=".", color=FRONTIER_COLOR, linestyle="None", markersize=8, label="Frontier cells"),
        Line2D([0], [0], marker="x", color=FRONTIER_COLOR, linestyle="None", markersize=7, label="Frontier reps"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor=ROBOT_COLORS.get(1, "#D32F2F"),
            linestyle="None",
            markersize=7,
            label="Robot 1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor=ROBOT_COLORS.get(2, "#1976D2"),
            linestyle="None",
            markersize=7,
            label="Robot 2",
        ),
        Line2D([0], [0], color=TRAJ_COLORS.get(1, "#FFCDD2"), linewidth=1.2, label="Trajectory"),
        Line2D([0], [0], color=PATH_COLOR.get(1, "#E57373"), linestyle="--", linewidth=1.2, label="Planned path"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor=ROBOT_COLORS.get(1, "#D32F2F"),
            linestyle="None",
            markersize=9,
            label="Current target",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.54),
        borderaxespad=0.0,
        fontsize=7,
        framealpha=0.85,
        ncol=1,
    )

    ax.set_title("CFPA-2 Demo")
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
