from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from ..core.allocator import TargetAssignment
from ..core.grid_map import Cell, OccupancyGrid
from ..core.robot import RobotState
from .plotter import draw_state


class Animator:
    def __init__(self, viz_cfg: dict):
        self.enable_live = bool(viz_cfg.get("enable_live_plot", True))
        self.save_animation = bool(viz_cfg.get("save_animation", False))
        self.save_video = bool(viz_cfg.get("save_video", False))
        self.video_format = str(viz_cfg.get("video_format", "mp4")).lower()
        self.fps = int(viz_cfg.get("animation_fps", 10))
        self.plot_every_n_steps = max(1, int(viz_cfg.get("plot_every_n_steps", 1)))
        self.show_frontier_cells = bool(viz_cfg.get("show_frontier_cells", True))

        self.fig = None
        self.ax = None
        self.frames: list[np.ndarray] = []

        if self.enable_live or self.save_animation or self.save_video:
            # Keep frame dimensions divisible by 16 to avoid ffmpeg auto-resizing warnings.
            self.fig, self.ax = plt.subplots(figsize=(11.04, 7.04))
            if self.enable_live:
                plt.ion()

    def should_draw(self, step: int) -> bool:
        return (self.enable_live or self.save_animation or self.save_video) and (step % self.plot_every_n_steps == 0)

    def update(
        self,
        step: int,
        grid: OccupancyGrid,
        robots: Sequence[RobotState],
        frontier_cells: Sequence[Cell],
        frontier_reps: Sequence[Cell],
        assignments: Sequence[TargetAssignment],
        explored_ratio: float,
        frontier_cell_count: int,
        frontier_cluster_count: int,
        joint_score: float | None,
        replan_count: int,
        mode: str,
        last_replan_reason: str,
    ) -> None:
        if not self.should_draw(step):
            return
        if self.ax is None:
            return

        self.ax.clear()
        draw_state(
            ax=self.ax,
            grid=grid,
            robots=robots,
            frontier_cells=frontier_cells,
            frontier_reps=frontier_reps,
            assignments=assignments,
            step=step,
            explored_ratio=explored_ratio,
            frontier_cell_count=frontier_cell_count,
            frontier_cluster_count=frontier_cluster_count,
            joint_score=joint_score,
            replan_count=replan_count,
            mode=mode,
            last_replan_reason=last_replan_reason,
            show_frontier_cells=self.show_frontier_cells,
        )

        self.fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
        self.fig.canvas.draw()

        if self.save_animation or self.save_video:
            frame = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
            self.frames.append(frame)

        if self.enable_live:
            plt.pause(0.001)

    def save_gif(self, output_path: str | Path) -> str | None:
        if not self.frames:
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_path, self.frames, fps=self.fps)
        return str(output_path)

    def save_live_video(self, output_path: str | Path) -> str | None:
        if not self.frames:
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()
        if ext not in (".mp4", ".mov", ".mkv", ".avi"):
            output_path = output_path.with_suffix(".mp4")

        try:
            with imageio.get_writer(output_path, fps=self.fps) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
            return str(output_path)
        except Exception:
            # Fallback when ffmpeg backend is unavailable.
            fallback = output_path.with_suffix(".gif")
            imageio.mimsave(fallback, self.frames, fps=self.fps)
            return str(fallback)

    def finalize(
        self,
        gif_output_path: str | Path | None = None,
        video_output_path: str | Path | None = None,
    ) -> tuple[str | None, str | None]:
        gif_path: str | None = None
        video_path: str | None = None
        if self.save_animation and gif_output_path is not None:
            gif_path = self.save_gif(gif_output_path)
        if self.save_video and video_output_path is not None:
            video_path = self.save_live_video(video_output_path)

        if self.enable_live and self.fig is not None:
            plt.ioff()
        if self.fig is not None:
            plt.close(self.fig)

        return gif_path, video_path
