"""Visualization utilities for the RL navigation environment.

Provides unified visualization API that works with any environment
(LightweightEnv, MuJoCoEnvAdapter, future robot envs).

Visualization modes:
    - null/false: no visualization
    - text: save actions.txt and meta.json only
    - pictures: text + PNG frames
    - video: text + PNG frames + MP4 video

Each environment implements render_episode_frame() for its own
rendering backend. The visualizer handles text logs, frame
selection, and video assembly.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════
_FONT_SIZE_OVERLAY = 12
_MAX_LINE_LENGTH = 100
_TEXT_PADDING = 2
_TEXT_Y_START = 5
_TEXT_X_START = 5
_TEXT_LINE_SPACING = 3
_BG_ALPHA = 200
_DETAIL_STEPS = 100
_FRAME_INTERVAL = 50


# ═══════════════════════════════════════════════════
# Environment render protocol
# ═══════════════════════════════════════════════════
@runtime_checkable
class RenderableEnv(Protocol):
    """Protocol for environments that can render scene frames."""

    def render_episode_frame(
        self,
        agent_pose: np.ndarray,
        goal_pose: np.ndarray,
        filepath: str,
        trail_poses: list[np.ndarray] | None = None,
        text: str = "",
        step_num: int = 0,
        distance: float = 0.0,
        result: str = "",
    ) -> None:
        """Render a scene frame to file.

        Each environment implements this with its own backend:
        - LightweightEnv: trimesh solid + x-ray split view
        - MuJoCoEnvAdapter: MuJoCo solid + x-ray split view
        - Future envs: camera feed, etc.

        Args:
            agent_pose: Agent pose [x,y,z,rx,ry,rz] in env's native frame.
            goal_pose: Goal pose [x,y,z,rx,ry,rz] in env's native frame.
            filepath: Output PNG path.
            trail_poses: Previous agent poses for trail visualization.
            text: Action description text overlay.
            step_num: Current step number.
            distance: Distance to goal in mm.
            result: Episode result string.
        """
        ...


# ═══════════════════════════════════════════════════
# Text overlay helper
# ═══════════════════════════════════════════════════
def add_text_overlay(img, text, step_num=0, distance=0.0, result=""):
    """Add text overlay to frame image.

    For adaptive mode: text is already a short label like
    "Step 001 | dist=42.3mm | MoveTangentially | SAC"

    For eval mode: text is interpretation string, extract action name.
    """
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    # Check if text is already a short label (from adaptive mode)
    if text.startswith("Step "):
        header = text
    elif "##### " in text:
        # Eval mode: extract action name from interpretation
        after = text.split("##### ", 1)[1]
        action_name = after.split(";")[0].strip()
        if ";" in after:
            detail = after.split(";", 1)[1].strip()
            action_name += ": " + detail.split(",")[0].strip()
        header = f"Step {step_num} | dist={distance:.1f}mm | {action_name}"
    else:
        # Fallback: use text as-is, truncate if too long
        header = (
            text[:_MAX_LINE_LENGTH]
            if len(text) > _MAX_LINE_LENGTH
            else text
        )

    bbox = draw.textbbox((_TEXT_X_START, _TEXT_Y_START), header, font=font)
    draw.rectangle(
        [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
        fill=(0, 0, 0, 200),
    )
    draw.text((_TEXT_X_START, _TEXT_Y_START), header, fill="white", font=font)

    return img


# ═══════════════════════════════════════════════════
# Episode text log (universal)
# ═══════════════════════════════════════════════════
def save_text_log(
    ep_dir: Path,
    result: str,
    goal_pose: np.ndarray,
    episode_poses: list[np.ndarray],
    episode_actions: list[str],
    extra_info: dict[str, Any] | None = None,
) -> None:
    """Save episode text log and metadata. Universal for all envs.

    Creates:
        - actions.txt: human-readable step-by-step log
        - meta.json: machine-readable metadata
    """
    ep_dir.mkdir(parents=True, exist_ok=True)

    # ═══ actions.txt ═══
    log_path = ep_dir / "actions.txt"
    with log_path.open("w") as f:
        # Header: extra_info first (object, level, mode, etc.)
        if extra_info:
            for key, val in extra_info.items():
                f.write(f"{key}: {val}\n")

        # Episode summary
        f.write(f"Result: {result}\n")
        f.write(f"Steps: {len(episode_actions)}\n")
        f.write(f"Goal: {goal_pose.tolist()}\n")

        if episode_poses:
            start_dist = float(np.linalg.norm(
                goal_pose[:3] - episode_poses[0][:3]
            ))
            end_dist = float(np.linalg.norm(
                goal_pose[:3] - episode_poses[-1][:3]
            ))
            f.write(f"Start distance: {start_dist:.1f}mm\n")
            f.write(f"End distance: {end_dist:.1f}mm\n")

        f.write("\n")
        for i, action in enumerate(episode_actions):
            if i < len(episode_poses) - 1:
                pose = episode_poses[i + 1]
                dist = float(np.linalg.norm(goal_pose[:3] - pose[:3]))
                f.write(f"Step {i+1:03d} (dist={dist:.1f}mm): {action}\n")
            else:
                f.write(f"Step {i+1:03d}: {action}\n")

    # ═══ meta.json ═══
    meta = {
        "result": result,
        "goal_pose": goal_pose.tolist(),
        "num_steps": len(episode_actions),
        "start_pose": episode_poses[0].tolist() if episode_poses else None,
        "end_pose": episode_poses[-1].tolist() if episode_poses else None,
    }
    if extra_info:
        meta["extra"] = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in extra_info.items()
        }
    meta_path = ep_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


# ═══════════════════════════════════════════════════
# Video creation (universal)
# ═══════════════════════════════════════════════════
def create_video_from_frames(
    episode_dir: Path,
    output_path: Path | None = None,
    fps: int = 5,
) -> Path | None:
    """Create MP4 video from saved PNG frames."""
    import glob

    frame_paths = sorted(glob.glob(str(episode_dir / "step_*.png")))
    if not frame_paths:
        return None

    if output_path is None:
        output_path = episode_dir / "episode.mp4"

    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(p) for p in frame_paths]
        imageio.mimwrite(str(output_path), frames, fps=fps, codec='libx264')
        logger.info("Video saved: %s (%d frames)", output_path, len(frames))
        return output_path
    except Exception:
        try:
            import cv2
            first = cv2.imread(frame_paths[0])
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            for p in frame_paths:
                writer.write(cv2.imread(p))
            writer.release()
            return output_path
        except Exception:
            logger.warning("Video creation skipped: %s", episode_dir)
            return None


# ═══════════════════════════════════════════════════
# Frame selection logic
# ═══════════════════════════════════════════════════
def should_save_frame(
    step_idx: int,
    total_steps: int,
    detail_steps: int = _DETAIL_STEPS,
    frame_interval: int = _FRAME_INTERVAL,
) -> bool:
    """Determine if this step should be saved as a frame."""
    if step_idx == 0:
        return True
    if step_idx <= detail_steps:
        return True
    if step_idx >= total_steps - int(frame_interval * 0.5):
        return True
    if step_idx % frame_interval == 0:
        return True
    return False


# ═══════════════════════════════════════════════════
# EpisodeVisualizer — unified API
# ═══════════════════════════════════════════════════
class EpisodeVisualizer:
    """Unified episode visualization for any environment.

    Manages per-level filtering and delegates rendering to
    the environment's render_episode_frame() method.

    Usage:
        viz = EpisodeVisualizer(output_dir, "banana", "eval", "pictures")

        # In episode loop:
        viz.save_episode(
            env=env,          # LightweightEnv or MuJoCoEnvAdapter
            episode=42,
            level=1,
            result="success",
            goal_pose=goal,
            poses=trajectory,
            actions=explanations,
            actions_short=short_labels,
        )

    Visualization modes:
        - "text": actions.txt + meta.json only
        - "pictures": text + PNG frames (env-specific rendering)
        - "video": text + PNG frames + MP4 video
    """

    def __init__(
        self,
        output_dir: Path,
        mesh_name: str = "",
        stage: str = "eval",
        max_per_type_per_level: int = 3,
        num_levels: int = 3,
        timeout_frame_interval: int = _FRAME_INTERVAL,
        visualize_mode: str = "text",
    ):
        self.output_dir = output_dir / f"viz_{stage}_{mesh_name}"
        self.max_per_type = max_per_type_per_level
        self.timeout_frame_interval = timeout_frame_interval
        self.visualize_mode = visualize_mode

        self.counts: dict[str, int] = {}
        for level in range(num_levels):
            for result in ("success", "collision", "timeout"):
                self.counts[f"level_{level}_{result}"] = 0

    def should_save(self, level: int, result: str) -> bool:
        """Check if we should save this episode."""
        key = f"level_{level}_{result}"
        return self.counts.get(key, 0) < self.max_per_type

    def save_episode(
        self,
        env: Any,
        episode: int,
        level: int,
        result: str,
        goal_pose: np.ndarray,
        poses: list[np.ndarray],
        actions: list[str],
        actions_short: list[str] | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """Save episode visualization.

        Works with any environment that implements render_episode_frame().
        Falls back to text-only if env doesn't support rendering.

        Args:
            env: Environment instance (LightweightEnv, MuJoCoEnvAdapter, etc.)
            episode: Episode number.
            level: Curriculum level.
            result: "success", "collision", or "timeout".
            goal_pose: Goal pose in env's native frame.
            poses: List of agent poses in env's native frame.
            actions: Full action description strings (for actions.txt).
            actions_short: Short action labels (for frames/video).
                If None, falls back to actions.
            extra_info: Optional dict with additional metadata.
        """
        if not self.should_save(level, result):
            return

        key = f"level_{level}_{result}"
        episode_id = f"ep_{episode + 1:05d}_L{level}_{result}"
        ep_dir = self.output_dir / episode_id

        # ═══ 1. Always save text log (full actions) ═══
        save_text_log(
            ep_dir=ep_dir,
            result=result,
            goal_pose=goal_pose,
            episode_poses=poses,
            episode_actions=actions,
            extra_info=extra_info,
        )

        # ═══ 2. Save frames if mode requires (short labels for overlay) ═══
        if self.visualize_mode in ("pictures", "video"):
            viz_labels = actions_short if actions_short else actions
            self._save_frames(
                env, ep_dir, goal_pose, poses, viz_labels, result
            )

        # ═══ 3. Save video if mode requires ═══
        if self.visualize_mode == "video":
            create_video_from_frames(ep_dir, fps=5)

        self.counts[key] += 1
        logger.info(
            "Saved episode %s to %s (%s)",
            episode_id, ep_dir, self.visualize_mode,
        )

    def _save_frames(
        self,
        env: Any,
        ep_dir: Path,
        goal_pose: np.ndarray,
        poses: list[np.ndarray],
        actions: list[str],
        result: str,
    ) -> None:
        """Save PNG frames using env's renderer."""
        total_steps = len(poses)
        trail_so_far: list[np.ndarray] = []
        has_renderer = hasattr(env, 'render_episode_frame')

        if not has_renderer:
            logger.warning(
                "Env %s has no render_episode_frame, skipping frames",
                type(env).__name__,
            )
            return

        for i in range(total_steps):
            trail_so_far.append(poses[i])

            if not should_save_frame(
                i, total_steps,
                frame_interval=self.timeout_frame_interval,
            ):
                continue

            action_text = actions[i - 1] if 0 < i <= len(actions) else "INIT"
            distance = float(np.linalg.norm(goal_pose[:3] - poses[i][:3]))

            try:
                filepath = str(ep_dir / f"step_{i:03d}.png")
                env.render_episode_frame(
                    agent_pose=poses[i],
                    goal_pose=goal_pose,
                    filepath=filepath,
                    trail_poses=(
                        trail_so_far[:-1]
                        if len(trail_so_far) > 1
                        else None
                    ),
                    text=action_text,
                    step_num=i,
                    distance=distance,
                    result=result,
                )
            except Exception as e:
                logger.debug("Frame render failed at step %d: %s", i, e)

    def get_stats(self) -> dict[str, int]:
        """Return current save counts."""
        return dict(self.counts)


# ═══════════════════════════════════════════════════
# Legacy compatibility
# ═══════════════════════════════════════════════════
# Keep old functions for backward compatibility
def save_episode_frames(
    env, goal_pose, episode_poses, episode_actions,
    output_dir, episode_id, result="unknown",
    timeout_frame_interval=_FRAME_INTERVAL,
    render_mode="text",
):
    """Legacy wrapper. Use EpisodeVisualizer.save_episode() instead."""
    ep_dir = Path(output_dir) / episode_id

    save_text_log(
        ep_dir=ep_dir,
        result=result,
        goal_pose=goal_pose,
        episode_poses=episode_poses,
        episode_actions=episode_actions,
    )

    if render_mode in ("pictures", "video"):
        if hasattr(env, 'render_episode_frame'):
            total = len(episode_poses)
            trail = []
            for i in range(total):
                trail.append(episode_poses[i])
                if should_save_frame(
                    i, total, frame_interval=timeout_frame_interval
                ):
                    action_text = (
                        episode_actions[i - 1]
                        if 0 < i <= len(episode_actions)
                        else "INIT"
                    )
                    distance = float(np.linalg.norm(
                        goal_pose[:3] - episode_poses[i][:3]
                    ))
                    try:
                        env.render_episode_frame(
                            agent_pose=episode_poses[i],
                            goal_pose=goal_pose,
                            filepath=str(ep_dir / f"step_{i:03d}.png"),
                            trail_poses=(
                                trail[:-1] if len(trail) > 1 else None
                            ),
                            text=action_text,
                            step_num=i,
                            distance=distance,
                            result=result,
                        )
                    except Exception:
                        pass
        else:
            # Fallback to old trimesh rendering
            _render_frames_trimesh_legacy(
                env, goal_pose, episode_poses, episode_actions,
                ep_dir, result, timeout_frame_interval,
            )

    if render_mode == "video":
        create_video_from_frames(ep_dir, fps=5)


def _render_frames_trimesh_legacy(env, goal_pose, poses, actions,
                                   ep_dir, result, interval):
    """Legacy trimesh rendering for envs without render_episode_frame."""
    # Import here to avoid circular deps
    from .visualize_env import render_frame_to_file as _old_render
    total = len(poses)
    trail = []
    for i in range(total):
        trail.append(poses[i])
        if should_save_frame(i, total, frame_interval=interval):
            text = actions[i-1] if 0 < i <= len(actions) else "INIT"
            dist = float(np.linalg.norm(goal_pose[:3] - poses[i][:3]))
            _old_render(
                env=env, agent_pose=poses[i], goal_pose=goal_pose,
                filepath=ep_dir / f"step_{i:03d}.png",
                text=text, step_num=i, distance=dist, result=result,
                trail_poses=trail[:-1] if len(trail) > 1 else None,
            )


# Keep old imports working
def visualize_agent_goal(env, agent_pose, goal_pose):
    """Legacy interactive visualization."""
    import trimesh
    from scipy.spatial.transform import Rotation
    scene = trimesh.Scene()
    mesh_copy = env.mesh.copy()
    mesh_copy.visual.face_colors = [200, 200, 200, 255]
    scene.add_geometry(mesh_copy)
    agent_sphere = trimesh.primitives.Sphere(radius=1.0, center=agent_pose[:3])
    agent_sphere.visual.face_colors = [0, 50, 255, 255]
    scene.add_geometry(agent_sphere)
    goal_sphere = trimesh.primitives.Sphere(radius=1.4, center=goal_pose[:3])
    goal_sphere.visual.face_colors = [0, 255, 0, 255]
    scene.add_geometry(goal_sphere)
    scene.show(smooth=False)


def create_video_from_episode(episode_dir, output_path=None, fps=5):
    """Legacy wrapper."""
    return create_video_from_frames(Path(episode_dir), output_path, fps)


def create_all_videos(output_dir, fps=5):
    """Create videos for all episodes in directory."""
    for ep_dir in sorted(Path(output_dir).iterdir()):
        if ep_dir.is_dir():
            create_video_from_frames(ep_dir, fps=fps)
