# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Visualization utilities for the RL navigation environment.

Provides functions for rendering agent trajectories, saving episode frames
as PNG images with action annotations, and interactive scene visualization.
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from pathlib import Path

    from tbp.hybrid_rl.lightweight_env import LightweightEnv

logger = logging.getLogger(__name__)

_MIN_RENDER_BYTES = 100
_FONT_SIZE_OVERLAY = 12
_MAX_LINE_LENGTH = 100
_AGENT_SPHERE_RADIUS = 0.5
_GOAL_SPHERE_RADIUS = 0.7
_TRAIL_SPHERE_RADIUS = 0.3
_GAZE_ARROW_LENGTH = 3
_NORM_EPSILON = 1e-12
_CROSS_PRODUCT_THRESHOLD = 1e-8
_TEXT_PADDING = 2
_TEXT_Y_START = 5
_TEXT_X_START = 5
_TEXT_LINE_SPACING = 3
_BG_ALPHA = 200
_FOV = 45.0
_CAMERA_DISTANCE_MULTIPLIER = 1.5
# ва числа: `2.0 → 1.5` (камера ближе) и `60.0 → 45.0` (уже угол обзора = зум). 
# Если будет слишком близко — подкрутите обратно, например `1.7` и `50.0`. Если хочется ещё ближе — `1.2` и `40.0`.


def _build_scene(
    env: LightweightEnv,
    agent_pose: np.ndarray,
    goal_pose: np.ndarray,
    trail_poses: list[np.ndarray] | None = None,
) -> trimesh.Scene:
    """Build a trimesh scene for rendering.

    Args:
        env: LightweightEnv instance.
        agent_pose: Agent pose [x, y, z, roll, pitch, yaw].
        goal_pose: Goal pose [x, y, z, roll, pitch, yaw].
        trail_poses: Optional list of previous agent poses for trail.

    Returns:
        Configured trimesh Scene with mesh, agent, goal, and trail.
    """
    scene = trimesh.Scene()

    axis = trimesh.creation.axis(origin_size=2.0, axis_length=10)
    scene.add_geometry(axis, geom_name="world_axis")
    scene.add_geometry(env.mesh, geom_name="mesh")

    agent_pos = agent_pose[:3]
    agent_rot = agent_pose[3:]

    agent_sphere = trimesh.primitives.Sphere(
        radius=_AGENT_SPHERE_RADIUS, center=agent_pos
    )
    agent_sphere.visual.face_colors = [0, 0, 255, 255]
    scene.add_geometry(agent_sphere, geom_name="agent")

    rot = Rotation.from_euler("xyz", agent_rot, degrees=True)
    forward_dir = rot.apply([0, 0, -1])
    line_vertices = np.array([
        agent_pos, agent_pos + forward_dir * _GAZE_ARROW_LENGTH
    ])
    arrow = trimesh.load_path(line_vertices, colors=[[255, 0, 0, 255]])
    scene.add_geometry(arrow, geom_name="gaze")

    goal_sphere = trimesh.primitives.Sphere(
        radius=_GOAL_SPHERE_RADIUS, center=goal_pose[:3]
    )
    goal_sphere.visual.face_colors = [0, 255, 0, 255]
    scene.add_geometry(goal_sphere, geom_name="goal")

    if trail_poses:
        for i, pose in enumerate(trail_poses):
            trail_sphere = trimesh.primitives.Sphere(
                radius=_TRAIL_SPHERE_RADIUS, center=pose[:3]
            )
            trail_sphere.visual.face_colors = [255, 255, 0, 128]
            scene.add_geometry(trail_sphere, geom_name=f"trail_{i}")

    return scene


def _compute_camera_transform(
    scene: trimesh.Scene,
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    fixed_offset_multiplier: float = _CAMERA_DISTANCE_MULTIPLIER,
) -> np.ndarray:
    """Вычисляет ракурс камеры, центрируясь на детали, но разворачиваясь к агенту."""
    center = scene.bounds.mean(axis=0)
    
    # ИСПРАВЛЕНО: вычитаем минимальные границы из максимальных
    mesh_size = float(np.linalg.norm(scene.bounds[1] - scene.bounds[0]))
    camera_distance = mesh_size * fixed_offset_multiplier

    to_agent = agent_pos - center
    to_agent_norm = np.linalg.norm(to_agent)
    
    if to_agent_norm > 1e-5:
        dir_to_agent = to_agent / to_agent_norm
    else:
        dir_to_agent = np.array([1.0, 0.0, 0.0])

    camera_pos = center + dir_to_agent * (camera_distance * 0.8)
    camera_pos[2] += camera_distance * 0.5  # ИСПРАВЛЕНО: прибавляем строго к оси Z (высота)

    forward = center - camera_pos
    forward /= np.linalg.norm(forward) + 1e-6

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    
    if np.linalg.norm(right) < 1e-3:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        
    right /= np.linalg.norm(right) + 1e-6
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-6

    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = -forward
    transform[:3, 3] = camera_pos

    return transform


def _add_text_to_image(
    png_bytes: bytes,
    text: str,
    step_num: int = 0,
    distance: float = 0.0,
    result: str = "",
) -> bytes:
    """Add text overlay on top of a PNG image.

    Args:
        png_bytes: Raw PNG image bytes.
        text: Action description text.
        step_num: Current step number.
        distance: Distance to goal in mm.
        result: Episode result ("success", "collision", "timeout").

    Returns:
        PNG image bytes with text overlay.
    """
    img = Image.open(io.BytesIO(png_bytes))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            _FONT_SIZE_OVERLAY,
        )
    except OSError:
        font = ImageFont.load_default()

    header = f"Step {step_num} | dist={distance:.1f}mm | {result}"
    lines = [header]
    lines.extend(
        text[i:i + _MAX_LINE_LENGTH]
        for i in range(0, len(text), _MAX_LINE_LENGTH)
    )

    y = _TEXT_Y_START
    for line in lines:
        text_bbox = draw.textbbox((_TEXT_X_START, y), line, font=font)
        draw.rectangle(
            [
                text_bbox[0] - _TEXT_PADDING,
                text_bbox[1] - 1,
                text_bbox[2] + _TEXT_PADDING,
                text_bbox[3] + 1,
            ],
            fill=(0, 0, 0, _BG_ALPHA),
        )
        draw.text((_TEXT_X_START, y), line, fill="white", font=font)
        y += text_bbox[3] - text_bbox[1] + _TEXT_LINE_SPACING

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_frame_to_file(
    env: LightweightEnv,
    agent_pose: np.ndarray,
    goal_pose: np.ndarray,
    filepath: Path,
    text: str = "",
    step_num: int = 0,
    distance: float = 0.0,
    result: str = "",
    trail_poses: list[np.ndarray] | None = None,
    resolution: tuple[int, int] = (1024, 768),
    fov: float = _FOV,
) -> None:
    """Render a frame to a PNG file with text annotation.

    Args:
        env: LightweightEnv instance.
        agent_pose: Agent pose [x, y, z, roll, pitch, yaw].
        goal_pose: Goal pose [x, y, z, roll, pitch, yaw].
        filepath: Output PNG file path.
        text: Action description text to overlay.
        step_num: Current step number.
        distance: Distance to goal in mm.
        result: Episode result string.
        trail_poses: Optional list of previous poses for trail visualization.
        resolution: Image resolution (width, height).

    Raises:
        RuntimeError: If the rendered image is empty or too small.
    """
    scene = _build_scene(env, agent_pose, goal_pose, trail_poses)

    camera_transform = _compute_camera_transform(
        scene=scene,
        agent_pos=agent_pose[:3],
        goal_pos=goal_pose[:3]
    )
    scene.camera_transform = camera_transform

    scene.camera.fov = (fov, fov)
    scene.camera.z_near = 1.0
    scene.camera.z_far = 100000.0
    scene.camera.resolution = resolution

    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        png_data = scene.save_image(resolution=resolution, visible=False, smooth=False)

        if png_data is None or len(png_data) < _MIN_RENDER_BYTES:
            msg = "Empty render output"
            raise RuntimeError(msg)

        if text:
            png_data = _add_text_to_image(
                png_data, text, step_num, distance, result
            )
        with filepath.open("wb") as f:
            f.write(png_data)
    except (RuntimeError, OSError):
        txt_path = filepath.with_suffix(".txt")
        with txt_path.open("w") as f:
            f.write(f"Step: {step_num}, Distance: {distance:.1f}mm\n")
            f.write(f"Agent: {agent_pose.tolist()}\n")
            f.write(f"Goal: {goal_pose.tolist()}\n")
            f.write(f"Action: {text}\n")
        logger.warning("Render failed for %s", filepath, exc_info=True)


def save_episode_frames(
    env: LightweightEnv,
    goal_pose: np.ndarray,
    episode_poses: list[np.ndarray],
    episode_actions: list[str],
    output_dir: Path,
    episode_id: str,
    result: str = "unknown",
    timeout_frame_interval: int = 50,
) -> None:
    """Save frames of an episode with action annotations.

    For timeout episodes, saves only every Nth frame and the last frame
    to avoid excessive file generation. For success and collision episodes,
    saves all frames.

    Args:
        env: LightweightEnv instance.
        goal_pose: Target pose [6D].
        episode_poses: List of agent poses at each step.
        episode_actions: List of action descriptions (action_explanations).
        output_dir: Root directory for saving.
        episode_id: Unique episode identifier.
        result: Episode result ("success", "collision", "timeout").
        timeout_frame_interval: For timeout episodes, save every Nth frame.
    """
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Always save full action log
    log_path = ep_dir / "actions.txt"
    with log_path.open("w") as f:
        f.write(f"Result: {result}\n")
        f.write(f"Goal: {goal_pose.tolist()}\n")
        f.write(f"Steps: {len(episode_actions)}\n")
        if episode_poses:
            f.write(f"Start: {episode_poses[0].tolist()}\n")
            f.write(f"End: {episode_poses[-1].tolist()}\n")
            start_dist = float(
                np.linalg.norm(goal_pose[:3] - episode_poses[0][:3])
            )
            end_dist = float(
                np.linalg.norm(goal_pose[:3] - episode_poses[-1][:3])
            )
            f.write(f"Start distance: {start_dist:.1f}mm\n")
            f.write(f"End distance: {end_dist:.1f}mm\n")
        f.write("\n")
        for i, action in enumerate(episode_actions):
            if i < len(episode_poses):
                dist = float(
                    np.linalg.norm(goal_pose[:3] - episode_poses[i][:3])
                )
            else:
                dist = 0.0
            f.write(f"Step {i+1:03d} (dist={dist:.1f}mm): {action}\n")

    meta = {
        "episode_id": episode_id,
        "result": result,
        "goal_pose": goal_pose.tolist(),
        "num_steps": len(episode_actions),
        "start_pose": (
            episode_poses[0].tolist() if episode_poses else None
        ),
        "end_pose": (
            episode_poses[-1].tolist() if episode_poses else None
        ),
    }
    meta_path = ep_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    # Determine which frames to save
    is_timeout = result == "timeout"
    last_step = len(episode_poses) - 1

    def should_save_frame(step_idx: int) -> bool:
        if not is_timeout:
            return True
        if step_idx == 0:
            return True
        if step_idx == last_step:
            return True
        if step_idx % timeout_frame_interval == 0:
            return True
        return False

    trail_so_far: list[np.ndarray] = []
    saved_count = 0

    if episode_poses:
        distance = float(
            np.linalg.norm(goal_pose[:3] - episode_poses[0][:3])
        )
        if should_save_frame(0):
            render_frame_to_file(
                env=env,
                agent_pose=episode_poses[0],
                goal_pose=goal_pose,
                filepath=ep_dir / "step_000.png",
                text="RESET (initial position)",
                step_num=0,
                distance=distance,
                result=result,
            )
            saved_count += 1
        trail_so_far.append(episode_poses[0])

    for i in range(1, len(episode_poses)):
        trail_so_far.append(episode_poses[i])

        if should_save_frame(i):
            action_text = (
                episode_actions[i - 1]
                if (i - 1) < len(episode_actions)
                else "unknown"
            )
            distance = float(
                np.linalg.norm(goal_pose[:3] - episode_poses[i][:3])
            )
            render_frame_to_file(
                env=env,
                agent_pose=episode_poses[i],
                goal_pose=goal_pose,
                filepath=ep_dir / f"step_{i:03d}.png",
                text=action_text,
                step_num=i,
                distance=distance,
                result=result,
                trail_poses=trail_so_far[:-1],
            )
            saved_count += 1

    logger.info(
        "Saved %d/%d frames to %s (%s)",
        saved_count,
        len(episode_poses),
        ep_dir,
        result,
    )
    # save video
    create_video_from_episode(ep_dir, fps=5)

def visualize_scene(
    env: LightweightEnv,
    goal_pose: np.ndarray,
) -> None:
    """Show interactive visualization of the scene.

    Args:
        env: LightweightEnv instance.
        goal_pose: Goal pose [x, y, z, roll, pitch, yaw].
    """
    scene = _build_scene(env, env.get_pose(), goal_pose)
    scene.show(smooth=False)


def visualize_agent_goal(
    env: LightweightEnv,
    agent_pose: np.ndarray,
    goal_pose: np.ndarray,
) -> None:
    """Show interactive visualization of agent and goal.

    Args:
        env: LightweightEnv instance.
        agent_pose: Agent pose [x, y, z, roll, pitch, yaw].
        goal_pose: Goal pose [x, y, z, roll, pitch, yaw].
    """
    scene = _build_scene(env, agent_pose, goal_pose)
    scene.show(smooth=False)

def create_video_from_episode(
    episode_dir: Path,
    output_path: Path | None = None,
    fps: int = 5,
) -> Path | None:
    """Создаёт MP4 видео из сохранённых PNG кадров эпизода.

    Args:
        episode_dir: Папка с step_XXX.png файлами (например output/ep_00001_L0_success/).
        output_path: Путь для сохранения видео. По умолчанию episode_dir/episode.mp4.
        fps: Кадров в секунду.

    Returns:
        Path к видео или None при ошибке.
    """
    import glob

    frames_pattern = str(episode_dir / "step_*.png")
    frame_paths = sorted(glob.glob(frames_pattern))

    if not frame_paths:
        logger.warning("No frames found in %s", episode_dir)
        return None

    if output_path is None:
        output_path = episode_dir / "episode.mp4"

    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(p) for p in frame_paths]
        imageio.mimwrite(str(output_path), frames, fps=fps, codec='libx264')
        logger.info("Video saved: %s (%d frames, %d fps)", output_path, len(frames), fps)
        return output_path
    except ImportError:
        # Fallback на OpenCV
        import cv2
        first = cv2.imread(frame_paths[0])
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for p in frame_paths:
            writer.write(cv2.imread(p))
        writer.release()
        logger.info("Video saved (cv2): %s (%d frames)", output_path, len(frame_paths))
        return output_path


def create_all_videos(output_dir: Path, fps: int = 5) -> None:
    """Создаёт видео для всех эпизодов в директории визуализаций.

    Args:
        output_dir: Корневая папка визуализаций (visualizations_stage_mesh/).
    """
    episode_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
    for ep_dir in episode_dirs:
        create_video_from_episode(ep_dir, fps=fps)

class EpisodeVisualizer:
    """Manages episode visualization with per-level filtering.
    
    Saves limited number of episodes per result type per curriculum level
    to avoid excessive file generation while ensuring coverage.
    
    Usage:
        visualizer = EpisodeVisualizer(
            output_dir=Path("results"),
            mesh_name="cube",
            stage="sac_train",
        )
        # In episode loop:
        visualizer.save_episode(
            env, episode, level, "success",
            goal_pose, poses, actions,
        )
    """

    def __init__(
        self,
        output_dir: Path,
        mesh_name: str = "",
        stage: str = "train",
        max_per_type_per_level: int = 3,
        num_levels: int = 3,
        timeout_frame_interval: int = 100,
    ):
        self.output_dir = output_dir / f"visualizations_{stage}_{mesh_name}"
        self.max_per_type = max_per_type_per_level
        self.timeout_frame_interval = timeout_frame_interval

        self.counts: dict[str, int] = {}
        for level in range(num_levels):
            for result in ("success", "collision", "timeout"):
                self.counts[f"level_{level}_{result}"] = 0

    def should_save(self, level: int, result: str) -> bool:
        """Check if we should save this episode.

        Args:
            level: Curriculum level index.
            result: Episode result ("success", "collision", "timeout").

        Returns:
            True if under the limit for this level+result combination.
        """
        key = f"level_{level}_{result}"
        return self.counts.get(key, 0) < self.max_per_type

    def save_episode(
        self,
        env: "LightweightEnv",
        episode: int,
        level: int,
        result: str,
        goal_pose: np.ndarray,
        poses: list[np.ndarray],
        actions: list[str],
    ) -> None:
        """Save episode visualization if under limit.

        Args:
            env: Environment instance for rendering.
            episode: Episode number.
            level: Curriculum level index.
            result: Episode result ("success", "collision", "timeout").
            goal_pose: Target pose array.
            poses: List of agent poses during episode.
            actions: List of action description strings.
        """
        if not self.should_save(level, result):
            return

        key = f"level_{level}_{result}"
        episode_id = f"ep_{episode + 1:05d}_L{level}_{result}"

        save_episode_frames(
            env=env,
            goal_pose=goal_pose,
            episode_poses=poses,
            episode_actions=actions,
            output_dir=self.output_dir,
            episode_id=episode_id,
            result=result,
            timeout_frame_interval=self.timeout_frame_interval,
        )
        self.counts[key] += 1

    def get_stats(self) -> dict[str, int]:
        """Return current save counts."""
        return dict(self.counts)
