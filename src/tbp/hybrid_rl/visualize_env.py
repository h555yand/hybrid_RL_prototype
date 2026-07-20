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
_CAMERA_DISTANCE_MULTIPLIER = 1.8
_NORM_EPSILON = 1e-12
_CROSS_PRODUCT_THRESHOLD = 1e-8
_TEXT_PADDING = 2
_TEXT_Y_START = 5
_TEXT_X_START = 5
_TEXT_LINE_SPACING = 3
_BG_ALPHA = 200


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
    fixed_offset_multiplier: float = 2.0,
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
        goal_pos=goal_pose[:3],
        fixed_offset_multiplier=2.0  # Уменьшите до 1.5, если хотите покрупнее
    )
    scene.camera_transform = camera_transform

    # ВАЖНО ДЛЯ OFF-LINE РЕНДЕРА:
    # Задаем фокусное расстояние (FOV) и расширяем плоскости отсечения,
    # чтобы стены и объекты гарантированно попадали в рендер и не исчезали.
    scene.camera.fov = (60.0, 60.0)
    scene.camera.z_near = 1.0
    scene.camera.z_far = 100000.0  # Огромное расстояние, чтобы видеть всю карту
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
) -> None:
    """Save all frames of an episode with action annotations.

    Creates a directory with PNG frames for each step, an actions log,
    and a JSON metadata file.

    Args:
        env: LightweightEnv instance.
        goal_pose: Target pose [6D].
        episode_poses: List of agent poses at each step.
        episode_actions: List of action descriptions (action_explanations).
        output_dir: Root directory for saving.
        episode_id: Unique episode identifier.
        result: Episode result ("success", "collision", "timeout").
    """
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

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

    trail_so_far: list[np.ndarray] = []

    if episode_poses:
        distance = float(
            np.linalg.norm(goal_pose[:3] - episode_poses[0][:3])
        )
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
        trail_so_far.append(episode_poses[0])

    for i in range(1, len(episode_poses)):
        trail_so_far.append(episode_poses[i])
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

    logger.info("Saved %d frames to %s", len(episode_poses), ep_dir)


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
