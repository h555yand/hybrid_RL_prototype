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

Visualization modes:
    - null/false: no visualization
    - text: save actions.txt and meta.json only
    - pictures: text + PNG frames
    - video: text + PNG frames + MP4 video
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
_AGENT_SPHERE_RADIUS = 1.0
_GOAL_SPHERE_RADIUS = 1.4
_TRAIL_SPHERE_RADIUS = 0.6
_GAZE_ARROW_LENGTH = 3
_NORM_EPSILON = 1e-12
_CROSS_PRODUCT_THRESHOLD = 1e-8
_TEXT_PADDING = 2
_TEXT_Y_START = 5
_TEXT_X_START = 5
_TEXT_LINE_SPACING = 3
_BG_ALPHA = 200
_FOV = 50.0
_CAMERA_DISTANCE_MULTIPLIER = 1.7
_DETAIL_STEPS = 100
_FRAME_INTERVAL = 50


def _build_scene(
    env: LightweightEnv,
    agent_pose: np.ndarray,
    goal_pose: np.ndarray,
    trail_poses: list[np.ndarray] | None = None,
    mesh_alpha: int = 80,
) -> trimesh.Scene:
    """Build a trimesh scene for rendering.

    Args:
        env: LightweightEnv instance.
        agent_pose: Agent pose [x, y, z, roll, pitch, yaw].
        goal_pose: Goal pose [x, y, z, roll, pitch, yaw].
        trail_poses: Optional list of previous agent poses for trail.
        mesh_alpha: Mesh transparency (0=invisible, 255=opaque).

    Returns:
        Configured trimesh Scene with mesh, agent, goal, and trail.
    """
    scene = trimesh.Scene()

    axis = trimesh.creation.axis(
        origin_size=2.0, axis_length=10
    )
    scene.add_geometry(axis, geom_name="world_axis")

    mesh_copy = env.mesh.copy()
    mesh_copy.visual.face_colors = [
        200, 200, 200, mesh_alpha
    ]
    scene.add_geometry(mesh_copy, geom_name="mesh")

    agent_pos = agent_pose[:3]
    agent_rot = agent_pose[3:]

    agent_sphere = trimesh.primitives.Sphere(
        radius=_AGENT_SPHERE_RADIUS,
        center=agent_pos,
    )
    agent_sphere.visual.face_colors = [
        0, 50, 255, 255
    ]
    scene.add_geometry(
        agent_sphere, geom_name="agent"
    )

    rot = Rotation.from_euler(
        "xyz", agent_rot, degrees=True
    )
    forward_dir = rot.apply([0, 0, -1])
    line_vertices = np.array([
        agent_pos,
        agent_pos
        + forward_dir * _GAZE_ARROW_LENGTH,
    ])
    arrow = trimesh.load_path(
        line_vertices,
        colors=[[255, 0, 0, 255]],
    )
    scene.add_geometry(arrow, geom_name="gaze")

    goal_sphere = trimesh.primitives.Sphere(
        radius=_GOAL_SPHERE_RADIUS,
        center=goal_pose[:3],
    )
    goal_sphere.visual.face_colors = [
        0, 255, 0, 255
    ]
    scene.add_geometry(
        goal_sphere, geom_name="goal"
    )

    if trail_poses:
        for i, pose in enumerate(trail_poses):
            trail_sphere = trimesh.primitives.Sphere(
                radius=_TRAIL_SPHERE_RADIUS,
                center=pose[:3],
            )
            trail_sphere.visual.face_colors = [
                255, 165, 0, 255
            ]
            scene.add_geometry(
                trail_sphere,
                geom_name=f"trail_{i}",
            )

    return scene


def _compute_camera_transform(
    scene: trimesh.Scene,
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    fixed_offset_multiplier: float = _CAMERA_DISTANCE_MULTIPLIER,
) -> np.ndarray:
    """Compute camera looking at midpoint between agent and goal.

    Camera positioned above and to the side, looking at the
    midpoint between agent and goal. Works for both solid
    and hollow objects.
    """
    midpoint = (agent_pos + goal_pos) / 2.0

    mesh_size = float(
        np.linalg.norm(
            scene.bounds[1] - scene.bounds[0]
        )
    )
    camera_distance = (
        mesh_size * fixed_offset_multiplier
    )

    mesh_center = scene.bounds.mean(axis=0)
    to_agent = agent_pos - mesh_center
    to_agent_horiz = to_agent.copy()
    to_agent_horiz[2] = 0
    to_agent_norm = np.linalg.norm(to_agent_horiz)

    if to_agent_norm > 1e-5:
        dir_horiz = to_agent_horiz / to_agent_norm
    else:
        to_goal = goal_pos - mesh_center
        to_goal[2] = 0
        tg_norm = np.linalg.norm(to_goal)
        if tg_norm > 1e-5:
            dir_horiz = -to_goal / tg_norm
        else:
            dir_horiz = np.array([1.0, 0.0, 0.0])

    camera_pos = (
        midpoint
        + dir_horiz * camera_distance * 0.7
    )
    camera_pos[2] = (
        max(agent_pos[2], goal_pos[2])
        + camera_distance * 0.5
    )

    forward = midpoint - camera_pos
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
    """Add text overlay on top of a PNG image."""
    img = Image.open(io.BytesIO(png_bytes))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/"
            "dejavu/DejaVuSansMono.ttf",
            _FONT_SIZE_OVERLAY,
        )
    except OSError:
        font = ImageFont.load_default()

    header = (
        f"Step {step_num} | "
        f"dist={distance:.1f}mm | {result}"
    )
    lines = [header]
    lines.extend(
        text[i:i + _MAX_LINE_LENGTH]
        for i in range(0, len(text), _MAX_LINE_LENGTH)
    )

    y = _TEXT_Y_START
    for line in lines:
        text_bbox = draw.textbbox(
            (_TEXT_X_START, y), line, font=font
        )
        draw.rectangle(
            [
                text_bbox[0] - _TEXT_PADDING,
                text_bbox[1] - 1,
                text_bbox[2] + _TEXT_PADDING,
                text_bbox[3] + 1,
            ],
            fill=(0, 0, 0, _BG_ALPHA),
        )
        draw.text(
            (_TEXT_X_START, y), line,
            fill="white", font=font,
        )
        y += (
            text_bbox[3] - text_bbox[1]
            + _TEXT_LINE_SPACING
        )

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
    """Render a split-view frame: solid + x-ray."""
    half_res = (resolution[0] // 2, resolution[1])

    cam = _compute_camera_transform(
        _build_scene(
            env, agent_pose, goal_pose,
            trail_poses, mesh_alpha=255,
        ),
        agent_pos=agent_pose[:3],
        goal_pos=goal_pose[:3],
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Solid view
        scene_solid = _build_scene(
            env, agent_pose, goal_pose,
            trail_poses, mesh_alpha=255,
        )
        scene_solid.camera_transform = cam
        scene_solid.camera.fov = (fov, fov)
        scene_solid.camera.z_near = 1.0
        scene_solid.camera.z_far = 100000.0
        scene_solid.camera.resolution = half_res
        png_solid = scene_solid.save_image(
            resolution=half_res,
            visible=False, smooth=False,
        )

        # X-ray view
        scene_xray = _build_scene(
            env, agent_pose, goal_pose,
            trail_poses, mesh_alpha=60,
        )
        scene_xray.camera_transform = cam
        scene_xray.camera.fov = (fov, fov)
        scene_xray.camera.z_near = 1.0
        scene_xray.camera.z_far = 100000.0
        scene_xray.camera.resolution = half_res
        png_xray = scene_xray.save_image(
            resolution=half_res,
            visible=False, smooth=False,
        )

        if (
            png_solid is None
            or png_xray is None
            or len(png_solid) < _MIN_RENDER_BYTES
            or len(png_xray) < _MIN_RENDER_BYTES
        ):
            msg = "Empty render output"
            raise RuntimeError(msg)

        # Merge side by side
        img_left = Image.open(io.BytesIO(png_solid))
        img_right = Image.open(io.BytesIO(png_xray))
        merged = Image.new(
            "RGBA", (resolution[0], resolution[1]),
        )
        merged.paste(img_left, (0, 0))
        merged.paste(img_right, (half_res[0], 0))

        # Text overlay
        if text:
            draw = ImageDraw.Draw(merged)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/"
                    "dejavu/DejaVuSansMono.ttf",
                    _FONT_SIZE_OVERLAY,
                )
            except OSError:
                font = ImageFont.load_default()

            header = (
                f"Step {step_num} | "
                f"dist={distance:.1f}mm | "
                f"{result}"
            )
            lines = [header]
            lines.extend(
                text[i:i + _MAX_LINE_LENGTH]
                for i in range(
                    0, len(text), _MAX_LINE_LENGTH
                )
            )
            y = _TEXT_Y_START
            for line in lines:
                bbox = draw.textbbox(
                    (_TEXT_X_START, y),
                    line, font=font,
                )
                draw.rectangle(
                    [
                        bbox[0] - _TEXT_PADDING,
                        bbox[1] - 1,
                        bbox[2] + _TEXT_PADDING,
                        bbox[3] + 1,
                    ],
                    fill=(0, 0, 0, _BG_ALPHA),
                )
                draw.text(
                    (_TEXT_X_START, y), line,
                    fill="white", font=font,
                )
                y += (
                    bbox[3] - bbox[1]
                    + _TEXT_LINE_SPACING
                )

            draw.text(
                (10, resolution[1] - 20),
                "SOLID", fill="white", font=font,
            )
            draw.text(
                (half_res[0] + 10, resolution[1] - 20),
                "X-RAY", fill="white", font=font,
            )

        buf = io.BytesIO()
        merged.save(buf, format="PNG")
        with filepath.open("wb") as f:
            f.write(buf.getvalue())

    except (RuntimeError, OSError):
        txt_path = filepath.with_suffix(".txt")
        with txt_path.open("w") as f:
            f.write(
                f"Step: {step_num}, "
                f"Distance: {distance:.1f}mm\n"
            )
            f.write(f"Agent: {agent_pose.tolist()}\n")
            f.write(f"Goal: {goal_pose.tolist()}\n")
            f.write(f"Action: {text}\n")
        logger.warning(
            "Render failed for %s",
            filepath, exc_info=True,
        )


def save_episode_frames(
    env: LightweightEnv,
    goal_pose: np.ndarray,
    episode_poses: list[np.ndarray],
    episode_actions: list[str],
    output_dir: Path,
    episode_id: str,
    result: str = "unknown",
    timeout_frame_interval: int = _FRAME_INTERVAL,
    render_mode: str = "text",
) -> None:
    """Save episode frames with action annotations.

    Args:
        env: LightweightEnv instance.
        goal_pose: Target pose [6D].
        episode_poses: List of agent poses at each step.
        episode_actions: List of action descriptions.
        output_dir: Root directory for saving.
        episode_id: Unique episode identifier.
        result: Episode result.
        timeout_frame_interval: Save every Nth frame after detail_steps.
        render_mode: "text" | "pictures" | "video".
    """
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # ═══ Always save action log ═══
    log_path = ep_dir / "actions.txt"
    with log_path.open("w") as f:
        f.write(f"Result: {result}\n")
        f.write(f"Goal: {goal_pose.tolist()}\n")
        f.write(f"Steps: {len(episode_actions)}\n")
        if episode_poses:
            f.write(
                f"Start: "
                f"{episode_poses[0].tolist()}\n"
            )
            f.write(
                f"End: "
                f"{episode_poses[-1].tolist()}\n"
            )
            start_dist = float(np.linalg.norm(
                goal_pose[:3] - episode_poses[0][:3]
            ))
            end_dist = float(np.linalg.norm(
                goal_pose[:3] - episode_poses[-1][:3]
            ))
            f.write(
                f"Start distance: "
                f"{start_dist:.1f}mm\n"
            )
            f.write(
                f"End distance: "
                f"{end_dist:.1f}mm\n"
            )
        f.write("\n")
        for i, action in enumerate(episode_actions):
            if i < len(episode_poses):
                dist = float(np.linalg.norm(
                    goal_pose[:3]
                    - episode_poses[i][:3]
                ))
            else:
                dist = 0.0
            f.write(
                f"Step {i+1:03d} "
                f"(dist={dist:.1f}mm): {action}\n"
            )

    meta = {
        "episode_id": episode_id,
        "result": result,
        "goal_pose": goal_pose.tolist(),
        "num_steps": len(episode_actions),
        "start_pose": (
            episode_poses[0].tolist()
            if episode_poses else None
        ),
        "end_pose": (
            episode_poses[-1].tolist()
            if episode_poses else None
        ),
    }
    meta_path = ep_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    if render_mode == "text":
        return

    # ═══ Render frames ═══
    last_step = len(episode_poses) - 1

    def should_save_frame(step_idx):
        if step_idx == 0:
            return True
        if step_idx <= _DETAIL_STEPS:
            return True
        if step_idx >= (last_step - int(timeout_frame_interval * 0.5)):
            return True
        if step_idx % timeout_frame_interval == 0:
            return True
        return False

    trail_so_far = []
    saved_count = 0

    if episode_poses:
        distance = float(np.linalg.norm(
            goal_pose[:3] - episode_poses[0][:3]
        ))
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
            distance = float(np.linalg.norm(
                goal_pose[:3]
                - episode_poses[i][:3]
            ))
            render_frame_to_file(
                env=env,
                agent_pose=episode_poses[i],
                goal_pose=goal_pose,
                filepath=(
                    ep_dir / f"step_{i:03d}.png"
                ),
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

    if render_mode == "video":
        create_video_from_episode(ep_dir, fps=5)


def visualize_scene(
    env: LightweightEnv,
    goal_pose: np.ndarray,
) -> None:
    """Show interactive visualization of the scene."""
    scene = _build_scene(
        env, env.get_pose(), goal_pose,
        mesh_alpha=255,
    )
    scene.show(smooth=False)


def visualize_agent_goal(
    env: LightweightEnv,
    agent_pose: np.ndarray,
    goal_pose: np.ndarray,
) -> None:
    """Show interactive visualization of agent and goal."""
    scene = _build_scene(
        env, agent_pose, goal_pose,
        mesh_alpha=255,
    )
    scene.show(smooth=False)


def create_video_from_episode(
    episode_dir: Path,
    output_path: Path | None = None,
    fps: int = 5,
) -> Path | None:
    """Create MP4 video from saved PNG frames."""
    import glob

    frames_pattern = str(
        episode_dir / "step_*.png"
    )
    frame_paths = sorted(glob.glob(frames_pattern))

    if not frame_paths:
        logger.warning(
            "No frames found in %s", episode_dir
        )
        return None

    if output_path is None:
        output_path = episode_dir / "episode.mp4"

    try:
        import imageio.v2 as imageio
        frames = [
            imageio.imread(p) for p in frame_paths
        ]
        imageio.mimwrite(
            str(output_path), frames,
            fps=fps, codec='libx264',
        )
        logger.info(
            "Video saved: %s (%d frames, %d fps)",
            output_path, len(frames), fps,
        )
        return output_path
    except Exception:
        try:
            import cv2
            first = cv2.imread(frame_paths[0])
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (w, h)
            )
            for p in frame_paths:
                writer.write(cv2.imread(p))
            writer.release()
            logger.info(
                "Video saved (cv2): %s (%d frames)",
                output_path, len(frame_paths),
            )
            return output_path
        except Exception:
            logger.warning(
                "Video creation skipped: %s",
                episode_dir,
            )
            return None


def create_all_videos(
    output_dir: Path, fps: int = 5
) -> None:
    """Create videos for all episodes in directory."""
    episode_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir()
    )
    for ep_dir in episode_dirs:
        create_video_from_episode(ep_dir, fps=fps)


class EpisodeVisualizer:
    """Manages episode visualization with per-level filtering.

    Saves limited number of episodes per result type per
    curriculum level.

    Visualization modes:
        - "text": actions.txt + meta.json only
        - "pictures": text + PNG frames (split view)
        - "video": text + PNG frames + MP4 video
    """

    def __init__(
        self,
        output_dir: Path,
        mesh_name: str = "",
        stage: str = "train",
        max_per_type_per_level: int = 3,
        num_levels: int = 3,
        timeout_frame_interval: int = _FRAME_INTERVAL,
        visualize_mode: str = "text",
    ):
        self.output_dir = (
            output_dir
            / f"visualizations_{stage}_{mesh_name}"
        )
        self.max_per_type = max_per_type_per_level
        self.timeout_frame_interval = (
            timeout_frame_interval
        )
        self.visualize_mode = visualize_mode

        self.counts: dict[str, int] = {}
        for level in range(num_levels):
            for result in (
                "success", "collision", "timeout"
            ):
                self.counts[
                    f"level_{level}_{result}"
                ] = 0

    def should_save(
        self, level: int, result: str
    ) -> bool:
        """Check if we should save this episode."""
        key = f"level_{level}_{result}"
        return (
            self.counts.get(key, 0)
            < self.max_per_type
        )

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
        """Save episode visualization if under limit."""
        if not self.should_save(level, result):
            return

        key = f"level_{level}_{result}"
        episode_id = (
            f"ep_{episode + 1:05d}"
            f"_L{level}_{result}"
        )

        save_episode_frames(
            env=env,
            goal_pose=goal_pose,
            episode_poses=poses,
            episode_actions=actions,
            output_dir=self.output_dir,
            episode_id=episode_id,
            result=result,
            timeout_frame_interval=(
                self.timeout_frame_interval
            ),
            render_mode=self.visualize_mode,
        )
        self.counts[key] += 1

    def get_stats(self) -> dict[str, int]:
        """Return current save counts."""
        return dict(self.counts)
    