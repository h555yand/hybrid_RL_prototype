# visualize_env.py
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import io
import json
import logging

logger = logging.getLogger(__name__)


def _build_scene(
    env,
    agent_pose: np.ndarray,
    goal_pose: np.ndarray,
    trail_poses: Optional[List[np.ndarray]] = None,
) -> trimesh.Scene:
    """Строит сцену для рендеринга."""
    scene = trimesh.Scene()

    axis = trimesh.creation.axis(origin_size=2.0, axis_length=10)
    scene.add_geometry(axis, geom_name="world_axis")
    scene.add_geometry(env.mesh, geom_name="mesh")

    agent_pos = agent_pose[:3]
    agent_rot = agent_pose[3:]

    agent_sphere = trimesh.primitives.Sphere(radius=0.5, center=agent_pos)
    agent_sphere.visual.face_colors = [0, 0, 255, 255]
    scene.add_geometry(agent_sphere, geom_name="agent")

    rot = R.from_euler("xyz", agent_rot, degrees=True)
    forward_dir = rot.apply([0, 0, -1])
    line_vertices = np.array([agent_pos, agent_pos + forward_dir * 3])
    arrow = trimesh.load_path(line_vertices, colors=[[255, 0, 0, 255]])
    scene.add_geometry(arrow, geom_name="gaze")

    goal_sphere = trimesh.primitives.Sphere(radius=0.7, center=goal_pose[:3])
    goal_sphere.visual.face_colors = [0, 255, 0, 255]
    scene.add_geometry(goal_sphere, geom_name="goal")

    if trail_poses:
        for i, pose in enumerate(trail_poses):
            trail_sphere = trimesh.primitives.Sphere(radius=0.3, center=pose[:3])
            trail_sphere.visual.face_colors = [255, 255, 0, 128]
            scene.add_geometry(trail_sphere, geom_name=f"trail_{i}")

    return scene


def _compute_camera_transform(
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    mesh_bounds: np.ndarray,
) -> np.ndarray:
    """Вычисляет фиксированную позицию камеры сбоку-сверху."""
    center = (agent_pos + goal_pos) / 2.0
    span = np.linalg.norm(agent_pos - goal_pos)
    mesh_size = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])
    camera_distance = max(span, mesh_size) * 1.8

    camera_pos = center + np.array([
        camera_distance * 0.7,
        camera_distance * 0.5,
        camera_distance * 0.7,
    ])

    forward = center - camera_pos
    forward /= (np.linalg.norm(forward) + 1e-12)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)
    up /= (np.linalg.norm(up) + 1e-12)

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
    """Добавляет текст поверх PNG изображения."""
    img = Image.open(io.BytesIO(png_bytes))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    header = f"Step {step_num} | dist={distance:.1f}mm | {result}"

    lines = [header]
    max_line_len = 100
    for i in range(0, len(text), max_line_len):
        lines.append(text[i:i + max_line_len])

    y = 5
    for line in lines:
        text_bbox = draw.textbbox((5, y), line, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1],
            fill=(0, 0, 0, 200),
        )
        draw.text((5, y), line, fill="white", font=font)
        y += text_bbox[3] - text_bbox[1] + 3

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_frame_to_file(
    env,
    agent_pose: np.ndarray,
    goal_pose: np.ndarray,
    filepath: Path,
    text: str = "",
    step_num: int = 0,
    distance: float = 0.0,
    result: str = "",
    trail_poses: Optional[List[np.ndarray]] = None,
    resolution: Tuple[int, int] = (800, 600),
):
    """Рендерит кадр в PNG файл с текстовой подписью."""
    scene = _build_scene(env, agent_pose, goal_pose, trail_poses)

    camera_transform = _compute_camera_transform(
        agent_pose[:3], goal_pose[:3], env.mesh.bounds,
    )
    scene.camera_transform = camera_transform

    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        png_data = scene.save_image(resolution=resolution)
        if text:
            png_data = _add_text_to_image(png_data, text, step_num, distance, result)
        with open(filepath, "wb") as f:
            f.write(png_data)
    except Exception as e:
        txt_path = filepath.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write(f"Render failed: {e}\n")
            f.write(f"Step: {step_num}\n")
            f.write(f"Agent: {agent_pose.tolist()}\n")
            f.write(f"Goal: {goal_pose.tolist()}\n")
            f.write(f"Action: {text}\n")
        logger.warning(f"Render failed for {filepath}: {e}")


def save_episode_frames(
    env,
    goal_pose: np.ndarray,
    episode_poses: List[np.ndarray],
    episode_actions: List[str],
    output_dir: Path,
    episode_id: str,
    result: str = "unknown",
):
    """Сохраняет все кадры эпизода с подписями действий.

    Args:
        env: LightweightEnv
        goal_pose: целевая поза [6D]
        episode_poses: список поз агента на каждом шаге
        episode_actions: список описаний действий (action_explanations)
        output_dir: корневая директория для сохранения
        episode_id: уникальный идентификатор эпизода
        result: "success", "collision", "timeout"
    """
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Сохранить лог действий
    log_path = ep_dir / "actions.txt"
    with open(log_path, "w") as f:
        f.write(f"Result: {result}\n")
        f.write(f"Goal: {goal_pose.tolist()}\n")
        f.write(f"Steps: {len(episode_actions)}\n")
        if episode_poses:
            f.write(f"Start: {episode_poses[0].tolist()}\n")
            f.write(f"End: {episode_poses[-1].tolist()}\n")
            start_dist = float(np.linalg.norm(goal_pose[:3] - episode_poses[0][:3]))
            end_dist = float(np.linalg.norm(goal_pose[:3] - episode_poses[-1][:3]))
            f.write(f"Start distance: {start_dist:.1f}mm\n")
            f.write(f"End distance: {end_dist:.1f}mm\n")
        f.write("\n")
        for i, action in enumerate(episode_actions):
            pose_str = episode_poses[i].tolist() if i < len(episode_poses) else "N/A"
            dist = float(np.linalg.norm(goal_pose[:3] - episode_poses[i][:3])) if i < len(episode_poses) else 0
            f.write(f"Step {i:03d} (dist={dist:.1f}mm): {action}\n")

    # Сохранить метаданные
    meta = {
        "episode_id": episode_id,
        "result": result,
        "goal_pose": goal_pose.tolist(),
        "num_steps": len(episode_actions),
        "start_pose": episode_poses[0].tolist() if episode_poses else None,
        "end_pose": episode_poses[-1].tolist() if episode_poses else None,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Рендерить кадры
    trail_so_far = []

    # Кадр 0: начальное состояние (до первого действия)
    if episode_poses:
        distance = float(np.linalg.norm(goal_pose[:3] - episode_poses[0][:3]))
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

    # Кадры 1..N: после каждого действия
    for i in range(1, len(episode_poses)):
        trail_so_far.append(episode_poses[i])
        action_text = episode_actions[i - 1] if (i - 1) < len(episode_actions) else "unknown"
        distance = float(np.linalg.norm(goal_pose[:3] - episode_poses[i][:3]))

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

    logger.info(f"Saved {len(episode_poses)} frames to {ep_dir}")


def visualize_scene(env, goal_pose: np.ndarray):
    """Интерактивная визуализация сцены."""
    scene = _build_scene(env, env.get_pose(), goal_pose)
    scene.show(smooth=False)


def visualize_agent_goal(env, agent_pose: np.ndarray, goal_pose: np.ndarray):
    """Интерактивная визуализация агента и цели."""
    scene = _build_scene(env, agent_pose, goal_pose)
    scene.show(smooth=False)
