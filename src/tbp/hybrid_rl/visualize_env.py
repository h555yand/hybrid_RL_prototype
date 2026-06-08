import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tbp.hybrid_rl.lightweight_env import LightweightEnv


def visualize_scene(env: LightweightEnv, goal_pose: np.ndarray):
    """
    Визуализирует сцену: меш, агента (сфера + стрелка взгляда) и цель (сфера).

    Args:
        env: экземпляр LightweightEnv
        goal_pose: 6D вектор [x, y, z, rx, ry, rz] — поза цели
    """
    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=2.0, axis_length=10)
    #scene.add_geometry(axis)
    # Добавляем в сцену, обязательно указав уникальное имя geom_name
    scene.add_geometry(axis, geom_name="world_axis")

    # Добавляем меш
    scene.add_geometry(env.mesh, geom_name="mesh")

    # Позиция и ориентация агента
    agent_pos = env.agent_pos
    agent_rot = env.agent_rot

    # Сфера — позиция агента (синяя)
    agent_sphere = trimesh.primitives.Sphere(radius=0.5, center=agent_pos)
    agent_sphere.visual.face_colors = [0, 0, 255, 255]  # синий
    scene.add_geometry(agent_sphere, geom_name="agent")

    # Стрелка — направление взгляда (красная)
    rot = R.from_euler("xyz", agent_rot, degrees=True)
    forward_dir = rot.apply([0, 0, -1])

    # Создаем массив точек для линии
    line_vertices = np.array([
        agent_pos,
        agent_pos + forward_dir * 3
    ])
    
    # Задаем цвет RGBA: [Красный, Зеленый, Синий, Альфа-канал]
    # Передаем список цветов по количеству сегментов (в данном случае 1 сегмент)
    arrow = trimesh.load_path(
        line_vertices, 
        colors=[[255, 0, 0, 255]]
    )
    
    scene.add_geometry(arrow, geom_name="gaze")

    # Цель — зелёная сфера
    goal_pos = goal_pose[:3]
    goal_sphere = trimesh.primitives.Sphere(radius=0.7, center=goal_pos)
    goal_sphere.visual.face_colors = [0, 255, 0, 255]  # зелёный
    scene.add_geometry(goal_sphere, geom_name="goal")

    # Показать сцену
    scene.show(smooth=False)
    #scene.show(viewer='gl', flags={'axis': True})


def visualize_agent_goal(env: LightweightEnv, agent_pose: np.ndarray, goal_pose: np.ndarray):
    """
    Визуализирует сцену: меш, агента (сфера + стрелка взгляда) и цель (сфера).

    Args:
        env: экземпляр LightweightEnv
        goal_pose: 6D вектор [x, y, z, rx, ry, rz] — поза цели
    """
    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=2.0, axis_length=10)
    #scene.add_geometry(axis)
    # Добавляем в сцену, обязательно указав уникальное имя geom_name
    scene.add_geometry(axis, geom_name="world_axis")

    # Добавляем меш
    scene.add_geometry(env.mesh, geom_name="mesh")

    # Позиция и ориентация агента
    agent_pos = agent_pose[:3]
    agent_rot = agent_pose[3:]

    # Сфера — позиция агента (синяя)
    agent_sphere = trimesh.primitives.Sphere(radius=0.5, center=agent_pos)
    agent_sphere.visual.face_colors = [0, 0, 255, 255]  # синий
    scene.add_geometry(agent_sphere, geom_name="agent")

    # Стрелка — направление взгляда (красная)
    rot = R.from_euler("xyz", agent_rot, degrees=True)
    forward_dir = rot.apply([0, 0, -1])
    # Создаем массив точек для линии
    line_vertices = np.array([
        agent_pos,
        agent_pos + forward_dir * 3
    ])
    # Задаем цвет RGBA: [Красный, Зеленый, Синий, Альфа-канал]
    # Передаем список цветов по количеству сегментов (в данном случае 1 сегмент)
    arrow = trimesh.load_path(
        line_vertices, 
        colors=[[255, 0, 0, 255]]
    )
    scene.add_geometry(arrow, geom_name="gaze")

    # Цель — зелёная сфера
    goal_pos = goal_pose[:3]
    goal_sphere = trimesh.primitives.Sphere(radius=0.7, center=goal_pos)
    goal_sphere.visual.face_colors = [0, 255, 0, 255]  # зелёный
    scene.add_geometry(goal_sphere, geom_name="goal")

    # Показать сцену
    scene.show(smooth=False)
    #scene.show(viewer='gl', flags={'axis': True})
