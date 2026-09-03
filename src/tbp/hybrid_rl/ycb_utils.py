"""YCB object utilities for hybrid RL.

Converts YCB .glb meshes to .stl in mm for use with LightweightEnv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import trimesh

logger = logging.getLogger(__name__)

# Default YCB objects for evaluation
DEFAULT_YCB_OBJECTS = {
    "ycb_mug": "025_mug",
    "ycb_bowl": "024_bowl",
    "ycb_can": "002_master_chef_can",
    "ycb_banana": "011_banana",
    "ycb_cracker_box": "003_cracker_box",
    "ycb_sugar_box": "004_sugar_box",
    "ycb_tomato_can": "005_tomato_soup_can",
    "ycb_mustard": "006_mustard_bottle",
    "ycb_power_drill": "035_power_drill",
    "ycb_scissors": "037_scissors",
}


def convert_ycb_objects(
    ycb_src_dir: str | Path,
    output_dir: str | Path,
    objects: Dict[str, str] | None = None,
    scale_to_mm: bool = True,
) -> Dict[str, Path]:
    """Convert YCB .glb meshes to .stl files.

    Args:
        ycb_src_dir: Path to YCB meshes directory
            (e.g. ~/Downloads/simulator_data/objects/ycb/meshes)
        output_dir: Directory to save .stl files
        objects: Dict of {local_name: ycb_folder_name}.
            If None, uses DEFAULT_YCB_OBJECTS.
        scale_to_mm: If True, scale from meters to mm.

    Returns:
        Dict of {local_name: Path to .stl file}
    """
    ycb_src = Path(ycb_src_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if objects is None:
        objects = DEFAULT_YCB_OBJECTS

    converted = {}

    for local_name, ycb_name in objects.items():
        stl_path = out_dir / f"{local_name}.stl"

        # Skip if already converted
        if stl_path.exists():
            logger.info("  %s: already exists, skipping", local_name)
            converted[local_name] = stl_path
            continue

        glb_path = ycb_src / ycb_name / "google_16k" / "textured.glb"
        if not glb_path.exists():
            logger.warning("  SKIP %s: %s not found", ycb_name, glb_path)
            continue

        try:
            mesh = trimesh.load(str(glb_path), force="mesh")

            if scale_to_mm:
                mesh.vertices *= 1000.0

            mesh.export(str(stl_path))
            converted[local_name] = stl_path

            logger.info(
                "  %s (%s): %d verts, extents=%s mm → %s",
                local_name,
                ycb_name,
                len(mesh.vertices),
                mesh.extents.round(1).tolist(),
                stl_path,
            )
        except Exception:
            logger.error("  ERROR converting %s", ycb_name, exc_info=True)

    logger.info("Converted %d/%d YCB objects", len(converted), len(objects))
    return converted


def discover_ycb_objects(ycb_src_dir: str | Path) -> List[str]:
    """List available YCB object folders.

    Args:
        ycb_src_dir: Path to YCB meshes directory.

    Returns:
        List of YCB object folder names.
    """
    ycb_src = Path(ycb_src_dir)
    if not ycb_src.exists():
        return []

    objects = []
    for d in sorted(ycb_src.iterdir()):
        if d.is_dir():
            glb = d / "google_16k" / "textured.glb"
            if glb.exists():
                objects.append(d.name)
    return objects
