# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Factory functions for creating demo 3D meshes.

Provides functions to generate parametric meshes (mug, cup, cube, etc.)
for training and evaluation of the RL navigation agent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def create_thin_cylinder(
    radius: float = 5.0,
    height: float = 120.0,
    sections: int = 32,
) -> trimesh.Trimesh:
    """Create a thin cylinder (pencil-like) mesh.

    Locally extreme principal curvature due to small radius.

    Args:
        radius: Radius in mm (small for pencil-like shape).
        height: Height in mm.
        sections: Number of circumference segments.

    Returns:
        Thin cylinder trimesh.
    """
    return trimesh.primitives.Cylinder(
        radius=radius, height=height, sections=sections
    )


def create_flat_square(
    width: float = 100.0,
    height: float = 100.0,
    thickness: float = 3.0,
) -> trimesh.Trimesh:
    """Create a flat square (sheet of paper-like) mesh.

    Locally flat surface with sharp edges.

    Args:
        width: Width in mm.
        height: Height in mm.
        thickness: Thickness in mm (thin).

    Returns:
        Flat square trimesh.
    """
    return trimesh.primitives.Box(
        extents=[width, height, thickness]
    )


def create_cone(
    radius: float = 40.0,
    height: float = 80.0,
    sections: int = 64,
) -> trimesh.Trimesh:
    """Create a cone mesh.

    Varying curvature from tip to base.

    Args:
        radius: Base radius in mm.
        height: Height in mm.
        sections: Number of circumference segments.

    Returns:
        Cone trimesh.
    """
    # trimesh doesn't have a cone primitive,
    # build from vertices
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)

    # Apex at top
    apex = np.array([[0.0, 0.0, height / 2]])

    # Base circle at bottom
    base_pts = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full(sections, -height / 2),
    ])

    # Center of base
    base_center = np.array([[0.0, 0.0, -height / 2]])

    vertices = np.vstack([apex, base_pts, base_center])
    apex_idx = 0
    base_start = 1
    center_idx = sections + 1

    faces = []
    # Side faces (apex to base edge)
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([apex_idx, base_start + i, base_start + j])

    # Base faces (center to base edge)
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([center_idx, base_start + j, base_start + i])

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=np.array(faces)
    )
    mesh.fix_normals()
    return mesh


def create_tea_cup(  # noqa: PLR0913
    bottom_radius: float = 25.0,
    top_radius: float = 42.0,
    body_height: float = 60.0,
    wall_thickness: float = 2.0,
    bottom_thickness: float = 2.5,
    handle_radius_major: float = 14.0,
    handle_radius_minor: float = 2.5,
    handle_angle_deg: float = 140.0,
    handle_segments: int = 20,
    body_segments: int = 64,
    circle_points: int = 8,
) -> trimesh.Trimesh:
    """Create a tea cup mesh with a conical body and toroidal handle.

    Args:
        bottom_radius: Bottom radius of the cup body in mm.
        top_radius: Top radius of the cup body in mm.
        body_height: Height of the cup body in mm.
        wall_thickness: Wall thickness in mm.
        bottom_thickness: Bottom thickness in mm.
        handle_radius_major: Major radius of the handle arc in mm.
        handle_radius_minor: Cross-section radius of the handle in mm.
        handle_angle_deg: Angular span of the handle in degrees.
        handle_segments: Number of segments along the handle arc.
        body_segments: Number of segments around the body circumference.
        circle_points: Number of points in handle cross-section.

    Returns:
        Combined trimesh of cup body and handle.
    """
    angles = np.linspace(0, 2 * np.pi, body_segments, endpoint=False)

    bottom_z = -body_height / 2
    top_z = body_height / 2
    bottom_pts = np.column_stack([
        bottom_radius * np.cos(angles),
        bottom_radius * np.sin(angles),
        np.full(body_segments, bottom_z),
    ])
    top_pts = np.column_stack([
        top_radius * np.cos(angles),
        top_radius * np.sin(angles),
        np.full(body_segments, top_z),
    ])
    outer = trimesh.convex.convex_hull(np.vstack([bottom_pts, top_pts]))

    inner_bottom_radius = bottom_radius - wall_thickness
    inner_top_radius = top_radius - wall_thickness
    inner_bottom_z = bottom_z + bottom_thickness
    inner_bottom_pts = np.column_stack([
        inner_bottom_radius * np.cos(angles),
        inner_bottom_radius * np.sin(angles),
        np.full(body_segments, inner_bottom_z),
    ])
    inner_top_pts = np.column_stack([
        inner_top_radius * np.cos(angles),
        inner_top_radius * np.sin(angles),
        np.full(body_segments, top_z),
    ])
    inner = trimesh.convex.convex_hull(
        np.vstack([inner_bottom_pts, inner_top_pts])
    )

    body = outer.difference(inner)
    handle = _create_toroidal_handle(
        center_x=(bottom_radius + top_radius) / 2,
        center_z=0.0,
        radius_major=handle_radius_major,
        radius_minor=handle_radius_minor,
        angle_deg=handle_angle_deg,
        segments=handle_segments,
        circle_points=circle_points,
    )

    return trimesh.util.concatenate([body, handle])


def create_mug(  # noqa: PLR0913
    body_radius: float = 30.0,
    body_height: float = 80.0,
    wall_thickness: float = 3.0,
    bottom_thickness: float = 3.0,
    handle_radius_major: float = 15.0,
    handle_radius_minor: float = 4.0,
    handle_angle_deg: float = 180.0,
    handle_segments: int = 32,
    body_segments: int = 64,
    circle_points: int = 8,
) -> trimesh.Trimesh:
    """Create a mug mesh with a cylindrical body and toroidal handle.

    Args:
        body_radius: Radius of the mug body in mm.
        body_height: Height of the mug body in mm.
        wall_thickness: Wall thickness in mm.
        bottom_thickness: Bottom thickness in mm.
        handle_radius_major: Major radius of the handle arc in mm.
        handle_radius_minor: Cross-section radius of the handle in mm.
        handle_angle_deg: Angular span of the handle in degrees.
        handle_segments: Number of segments along the handle arc.
        body_segments: Number of segments around the body circumference.
        circle_points: Number of points in handle cross-section.

    Returns:
        Combined trimesh of mug body and handle.
    """
    outer = trimesh.primitives.Cylinder(
        radius=body_radius, height=body_height, sections=body_segments,
    )
    inner = trimesh.primitives.Cylinder(
        radius=body_radius - wall_thickness,
        height=body_height - bottom_thickness,
        sections=body_segments,
    )
    inner.apply_translation([0, 0, bottom_thickness / 2.0])
    body = outer.difference(inner)

    handle = _create_toroidal_handle(
        center_x=body_radius,
        center_z=0.0,
        radius_major=handle_radius_major,
        radius_minor=handle_radius_minor,
        angle_deg=handle_angle_deg,
        segments=handle_segments,
        circle_points=circle_points,
    )

    return trimesh.util.concatenate([body, handle])


def _create_toroidal_handle(
    center_x: float,
    center_z: float,
    radius_major: float,
    radius_minor: float,
    angle_deg: float,
    segments: int,
    circle_points: int,
) -> trimesh.Trimesh:
    """Create a toroidal handle mesh.

    Args:
        center_x: X position of the handle arc center.
        center_z: Z position of the handle arc center.
        radius_major: Major radius of the arc.
        radius_minor: Cross-section radius.
        angle_deg: Angular span in degrees.
        segments: Number of segments along the arc.
        circle_points: Number of points in cross-section.

    Returns:
        Handle trimesh.
    """
    handle_angles = np.linspace(
        -np.radians(angle_deg) / 2,
        np.radians(angle_deg) / 2,
        segments,
    )

    vertices_all = []
    for angle in handle_angles:
        center = np.array([
            center_x + radius_major * np.cos(angle),
            0.0,
            center_z + radius_major * np.sin(angle),
        ])

        radial = center - np.array([center_x, 0.0, center_z])
        radial_len = np.linalg.norm(radial)
        if radial_len > 1e-12:
            radial /= radial_len
        else:
            radial = np.array([1.0, 0.0, 0.0])

        tangent = np.array([
            -radius_major * np.sin(angle),
            0.0,
            radius_major * np.cos(angle),
        ])
        tangent /= np.linalg.norm(tangent) + 1e-12

        binormal = np.cross(tangent, radial)
        binormal /= np.linalg.norm(binormal) + 1e-12

        for j in range(circle_points):
            theta = 2.0 * np.pi * j / circle_points
            point = (
                center
                + radius_minor * np.cos(theta) * radial
                + radius_minor * np.sin(theta) * binormal
            )
            vertices_all.append(point)

    vertices_all = np.array(vertices_all)

    faces_all = []
    for i in range(segments - 1):
        for j in range(circle_points):
            j_next = (j + 1) % circle_points
            v0 = i * circle_points + j
            v1 = i * circle_points + j_next
            v2 = (i + 1) * circle_points + j
            v3 = (i + 1) * circle_points + j_next
            faces_all.append([v0, v2, v1])
            faces_all.append([v1, v2, v3])

    handle = trimesh.Trimesh(
        vertices=vertices_all, faces=np.array(faces_all)
    )
    handle.fix_normals()
    return handle


def create_vase(
    bottom_radius: float = 15.0,
    body_radius: float = 35.0,
    neck_radius: float = 12.0,
    top_radius: float = 18.0,
    body_height: float = 45.0,
    neck_height: float = 25.0,
    wall_thickness: float = 2.0,
    bottom_thickness: float = 2.5,
    body_segments: int = 64,
) -> trimesh.Trimesh:
    """Create a vase mesh with bulging body and narrow neck.

    Profile (cross-section):
        top_radius (18mm)
           |    |
           |    |  neck (25mm)
          neck_radius (12mm)
         /      \\
        /        \\  body (45mm)
       body_radius (35mm)
        \\        /
         \\______/
       bottom_radius (15mm)

    Args:
        bottom_radius: Bottom radius in mm.
        body_radius: Maximum body radius in mm.
        neck_radius: Narrowest neck radius in mm.
        top_radius: Top opening radius in mm.
        body_height: Height of the body section in mm.
        neck_height: Height of the neck section in mm.
        wall_thickness: Wall thickness in mm.
        bottom_thickness: Bottom thickness in mm.
        body_segments: Number of segments around circumference.

    Returns:
        Vase trimesh.
    """
    total_height = body_height + neck_height
    n_profile = 20
    angles = np.linspace(0, 2 * np.pi, body_segments, endpoint=False)

    # Build profile: radius as function of height
    # Body: bottom_radius → body_radius → neck_radius (sine curve)
    # Neck: neck_radius → top_radius (linear)
    profile_heights = []
    profile_radii_outer = []
    profile_radii_inner = []

    # Body section (0 to body_height)
    for i in range(n_profile):
        t = i / (n_profile - 1)
        z = -total_height / 2 + t * body_height
        # Sine bulge: starts at bottom_radius, peaks at body_radius,
        # ends at neck_radius
        if t < 0.5:
            r = bottom_radius + (body_radius - bottom_radius) * np.sin(
                t * np.pi
            )
        else:
            r = neck_radius + (body_radius - neck_radius) * np.sin(
                (1 - t) * np.pi
            )
        profile_heights.append(z)
        profile_radii_outer.append(r)
        r_inner = max(r - wall_thickness, 1.0)
        profile_radii_inner.append(r_inner)

    # Neck section (body_height to total_height)
    for i in range(1, n_profile):
        t = i / (n_profile - 1)
        z = -total_height / 2 + body_height + t * neck_height
        r = neck_radius + (top_radius - neck_radius) * t
        profile_heights.append(z)
        profile_radii_outer.append(r)
        r_inner = max(r - wall_thickness, 1.0)
        profile_radii_inner.append(r_inner)

    n_rings = len(profile_heights)

    # Build outer shell vertices
    outer_verts = []
    for i in range(n_rings):
        r = profile_radii_outer[i]
        z = profile_heights[i]
        for angle in angles:
            outer_verts.append([
                r * np.cos(angle),
                r * np.sin(angle),
                z,
            ])
    outer_verts = np.array(outer_verts)
    outer_hull = trimesh.convex.convex_hull(outer_verts)

    # Build inner shell vertices (skip bottom for solid bottom)
    inner_start = 1  # skip first ring (bottom is solid)
    inner_verts = []
    for i in range(inner_start, n_rings):
        r = profile_radii_inner[i]
        z = profile_heights[i]
        if i == inner_start:
            z = profile_heights[inner_start] + bottom_thickness
        for angle in angles:
            inner_verts.append([
                r * np.cos(angle),
                r * np.sin(angle),
                z,
            ])
    inner_verts = np.array(inner_verts)
    inner_hull = trimesh.convex.convex_hull(inner_verts)

    return outer_hull.difference(inner_hull)


def prepare_demo_meshes(data_dir: Path) -> None:
    """Generate and save all demo meshes to a directory.

    Creates geometric primitives and complex objects for training.

    Args:
        data_dir: Directory to save mesh files to.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Basic primitives
    trimesh.primitives.Box(extents=[80, 80, 80]).export(
        str(data_dir / "cube.stl")
    )
    trimesh.primitives.Sphere(radius=50).export(
        str(data_dir / "sphere.stl")
    )
    trimesh.primitives.Cylinder(radius=35, height=100).export(
        str(data_dir / "cylinder.stl")
    )

    # Additional primitives for diverse local geometry
    create_thin_cylinder().export(
        str(data_dir / "thin_cylinder.stl")
    )
    create_flat_square().export(
        str(data_dir / "flat_square.stl")
    )
    create_cone().export(
        str(data_dir / "cone.stl")
    )

    # Complex objects
    create_mug().export(str(data_dir / "mug.stl"))
    create_tea_cup().export(str(data_dir / "cup.stl"))
    create_vase().export(str(data_dir / "vase.stl"))


# show objects
# trimesh.primitives.Box(extents=[80, 80, 80]).show()
# trimesh.primitives.Cylinder(radius=35, height=100).show()
# create_mug().show()
# create_tea_cup().show()
# create_vase().show()
