#!/usr/bin/env python3
"""Export a point-prompt visualization GLB via Blender."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--points-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args(argv)


def load_glb(glb_path: Path) -> None:
    bpy.ops.import_scene.gltf(filepath=str(glb_path), merge_vertices=False)
    bpy.context.view_layer.update()


def get_scene_meshes() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]


def scene_bbox() -> tuple[Vector, Vector]:
    bbox_min = Vector((float("inf"), float("inf"), float("inf")))
    bbox_max = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in get_scene_meshes():
        found = True
        for coord in obj.bound_box:
            world_coord = obj.matrix_world @ Vector(coord)
            bbox_min.x = min(bbox_min.x, world_coord.x)
            bbox_min.y = min(bbox_min.y, world_coord.y)
            bbox_min.z = min(bbox_min.z, world_coord.z)
            bbox_max.x = max(bbox_max.x, world_coord.x)
            bbox_max.y = max(bbox_max.y, world_coord.y)
            bbox_max.z = max(bbox_max.z, world_coord.z)
    if not found:
        raise RuntimeError("No mesh objects found in the scene")
    return bbox_min, bbox_max


def add_prompt_spheres(points: list[list[float]]) -> None:
    bbox_min, bbox_max = scene_bbox()
    radius = max(0.03 * max(bbox_max - bbox_min), 1e-3)
    material = bpy.data.materials.new(name="PointPromptMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.2
    for prompt_idx, point in enumerate(points):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=tuple(float(v) for v in point))
        sphere = bpy.context.active_object
        sphere.name = f"point_prompt_{prompt_idx}"
        sphere.data.materials.clear()
        sphere.data.materials.append(material)


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    load_glb(args.mesh_path.resolve())
    points = json.loads(args.points_json.read_text(encoding="utf-8"))
    if not isinstance(points, list):
        raise ValueError("Expected points JSON to be a list of [x, y, z] coordinates")
    add_prompt_spheres(points)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(args.output_path),
        export_format="GLB",
        use_selection=False,
    )


if __name__ == "__main__":
    main()
