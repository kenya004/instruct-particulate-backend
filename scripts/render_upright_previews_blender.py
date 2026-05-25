#!/usr/bin/env python3
"""Render upright-orientation preview thumbnails inside headless Blender."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:  # pragma: no cover - Blender-only runtime dependency
    bpy = None
    Matrix = None
    Vector = None


if bpy is not None:
    IMPORT_FUNCTIONS = {
        "obj": bpy.ops.wm.obj_import,
        "glb": bpy.ops.import_scene.gltf,
        "gltf": bpy.ops.import_scene.gltf,
        "fbx": bpy.ops.import_scene.fbx,
        "stl": bpy.ops.wm.stl_import,
        "dae": bpy.ops.wm.collada_import,
        "ply": bpy.ops.wm.ply_import,
    }
else:  # pragma: no cover
    IMPORT_FUNCTIONS = {}


UP_DIR_ROTATIONS = {
    "+X": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    "-X": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    "+Y": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "-Y": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    "+Z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "-Z": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--camera-distance", type=float, default=2.25)
    parser.add_argument("--pitch-deg", type=float, default=58.0)
    parser.add_argument("--azimuth-deg", type=float, default=225.0)
    parser.add_argument("--up-dirs", type=str, required=True)
    return parser.parse_args(argv)


def canonicalize_up_dir(up_dir: str) -> str:
    token = str(up_dir).strip().upper()
    if token in {"X", "Y", "Z"}:
        token = f"+{token}"
    if token not in UP_DIR_ROTATIONS:
        raise ValueError(f"Invalid up direction: {up_dir}")
    return token


def up_dir_slug(up_dir: str) -> str:
    return up_dir.replace("+", "pos").replace("-", "neg")


def reset_scene() -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    return camera


def configure_render(resolution: int) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
    except TypeError:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.97, 0.98, 0.99)
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if hasattr(scene, "display") and hasattr(scene.display, "shading"):
        shading = scene.display.shading
        for attr, value in (
            ("light", "STUDIO"),
            ("color_type", "TEXTURE"),
            ("background_type", "COLOR"),
            ("show_cavity", False),
            ("show_object_outline", False),
            ("show_xray", False),
        ):
            try:
                setattr(shading, attr, value)
            except (AttributeError, TypeError):
                pass
        try:
            shading.background_color = (0.97, 0.98, 0.99)
        except (AttributeError, TypeError):
            pass
        try:
            scene.display.render_aa = "8"
        except (AttributeError, TypeError):
            pass
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 8


def load_object(mesh_path: Path) -> None:
    suffix = mesh_path.suffix.lower().lstrip(".")
    if suffix not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported mesh type for upright preview render: {mesh_path.suffix}")
    importer = IMPORT_FUNCTIONS[suffix]
    if suffix in {"glb", "gltf"}:
        try:
            result = importer(filepath=str(mesh_path), merge_vertices=False)
        except TypeError:
            result = importer(filepath=str(mesh_path))
    elif suffix == "obj":
        try:
            result = importer(filepath=str(mesh_path), forward_axis="Y", up_axis="Z")
        except TypeError:
            result = importer(filepath=str(mesh_path))
    else:
        result = importer(filepath=str(mesh_path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender failed to import mesh: {mesh_path}")
    bpy.context.view_layer.update()


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]


def create_preview_root() -> bpy.types.Object:
    root = bpy.data.objects.new("UprightPreviewRoot", None)
    bpy.context.scene.collection.objects.link(root)
    imported_roots = [
        obj
        for obj in bpy.context.scene.objects.values()
        if obj.parent is None and obj.type not in {"CAMERA", "LIGHT"}
    ]
    for obj in imported_roots:
        if obj == root:
            continue
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    return root


def assign_default_material() -> None:
    default_material = None
    for obj in mesh_objects():
        if len(obj.data.materials) > 0:
            continue
        if default_material is None:
            default_material = bpy.data.materials.new("Preview_Default_Material")
            default_material.diffuse_color = (0.72, 0.78, 0.86, 1.0)
        obj.data.materials.append(default_material)


def scene_bbox() -> tuple[Vector, Vector]:
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in mesh_objects():
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
        raise RuntimeError("No mesh objects found in the imported scene")
    return bbox_min, bbox_max


def rotation_matrix_for_up(up_dir: str) -> Matrix:
    rows = UP_DIR_ROTATIONS[canonicalize_up_dir(up_dir)]
    return Matrix(rows).to_4x4()


def import_basis_to_blender(mesh_path: Path) -> Matrix:
    suffix = mesh_path.suffix.lower().lstrip(".")
    if suffix in {"glb", "gltf"}:
        return rotation_matrix_for_up("+Y")
    return Matrix.Identity(4)


def orient_and_normalize(
    root: bpy.types.Object,
    *,
    up_dir: str,
    import_basis: Matrix,
) -> None:
    # Blender's glTF importer maps source +Y-up coordinates into Blender +Z-up
    # space. Undo that import basis first, then apply the same source-space
    # up-direction rotation used by the inference/trimesh path.
    rotation = rotation_matrix_for_up(up_dir) @ import_basis.inverted()
    root.matrix_world = rotation
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    max_extent = max(float(extent.x), float(extent.y), float(extent.z), 1e-6)
    scale = 1.0 / max_extent
    scale_matrix = Matrix.Diagonal((scale, scale, scale, 1.0))
    center_matrix = Matrix.Translation(-center)
    root.matrix_world = scale_matrix @ center_matrix @ rotation
    bpy.context.view_layer.update()


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    forward = target - Vector(camera.location)
    if forward.length == 0:
        forward = Vector((0.0, 0.0, -1.0))
    forward.normalize()
    up = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(up)) > 0.999:
        up = Vector((0.0, 1.0, 0.0))
    right = forward.cross(up)
    right.normalize()
    corrected_up = right.cross(forward)
    corrected_up.normalize()
    rotation_matrix = Matrix(
        (
            (right.x, corrected_up.x, (-forward).x),
            (right.y, corrected_up.y, (-forward).y),
            (right.z, corrected_up.z, (-forward).z),
        )
    )
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = rotation_matrix.to_euler("XYZ")


def configure_camera(
    camera: bpy.types.Object,
    *,
    camera_distance: float,
    pitch_deg: float,
    azimuth_deg: float,
) -> None:
    camera.data.type = "PERSP"
    camera.data.lens = 58.0
    camera.data.sensor_width = 36.0
    pitch = math.radians(float(pitch_deg))
    azimuth = math.radians(float(azimuth_deg))
    horizontal_distance = float(camera_distance) * math.sin(pitch)
    camera.location = (
        horizontal_distance * math.sin(azimuth),
        horizontal_distance * math.cos(azimuth),
        float(camera_distance) * math.cos(pitch),
    )
    look_at(camera, Vector((0.0, 0.0, 0.0)))


def add_lighting() -> None:
    ambient = bpy.context.scene.world
    if ambient is not None:
        ambient.color = (0.97, 0.98, 0.99)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 4.0))
    key = bpy.context.active_object
    key.name = "Preview_Key_Light"
    key.data.energy = 650.0
    key.data.size = 4.0
    look_at(key, Vector((0.0, 0.0, 0.0)))

    bpy.ops.object.light_add(type="AREA", location=(-3.5, 3.0, 2.0))
    fill = bpy.context.active_object
    fill.name = "Preview_Fill_Light"
    fill.data.energy = 180.0
    fill.data.size = 5.0
    look_at(fill, Vector((0.0, 0.0, 0.0)))


def main() -> None:
    if bpy is None or Matrix is None or Vector is None:
        raise ImportError("render_upright_previews_blender.py must be run inside Blender.")
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = reset_scene()
    configure_render(int(args.resolution))
    load_object(args.mesh_path.resolve())
    assign_default_material()
    root = create_preview_root()
    import_basis = import_basis_to_blender(args.mesh_path.resolve())
    add_lighting()
    configure_camera(
        camera,
        camera_distance=float(args.camera_distance),
        pitch_deg=float(args.pitch_deg),
        azimuth_deg=float(args.azimuth_deg),
    )

    up_dirs = [canonicalize_up_dir(token) for token in str(args.up_dirs).split(",") if token.strip()]
    for up_dir in up_dirs:
        orient_and_normalize(root, up_dir=up_dir, import_basis=import_basis)
        image_path = output_dir / f"up_{up_dir_slug(up_dir)}.png"
        bpy.context.scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered upright preview for {up_dir}: {image_path}")


if __name__ == "__main__":
    main()
