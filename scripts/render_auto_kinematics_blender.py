#!/usr/bin/env python3
"""Render auto-kinematics mesh views inside headless Blender."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:  # pragma: no cover - Blender-only runtime dependency
    bpy = None
    Matrix = None
    Vector = None

try:
    from PIL import Image
except ImportError:
    Image = None


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
else:  # pragma: no cover - used only in non-Blender unit tests
    IMPORT_FUNCTIONS = {}


def orient_saved_raster_payload(buffer: np.ndarray) -> np.ndarray:
    """Match the final saved image orientation for per-pixel render payloads."""
    oriented = np.flip(np.flip(np.asarray(buffer), axis=0), axis=1)
    return oriented.copy()


def orient_saved_render_image(image: Image.Image) -> Image.Image:
    """Apply the same output-space transform as the saved raster payloads."""
    oriented = image.transpose(Image.FLIP_TOP_BOTTOM)
    oriented = oriented.transpose(Image.FLIP_LEFT_RIGHT)
    return oriented


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--camera-distance", type=float, required=True)
    parser.add_argument("--pitch-deg", type=float, required=True)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--azimuths", type=float, nargs="+", required=True)
    return parser.parse_args(argv)


def load_object(mesh_path: Path) -> None:
    suffix = mesh_path.suffix.lower().lstrip(".")
    if suffix not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported mesh type for Blender auto-kinematics render: {mesh_path.suffix}")
    importer = IMPORT_FUNCTIONS[suffix]
    if suffix in {"glb", "gltf"}:
        try:
            result = importer(filepath=str(mesh_path), merge_vertices=False)
        except TypeError:
            result = importer(filepath=str(mesh_path))
    else:
        result = importer(filepath=str(mesh_path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender failed to import mesh: {mesh_path}")
    bpy.context.view_layer.update()
    for material in bpy.data.materials:
        material.use_backface_culling = True


def matrix_to_numpy(matrix: object) -> np.ndarray:
    return np.array([list(row) for row in matrix], dtype=np.float32)


def reset_scene() -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    return camera


def init_render(*, engine: str, resolution: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.dither_intensity = 0.0
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene.view_settings, "look"):
        try:
            scene.view_settings.look = "Medium High Contrast"
        except TypeError:
            pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if engine == "CYCLES":
        scene.cycles.device = "GPU"
        scene.cycles.samples = int(samples)
        scene.cycles.filter_type = "GAUSSIAN"
        scene.cycles.filter_width = 1.5
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 8
        scene.cycles.transmission_bounces = 8
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "denoiser"):
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.get_devices()
            available_types = {device.type for device in prefs.devices}
            for device_type in ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"):
                if device_type in available_types:
                    prefs.compute_device_type = device_type
                    break
        except Exception:
            pass


def init_lighting() -> None:
    def look_at_origin(light_obj: bpy.types.Object) -> None:
        target = Vector((0.0, 0.0, 0.0))
        direction = target - Vector(light_obj.location)
        if direction.length == 0:
            return
        direction.normalize()
        up = Vector((0.0, 0.0, -1.0))
        if abs(direction.dot(up)) > 0.999:
            up = Vector((0.0, 1.0, 0.0))
        right = direction.cross(up)
        right.normalize()
        corrected_up = right.cross(direction)
        corrected_up.normalize()
        rotation_matrix = Matrix(
            (
                (right.x, corrected_up.x, -direction.x),
                (right.y, corrected_up.y, -direction.y),
                (right.z, corrected_up.z, -direction.z),
            )
        ).to_4x4()
        light_obj.rotation_mode = "XYZ"
        light_obj.rotation_euler = (Matrix.Rotation(0, 4, "X") @ rotation_matrix).to_euler("XYZ")

    top_light = bpy.data.objects.new(
        "Top_Light",
        bpy.data.lights.new("Top_Light", type="AREA"),
    )
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 1500
    top_light.location = (0.0, 0.0, -8.0)
    top_light.data.size = 3.0
    top_light.rotation_mode = "XYZ"
    top_light.rotation_euler = (0.0, 0.0, 0.0)

    light_distance = 5.0
    light_height = 3.0
    light_energy_base = 1200.0
    light_size = 2.0
    side_specs = (
        ("Light_X_Pos", (light_distance, 0.0, -light_height), light_energy_base * 0.6),
        ("Light_X_Neg", (-light_distance, 0.0, -light_height), light_energy_base * 1.4),
        ("Light_Y_Pos", (0.0, light_distance, -light_height), light_energy_base),
        ("Light_Y_Neg", (0.0, -light_distance, -light_height), light_energy_base),
    )
    for name, location, energy in side_specs:
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, type="AREA"))
        bpy.context.collection.objects.link(light)
        light.data.energy = energy
        light.location = location
        light.data.size = light_size
        look_at_origin(light)


def get_scene_root_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects.values() if not obj.parent]


def get_scene_meshes() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]


def scene_bbox() -> tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes():
        found = True
        for coord in obj.bound_box:
            world_coord = obj.matrix_world @ Vector(coord)
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, world_coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, world_coord))
    if not found:
        raise RuntimeError("No mesh objects found in the scene")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> None:
    root_objects = get_scene_root_objects()
    if len(root_objects) > 1:
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        for obj in root_objects:
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1.0 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Camera"].parent = None
    bpy.context.view_layer.update()


def configure_camera(camera: bpy.types.Object, *, resolution: int) -> np.ndarray:
    camera.data.type = "PERSP"
    camera.data.lens_unit = "MILLIMETERS"
    camera.data.sensor_width = 36.0
    camera.data.sensor_height = 36.0
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.lens = 50.0

    fx = float(resolution) * float(camera.data.lens) / float(camera.data.sensor_width)
    fy = float(resolution) * float(camera.data.lens) / float(camera.data.sensor_height)
    cx = 0.5 * (float(resolution) - 1.0)
    cy = 0.5 * (float(resolution) - 1.0)
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def set_camera_pose(
    camera: bpy.types.Object,
    *,
    camera_distance: float,
    azimuth_deg: float,
    pitch_deg: float,
) -> None:
    azimuth = math.radians(float(azimuth_deg))
    pitch = math.radians(float(pitch_deg))
    horizontal_distance = float(camera_distance) * math.sin(pitch)
    y = horizontal_distance * math.cos(azimuth)
    x = horizontal_distance * math.sin(azimuth)
    z = float(camera_distance) * math.cos(pitch)
    camera.location = Vector((x, y, z))

    target = Vector((0.0, 0.0, 0.0))
    forward = target - camera.location
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
    world_up = Vector((0.0, 0.0, 1.0))
    if corrected_up.dot(world_up) < 0:
        right = -right
        corrected_up = -corrected_up

    rotation_matrix = Matrix(
        (
            (right.x, corrected_up.x, (-forward).x),
            (right.y, corrected_up.y, (-forward).y),
            (right.z, corrected_up.z, (-forward).z),
        )
    )
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = rotation_matrix.to_euler("XYZ")


def get_world_to_camera(camera: bpy.types.Object) -> np.ndarray:
    location, rotation = camera.matrix_world.decompose()[0:2]
    rotation_world_to_camera = rotation.to_matrix().transposed()
    translation_world_to_camera = -(rotation_world_to_camera @ location)
    return np.asarray(
        [
            list(rotation_world_to_camera[0]) + [float(translation_world_to_camera[0])],
            list(rotation_world_to_camera[1]) + [float(translation_world_to_camera[1])],
            list(rotation_world_to_camera[2]) + [float(translation_world_to_camera[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rasterize_view(
    camera: bpy.types.Object,
    *,
    intrinsic: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_to_world = matrix_to_numpy(camera.matrix_world)
    world_to_camera = get_world_to_camera(camera)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    width = int(resolution)
    height = int(resolution)

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    dir_x = (grid_x - cx) / fx
    dir_y = -(grid_y - cy) / fy
    directions_camera = np.stack(
        [dir_x, dir_y, np.ones_like(dir_x)],
        axis=-1,
    ).reshape(-1, 3)
    directions_camera /= np.linalg.norm(directions_camera, axis=-1, keepdims=True)

    right = camera_to_world[:3, 0]
    up = camera_to_world[:3, 1]
    forward = -camera_to_world[:3, 2]
    directions_world = (
        directions_camera[:, 0:1] * right[None, :]
        + directions_camera[:, 1:2] * up[None, :]
        + directions_camera[:, 2:3] * forward[None, :]
    )
    directions_world /= np.linalg.norm(directions_world, axis=-1, keepdims=True)
    camera_origin = np.asarray(camera_to_world[:3, 3], dtype=np.float64)

    flat_face_ids = np.full((height * width,), -1, dtype=np.int32)
    flat_hit_points = np.full((height * width, 3), np.nan, dtype=np.float32)
    flat_normals = np.full((height * width, 3), np.nan, dtype=np.float32)
    flat_depth = np.full((height * width,), np.nan, dtype=np.float32)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin_vec = Vector(camera_origin.tolist())
    for ray_idx, direction in enumerate(directions_world):
        hit, location, normal, _, _, _ = bpy.context.scene.ray_cast(
            depsgraph,
            origin_vec,
            Vector(direction.tolist()),
        )
        if not hit:
            continue
        flat_face_ids[ray_idx] = 0
        hit_point = np.asarray((location.x, location.y, location.z), dtype=np.float32)
        hit_normal = np.asarray((normal.x, normal.y, normal.z), dtype=np.float32)
        normal_norm = float(np.linalg.norm(hit_normal))
        if normal_norm > 1e-8:
            hit_normal = hit_normal / normal_norm
        flat_hit_points[ray_idx] = hit_point
        flat_normals[ray_idx] = hit_normal
        flat_depth[ray_idx] = float((world_to_camera[:3, :3] @ hit_point.astype(np.float64) + world_to_camera[:3, 3])[2])

    return (
        camera_to_world,
        world_to_camera,
        flat_face_ids.reshape(height, width),
        flat_hit_points.reshape(height, width, 3),
        flat_normals.reshape(height, width, 3),
        flat_depth.reshape(height, width),
    )


def main() -> None:
    if bpy is None or Matrix is None or Vector is None:
        raise ImportError("render_auto_kinematics_blender.py must be run inside Blender.")
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = reset_scene()
    init_render(
        engine=str(args.engine),
        resolution=int(args.resolution),
        samples=int(args.samples),
    )
    load_object(args.mesh_path.resolve())
    normalize_scene()
    init_lighting()
    intrinsic = configure_camera(
        camera,
        resolution=int(args.resolution),
    )

    for image_id, azimuth_deg in enumerate(args.azimuths):
        set_camera_pose(
            camera,
            camera_distance=float(args.camera_distance),
            azimuth_deg=float(azimuth_deg),
            pitch_deg=float(args.pitch_deg),
        )
        bpy.context.view_layer.update()
        image_path = output_dir / f"view_{image_id:03d}.png"
        camera_path = output_dir / f"view_{image_id:03d}_camera.npz"
        bpy.context.scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        (
            camera_to_world,
            world_to_camera,
            face_ids,
            hit_points,
            normals,
            depth,
        ) = rasterize_view(
            camera,
            intrinsic=intrinsic,
            resolution=int(args.resolution),
        )
        np.savez_compressed(
            camera_path,
            intrinsic=intrinsic.astype(np.float32),
            camera_to_world=camera_to_world.astype(np.float32),
            world_to_camera=world_to_camera.astype(np.float32),
            face_ids=orient_saved_raster_payload(face_ids).astype(np.int32),
            hit_points=orient_saved_raster_payload(hit_points).astype(np.float32),
            normals=orient_saved_raster_payload(normals).astype(np.float32),
            depth=orient_saved_raster_payload(depth).astype(np.float32),
            azimuth_deg=np.float32(azimuth_deg),
            elevation_deg=np.float32(90.0 - float(args.pitch_deg)),
            pitch_deg=np.float32(args.pitch_deg),
        )
        if Image is not None:
            with Image.open(image_path) as image:
                oriented = orient_saved_render_image(image)
                oriented.save(image_path)
        print(
            f"Rendered auto-kinematics Blender view {image_id + 1}/{len(args.azimuths)} "
            f"(azimuth={float(azimuth_deg):.1f}, pitch={float(args.pitch_deg):.1f})"
        )


if __name__ == "__main__":
    main()
