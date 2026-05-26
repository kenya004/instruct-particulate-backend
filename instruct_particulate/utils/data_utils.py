from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


CANONICAL_UP_DIRS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def canonicalize_up_dir(up_dir: str) -> str:
    """Normalize a user-provided up-direction token to one of ``CANONICAL_UP_DIRS``."""
    if not isinstance(up_dir, str):
        raise ValueError(f"Expected up direction as a string, got {type(up_dir).__name__}")
    token = up_dir.strip().upper()
    if token in {"X", "Y", "Z"}:
        token = f"+{token}"
    if token not in CANONICAL_UP_DIRS:
        raise ValueError(
            "Invalid up direction "
            f"{up_dir!r}. Expected one of {', '.join(CANONICAL_UP_DIRS)} "
            "(shorthand X/Y/Z is also accepted)."
        )
    return token


def up_dir_rotation_matrix(
    source_up_dir: str,
    target_up_dir: str = "+Z",
) -> np.ndarray:
    """Return the rotation matrix that maps ``source_up_dir`` to ``target_up_dir``."""
    canonical_source_up_dir = canonicalize_up_dir(source_up_dir)
    canonical_target_up_dir = canonicalize_up_dir(target_up_dir)
    rotations = {
        "+X": np.asarray([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "-X": np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
        "+Y": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "-Y": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32),
        "+Z": np.eye(3, dtype=np.float32),
        "-Z": np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32),
    }
    return (rotations[canonical_target_up_dir].T @ rotations[canonical_source_up_dir]).astype(
        np.float32,
        copy=False,
    )


def up_dir_rotation_matrix_to_z(up_dir: str) -> np.ndarray:
    """Return the rotation matrix that maps the declared up axis to ``+Z``."""
    return up_dir_rotation_matrix(up_dir, "+Z")


def reorient_mesh_to_z_up(
    mesh: trimesh.Trimesh,
    up_dir: str,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return a mesh copy rotated so its declared up direction becomes ``+Z``."""
    rotation = up_dir_rotation_matrix_to_z(up_dir)
    transformed_mesh = mesh.copy()
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transformed_mesh.apply_transform(transform)
    return transformed_mesh, rotation


def load_obj_raw_preserve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from an OBJ file while preserving vertex order.

    Args:
        path (Path): Path to the OBJ file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - vertices: Nx3 array of vertex positions
            - faces: Mx3 array of face indices (0-based)
    """
    verts, faces = [], []
    with path.open() as fh:
        for ln in fh:
            if ln.startswith('v '):   # keep order *exactly* as file
                _, x, y, z = ln.split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif ln.startswith('f '):
                toks = ln[2:].strip().split()
                if len(toks) == 3:
                    faces.append([int(t.split('/')[0]) - 1 for t in toks])
                else:
                    faces.append([int(t.split('/')[0]) - 1 for t in toks[:3]])
                    for i in range(2, len(toks) - 1):
                        faces.append([int(toks[0].split('/')[0]) - 1,
                                    int(toks[i].split('/')[0]) - 1,
                                    int(toks[i + 1].split('/')[0]) - 1])
    return np.asarray(verts, float), np.asarray(faces, int)


def load_trimesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh while preserving OBJ vertex order when possible."""
    path = Path(path)
    if path.suffix.lower() == ".obj":
        vertices, faces = load_obj_raw_preserve(path)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        mesh = trimesh.load(path, process=False, maintain_order=True)
        if isinstance(mesh, trimesh.Scene):
            transformed_geometry = []
            for node_name in mesh.graph.nodes_geometry:
                transform, geometry_name = mesh.graph[node_name]
                geometry = mesh.geometry[geometry_name].copy()
                geometry.apply_transform(transform)
                transformed_geometry.append(geometry)
            if not transformed_geometry:
                raise ValueError(f"Loaded scene from {path} does not contain any mesh geometry")
            mesh = trimesh.util.concatenate(tuple(transformed_geometry))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a trimesh.Trimesh from {path}, got {type(mesh).__name__}")
    if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Loaded mesh from {path} is empty")
    return mesh


def normalize_points_to_unit_extent(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center points at the bbox midpoint and scale by the max bbox extent."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("Cannot normalize an empty point set")

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    max_extent = float(extent.max())
    if max_extent <= 0.0:
        raise ValueError("Cannot normalize degenerate geometry with zero spatial extent")
    scale = 1.0 / max_extent
    normalized = (points - center) * scale
    return normalized.astype(np.float32, copy=False), center.astype(np.float32, copy=False), float(scale)


def normalize_mesh(
    mesh: trimesh.Trimesh,
) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
    """Return a normalized mesh copy plus the bbox center and scalar scale."""
    normalized_vertices, center, scale = normalize_points_to_unit_extent(mesh.vertices)
    normalized_mesh = mesh.copy()
    normalized_mesh.vertices = normalized_vertices
    return normalized_mesh, center, scale


def sharp_sample_pointcloud(mesh, num_points: int = 8192):
    V = mesh.vertices
    N = mesh.face_normals
    F = mesh.faces
    
    edge_to_faces = {}
    
    for face_idx in range(len(F)):
        face = F[face_idx]
        edges = [
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0])
        ]
        
        for edge in edges:
            edge_key = tuple(sorted(edge))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(face_idx)
    
    sharp_edges = []
    sharp_edge_normals = []
    sharp_edge_faces = []
    cos_30 = np.cos(np.radians(30))  # ≈ 0.866
    cos_150 = np.cos(np.radians(150))  # ≈ -0.866
    
    for edge_key, face_indices in edge_to_faces.items():
        if len(face_indices) < 2:
            continue
        
        is_sharp = False
        for i in range(len(face_indices)):
            for j in range(i + 1, len(face_indices)):
                n1 = N[face_indices[i]]
                n2 = N[face_indices[j]]
                dot_product = np.dot(n1, n2)
                
                if cos_150 < dot_product < cos_30 and np.linalg.norm(n1) > 1e-8 and np.linalg.norm(n2) > 1e-8:
                    is_sharp = True
                    sharp_edges.append(edge_key)
                    averaged_normal = (n1 + n2) / 2
                    sharp_edge_normals.append(averaged_normal)
                    sharp_edge_faces.append(face_indices)  # Store all adjacent faces
                    break
            if is_sharp:
                break
    
    edge_a = np.array([edge[0] for edge in sharp_edges], dtype=np.int32)
    edge_b = np.array([edge[1] for edge in sharp_edges], dtype=np.int32)
    sharp_edge_normals = np.array(sharp_edge_normals, dtype=np.float64)

    if len(sharp_edges) == 0:
        samples = np.zeros((0, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float64)
        edge_indices = np.zeros((0,), dtype=np.int32)
        vertex_ids_a = np.zeros((0,), dtype=np.int32)
        vertex_ids_b = np.zeros((0,), dtype=np.int32)
        return samples, normals, edge_indices, sharp_edge_faces, vertex_ids_a, vertex_ids_b

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num_points)
    w = np.random.rand(num_points, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = sharp_edge_normals[index]
    vertex_ids_a = edge_a[index]
    vertex_ids_b = edge_b[index]
    return samples, normals, index, sharp_edge_faces, vertex_ids_a, vertex_ids_b


def sample_points(mesh, num_points, sharp_point_ratio):
    """Sample exactly ``num_points`` from mesh using sharp edge and uniform sampling."""
    num_points_sharp_edges = int(num_points * sharp_point_ratio)
    num_points_uniform = num_points - num_points_sharp_edges
    points_sharp, normals_sharp, edge_indices, sharp_edge_faces, _, _ = sharp_sample_pointcloud(mesh, num_points_sharp_edges)

    # If no sharp edges were found, sample all points uniformly
    if len(points_sharp) == 0 and sharp_point_ratio > 0:
        print("Warning: No sharp edges found, sampling all points uniformly")
        num_points_uniform = num_points

    if num_points_uniform > 0:
        points_uniform, face_indices = mesh.sample(num_points_uniform, return_index=True)
        normals_uniform = mesh.face_normals[face_indices]
    else:
        points_uniform = np.zeros((0, 3), dtype=np.float64)
        normals_uniform = np.zeros((0, 3), dtype=np.float64)
        face_indices = np.zeros((0,), dtype=np.int32)

    points = np.concatenate([points_sharp, points_uniform], axis=0)
    normals = np.concatenate([normals_sharp, normals_uniform], axis=0)
    sharp_flag = np.concatenate([
        np.ones(len(points_sharp), dtype=np.bool_),
        np.zeros(len(points_uniform), dtype=np.bool_)
    ], axis=0)
    
    # For each sharp point, randomly select one of the adjacent faces from the edge
    sharp_face_indices = np.zeros(len(points_sharp), dtype=np.int32)
    for i, edge_idx in enumerate(edge_indices):
        adjacent_faces = sharp_edge_faces[edge_idx]
        # Randomly select one of the adjacent faces
        sharp_face_indices[i] = np.random.choice(adjacent_faces)
    
    face_indices = np.concatenate([
        sharp_face_indices,
        face_indices
    ], axis=0)
    
    return points, normals, sharp_flag, face_indices


def sample_points_per_face(mesh, num_points_per_face):
    """Sample uniformly inside every face with an equal point count per face."""
    num_points_per_face = int(num_points_per_face)
    if num_points_per_face <= 0:
        raise ValueError(f"num_points_per_face must be positive, got {num_points_per_face}")

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape[0] == 0:
        raise ValueError("Cannot sample per-face query points from a mesh with no faces")

    face_indices = np.repeat(
        np.arange(faces.shape[0], dtype=np.int64),
        num_points_per_face,
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = vertices[faces[face_indices]]

    r1 = np.random.random((face_indices.shape[0], 1))
    r2 = np.random.random((face_indices.shape[0], 1))
    sqrt_r1 = np.sqrt(r1)
    barycentric = np.concatenate(
        (
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    points = (triangles * barycentric[:, :, None]).sum(axis=1)
    normals = np.asarray(mesh.face_normals, dtype=np.float32)[face_indices]
    sharp_flag = np.zeros((face_indices.shape[0],), dtype=np.bool_)
    return (
        points.astype(np.float32, copy=False),
        normals.astype(np.float32, copy=False),
        sharp_flag,
        face_indices,
    )
