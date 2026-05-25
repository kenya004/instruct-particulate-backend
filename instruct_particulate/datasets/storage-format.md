# Dataset Storage Format

Each dataset is stored under a single root directory, referred to below as `dataset_root_dir`.
Every data point in the dataset is stored as its own `.npz` file somewhere under that root.

## Directory Layout

The dataset layout is:

```text
dataset_root_dir/
  sample_000001.npz
  sample_000002.npz
  ...
```

File names are not semantically important as long as each file is a valid data point and the
data loader knows how to enumerate them.

## Per-Example `.npz` Format

Each `.npz` file stores one articulated object instance with the following keys:

### `points`

- Type: NumPy array
- Shape: `(N, 3)`
- Meaning: 3D point coordinates

### `normals`

- Type: NumPy array
- Shape: `(N, 3)`
- Meaning: surface normal for each point in `points`

### `link_ids`

- Type: NumPy array
- Shape: `(N,)`
- Meaning: per-point link assignment
- Convention: use `-1` for points without a part annotation

The `i`-th value in `link_ids` corresponds to the `i`-th point in `points` and `normals`.

### `sharp_point_flag`

- Type: boolean NumPy array
- Shape: `(N,)`
- Meaning: indicates whether each point was sampled from a sharp edge
- Convention: `True` for points sampled from sharp edges, `False` for points sampled uniformly from the surface

The `i`-th value in `sharp_point_flag` corresponds to the `i`-th point in `points`, `normals`, and
`link_ids`.

### `links`

- Type: JSON-encoded list of link records
- Meaning: metadata for all links referenced by `link_ids`

Each entry in `links` has the following structure:

```json
{
  "link_id": 1,
  "name": ["left_door", "door_panel"],
  "embedding": [[0.12, -0.34, 0.56], [0.78, -0.90, 0.12]]
}
```

Fields:

- `link_id`: integer link identifier used by `link_ids`
- `name`: either a single human-readable link name string or a list of
  alternative strings for the same link
- `embedding`: cached text embedding(s) for the link

Name and embedding formats:

- If `name` is a string, `embedding` must be a single `D`-dimensional vector.
- If `name` is a list of `M` strings, `embedding` must be an array of shape
  `(M, D)`, where row `i` matches `name[i]`.

During dataset loading, one name variant is sampled independently for each link
in each sample. The loader selects the embedding row with the same sampled
variant index.

### `joints`

- Type: optional JSON-encoded list of joint records
- Meaning: articulation graph connecting links in the object

Each entry in `joints` has the following structure:

```json
{
  "joint_id": 0,
  "parent_link_id": 0,
  "child_link_id": 1,
  "joint_description": "The door is hinged along its left vertical edge and swings outward to open relative to the base frame.",
  "is_revolute": true,
  "is_prismatic": false,
  "revolute_axis": [0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
  "prismatic_axis": null,
  "revolute_range": [-1.57, 1.57],
  "prismatic_range": null
}
```

Fields:

- `joint_id`: integer joint identifier
- `parent_link_id`: integer ID of the parent link
- `child_link_id`: integer ID of the child link
- `joint_description`: natural-language description of the joint motion
- `is_revolute`: whether the joint has a revolute motion component
- `is_prismatic`: whether the joint has a prismatic motion component
- `revolute_axis`: optional Plucker coordinate in `R^6` for the revolute axis
- `prismatic_axis`: optional Plucker coordinate in `R^6` for the prismatic axis
- `revolute_range`: optional range in `R^2` storing the low and high angular limits
- `prismatic_range`: optional range in `R^2` storing the low and high translational limits

If a motion component is unavailable, store its axis and range as `null` and set
its boolean flag to `false`.

If a revolute motion component is known but has no finite stored limits
(for example, a continuous joint), keep `is_revolute = true`, store the
revolute axis, and set `revolute_range = null`.

If `joints` is omitted entirely or stored as an empty list, the sample is
treated as a segmentation-only example with no kinematic structure.

Units:

- `revolute_range`: radians
- `prismatic_range`: length units in the same coordinate system as the geometry

When a joint has both motion components, the revolute and prismatic axes may be
different and are stored independently.

## Consistency Requirements

Each `.npz` file must satisfy the following:

- `points.shape[0] == normals.shape[0] == link_ids.shape[0]`
- `sharp_point_flag.shape[0] == points.shape[0]`
- Every non-negative value in `link_ids` must appear as a `link_id` in `links`
- Every `parent_link_id` and `child_link_id` in `joints` must refer to a valid `link_id` in `links`
- `link_id` values are unique within a sample
- `joint_id` values are unique within a sample
- If `name` is a list, it must be non-empty and every element must be a string
- Every link must define a cached `embedding`
- If `name` is a string, `embedding` is 1D
- If `name` is a list of length `M`, `embedding` has shape `(M, D)`
- All link embeddings within a sample share the same feature dimension `D`
- `is_revolute` and `is_prismatic` are booleans
- At least one of `is_revolute` or `is_prismatic` is `true`
- If not `null`, `revolute_axis` has length 6
- If not `null`, `prismatic_axis` has length 6
- If not `null`, `revolute_range` has length 2
- If not `null`, `prismatic_range` has length 2
- `is_revolute == true` allows `revolute_range` to be `null` for continuous or otherwise unbounded joints
- `revolute_range != null` implies `revolute_axis != null`
- `is_revolute == false` implies `revolute_axis` and `revolute_range` are `null`
- `is_prismatic == true` allows `prismatic_axis` and `prismatic_range` to be present independently
- `prismatic_range != null` implies `prismatic_axis != null`
- `is_prismatic == false` implies `prismatic_axis` and `prismatic_range` are `null`

If `joints` is omitted, the joint-related consistency requirements do not
apply for that sample.

## Serialization Notes

`links` and `joints` are structured metadata rather than dense numeric arrays. When saved inside an
`.npz`, they should be serialized in a JSON-compatible form, for example as UTF-8 JSON strings.

In other words, each sample file contains:

- dense NumPy arrays for geometry and per-point annotations
- JSON-serialized metadata for links
- optional JSON-serialized metadata for joints
