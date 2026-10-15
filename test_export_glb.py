import json
from pathlib import Path
from app import InstructParticulateApp


def take_last(generator):
    last = None
    for item in generator:
        last = item
        print("status:", item[1])
    return last


mesh_path = "examples/000.glb"
up_dir = "+Z"

tree = {
    "links": [
        {"id": 0, "name": "base"},
        {"id": 1, "name": "handle"},
    ],
    "joints": [
        {"parent": 0, "child": 1, "type": "revolute"},
    ],
}

app = InstructParticulateApp()

seg_payload, status, _ = take_last(app.predict_segmentation_payload(
    mesh_path,
    None,
    json.dumps(tree),
    json.dumps({"prompts": []}),
    up_dir,
    20000,
    3,
    20000,
    50,
    True,
    True,
    1.0,
))

print("segmentation status:", status)

motion_request, visualization_path, status, _ = app.postprocess_segmentation_payload(seg_payload)

print("postprocess status:", status)
print("visualization_path:", visualization_path)

motion_payload, status, _ = take_last(app.predict_motion_payload(motion_request))

print("motion status:", status)

animated_glb, parts_glb, output_dir, final_status, _ = app.finish_predict_payload(motion_payload)

print("final status:", final_status)
print("animated_glb:", animated_glb)
print("parts_glb:", parts_glb)
print("output_dir:", output_dir)

output_dir = Path(output_dir)
print("files:")
for path in sorted(output_dir.rglob("*")):
    if path.is_file():
        print(path)