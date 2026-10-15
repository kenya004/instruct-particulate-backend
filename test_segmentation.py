import json
from app import InstructParticulateApp

def take_last(generator):
    last = None
    for item in generator:
        last = item
        print("progress:", type(item))
    return last

mesh_path = "examples/000.glb"

tree = {
    "links": [
        {"id": 0, "name": "base"},
        {"id": 1, "name": "handle"},
    ],
    "joints": [
        {"parent": 0, "child": 1, "type": "revolute"},
    ],
}

prompt_json = {"prompts": []}

app = InstructParticulateApp()

result = take_last(app.predict_segmentation_payload(
    mesh_path,
    None,
    json.dumps(tree),
    json.dumps(prompt_json),
    "+Z",
    20000,
    3,
    20000,
    50,
    True,
    True,
    1.0,
))

print("RESULT:")
print(result)
