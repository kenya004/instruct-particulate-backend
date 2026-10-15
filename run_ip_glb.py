import argparse
import json
from pathlib import Path

from app import InstructParticulateApp


def take_last(generator):
    last = None
    for item in generator:
        last = item
        print("status:", item[1])
    return last


def build_tree(part_names, joint_type):
    return {
        "links": [
            {"id": i, "name": name}
            for i, name in enumerate(part_names)
        ],
        "joints": [
            {
                "parent": 0,
                "child": i,
                "type": joint_type,
            }
            for i in range(1, len(part_names))
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Input .glb path")
    parser.add_argument("--parts", required=True, help="Comma-separated part names")
    parser.add_argument("--joint-type", default="revolute", choices=["revolute", "prismatic"])
    parser.add_argument("--up-dir", default="+Z")
    parser.add_argument("--num-query-points", type=int, default=20000)
    parser.add_argument("--query-batch-size", type=int, default=20000)
    args = parser.parse_args()

    part_names = [p.strip() for p in args.parts.split(",") if p.strip()]
    if len(part_names) < 2:
        raise ValueError("At least two part names are required.")

    tree = build_tree(part_names, args.joint_type)

    app = InstructParticulateApp()

    seg_payload, status, _ = take_last(app.predict_segmentation_payload(
        args.mesh,
        None,
        json.dumps(tree),
        json.dumps({"prompts": []}),
        args.up_dir,
        args.num_query_points,
        3,
        args.query_batch_size,
        50,
        True,
        True,
        1.0,
    ))

    print("segmentation:", status)

    motion_request, visualization_path, status, _ = app.postprocess_segmentation_payload(seg_payload)
    print("postprocess:", status)
    print("visualization:", visualization_path)

    motion_payload, status, _ = take_last(app.predict_motion_payload(motion_request))
    print("motion:", status)

    animated_glb, parts_glb, output_dir, final_status, _ = app.finish_predict_payload(motion_payload)

    print("final:", final_status)
    print("animated_glb:", animated_glb)
    print("parts_glb:", parts_glb)
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
