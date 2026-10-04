"""Build data/manifests/{whatsup_a,whatsup_b,coco_spatial}.json from the What'sUp release and COCO val2017, with
image paths relative to --data-root. Rebuilding reproduces the shipped lists in the same order.

    python scripts/build_manifests.py --whatsup DATA/whatsup --coco DATA/coco/val2017 --data-root DATA
"""
import argparse
import json
from pathlib import Path


def controlled(wu, json_name, img_dir, name, root, out):
    data = json.loads(Path(wu, json_name).read_text())
    items, missing = [], 0
    for d in data:
        p = Path(wu, img_dir, Path(d["image_path"]).name)
        if not p.exists():
            missing += 1
            continue
        items.append({"image": str(p.relative_to(root)).replace("\\", "/"), "captions": d["caption_options"], "correct": 0})
    Path(out, name + ".json").write_text(json.dumps({"name": name, "items": items}))
    print(name, "items:", len(items), "missing images:", missing)


def coco_two(wu, coco, root, out):
    data = json.loads(Path(wu, "coco_qa_two_obj.json").read_text())
    items, missing = [], 0
    for img_id, cap_true, cap_false in data:
        p = Path(coco, "%012d.jpg" % img_id)
        if not p.exists():
            missing += 1
            continue
        items.append({"image": str(p.relative_to(root)).replace("\\", "/"), "captions": [cap_true, cap_false], "correct": 0})
    Path(out, "coco_spatial.json").write_text(json.dumps({"name": "coco_spatial", "items": items}))
    print("coco_spatial items:", len(items), "missing images:", missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--whatsup", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "manifests"))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root).resolve()
    controlled(Path(a.whatsup).resolve(), "controlled_images_dataset.json", "controlled_images", "whatsup_a", root, a.out)
    controlled(Path(a.whatsup).resolve(), "controlled_clevr_dataset.json", "controlled_clevr", "whatsup_b", root, a.out)
    coco_two(Path(a.whatsup).resolve(), Path(a.coco).resolve(), root, a.out)
