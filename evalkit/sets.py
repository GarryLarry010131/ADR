"""The evaluation sets and their loaders. Images are read from the benchmarks' own files under a data root; each
item carries a source-image identifier for the derangement null and a stratum or task label.

    vgrel         VG-Relation (ARO), fixed 6000-item sample, relation crops; strata leftright, vertical, other
    vgrel_verbs   the pairs of that sample outside the spatial lexicon, with their relation stratum (verb is reported)
    coco_spatial  COCO two-object pairs of the What'sUp release, two options
    whatsup_a     What'sUp Controlled Images A, four options
    whatsup_b     What'sUp Controlled Images B as binary pairs; tasks front_behind and left_right
    vsr           VSR, horizontal and vertical relations, one statement with a label
    valse         VALSE actant swap, caption against foil, pairs with at least two validating votes
"""
import io
import json
import os
import re
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]
MANIFESTS = HERE / "data" / "manifests"

# derangement offsets of the published runs
K = {"vgrel": 997, "vgrel_verbs": 997, "coco_spatial": 101, "whatsup_a": 101, "whatsup_b": 997, "vsr": 101, "valse": 101}
KIND = {"vgrel": "pair", "vgrel_verbs": "pair", "coco_spatial": "kway", "whatsup_a": "kway", "whatsup_b": "pair",
        "vsr": "truefalse", "valse": "pair"}

LR_RELS = {"to the left of", "to the right of"}
HF_REPO = "datasets--gowitheflow--ARO-Visual-Relation"
VSR_HORIZ = {"left of", "right of", "at the left side of", "at the right side of"}
VSR_VERT = {"above", "below", "under", "over", "on top of", "beneath", "on"}
VSR_SPLITS = [("train", "random_train.jsonl"), ("dev", "random_dev.jsonl"), ("test", "random_test.jsonl")]
WUB_OPP = {"in front of": "behind", "behind": "in front of", "to the left of": "to the right of", "to the right of": "to the left of"}
VG_SYMM = {"near", "next to", "by", "with", "beside", "at", "around", "close to", "standing next to", "standing by",
           "sitting next to", "walking next to", "lying next to", "standing near", "in between"}
VG_CONTAIN = {"in", "inside"}
VG_GEN = {"of"}
VG_SPATIAL_LEFTOVER = {"to the left of", "to the right of", "on"}


def _vgrel_crop(raw, bb):
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if bb and bb["w"] >= 8 and bb["h"] >= 8:
        img = img.crop((bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]))
    return img


def load_vgrel(cache_dir=None, n_per=6000, covers=None):
    """The 6000-item pick in pick order, drawn by numpy default_rng(0) over the parquet rows in shard order.
    With `covers` (the main-lexicon coverage test) the stratum is leftright, vertical (both captions covered,
    relation not left/right) or other."""
    import numpy as np
    import pyarrow.parquet as pq
    cache_dir = cache_dir or os.environ.get("ADR_CACHE_DIR") or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    hub = Path(cache_dir) / "hub" if (Path(cache_dir) / "hub").exists() else Path(cache_dir)
    snap = next((hub / HF_REPO).glob("snapshots/*/data"))
    files = sorted(snap.glob("*.parquet"))
    counts = [pq.read_metadata(f).num_rows for f in files]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(sum(counts), size=min(n_per, sum(counts)), replace=False))
    offs = np.cumsum([0] + counts)
    items = []
    for si, f in enumerate(files):
        loc = pick[(pick >= offs[si]) & (pick < offs[si + 1])] - offs[si]
        if len(loc) == 0:
            continue
        tbl = pq.read_table(f)
        for it in tbl.take(loc).to_pylist():
            im = it["image"] if isinstance(it.get("image"), dict) else {"bytes": it["bytes"], "path": ""}
            raw, bb = im["bytes"], it.get("bbox")
            rel = it.get("relation_name", "")
            caps = [it["true_caption"], it["false_caption"]]
            if rel in LR_RELS:
                st = "leftright"
            elif covers is not None and covers(caps[0]) and covers(caps[1]):
                st = "vertical"
            else:
                st = "other"
            items.append({"image": im.get("path") or "", "captions": caps, "correct": 0, "rel": rel, "bbox": bb,
                          "stratum": st, "load": (lambda raw=raw, bb=bb: _vgrel_crop(raw, bb))})
        del tbl
    assert all(it["image"] for it in items), "VG-Relation rows need their source-image path for the null"
    return items


def _anchor_parse(cap, rel, nlp, stop):
    """Subject and object word indices around the relation string, or None."""
    doc = nlp(cap.lower())
    toks = [t for t in doc if t.is_alpha and len(t.text) > 2 and t.text not in stop][:12]
    if not toks:
        return None
    idx = {t.i: k for k, t in enumerate(toks)}
    mt = re.search(r"(?<![a-z])" + re.escape(rel.lower()) + r"(?![a-z])", doc.text)
    if mt is None:
        return None
    s0, s1 = mt.span()
    subj = obj = None
    for t in doc:
        if t.idx + len(t.text) <= s0 and t.pos_ in ("NOUN", "PROPN") and t.i in idx:
            subj = t
        if t.idx >= s1 and t.pos_ in ("NOUN", "PROPN") and t.i in idx and obj is None:
            obj = t
    if subj is None or obj is None or subj.i == obj.i:
        return None
    return idx[subj.i], idx[obj.i]


def _vg_stratum(rel):
    if "front of" in rel or "behind" in rel:
        return "depth"
    if rel in VG_SYMM:
        return "symm"
    if rel in VG_CONTAIN:
        return "contain"
    if rel in VG_GEN:
        return "gen"
    if rel in VG_SPATIAL_LEFTOVER:
        return "spatial_leftover"
    return "verb"


def load_vgrel_verbs(vgrel_items, covers):
    """Pairs of the VG-Relation sample with neither caption covered by the spatial lexicon and both captions
    resolving around the relation word. The whole list is kept because the published null was drawn over it."""
    from adr.lexicon import STOP, nlp
    items = []
    for it in vgrel_items:
        rel = it["rel"]
        if covers(it["captions"][0]) or covers(it["captions"][1]):
            continue
        if _anchor_parse(it["captions"][0], rel, nlp(), STOP) is None or _anchor_parse(it["captions"][1], rel, nlp(), STOP) is None:
            continue
        rec = dict(it)
        rec["stratum"] = _vg_stratum(rel)
        items.append(rec)
    return items


def load_kway(name, data_root):
    spec = json.loads((MANIFESTS / (name + ".json")).read_text())
    assert spec["name"] == name
    items = []
    for it in spec["items"]:
        p = str(Path(data_root, it["image"]))
        items.append({"image": p, "captions": list(it["captions"]), "correct": int(it["correct"]),
                      "load": (lambda p=p: Image.open(p).convert("RGB"))})
    return items


def _wub_relation(cap):
    c = " " + cap.lower() + " "
    for ph in ("in front of", "behind", "to the left of", "to the right of"):
        if " " + ph + " " in c:
            return ph
    raise ValueError("unclassifiable caption: %s" % cap)


def load_whatsup_b(data_root, parse):
    """Binary pairs: the true caption against the opposite relation with the same argument order. `parse` must
    cover the depth phrases; pairs whose captions do not resolve to the same subject and object are excluded."""
    spec = json.loads((MANIFESTS / "whatsup_b.json").read_text())
    items, excluded = [], 0
    for it in spec["items"]:
        caps = it["captions"]
        rels = [_wub_relation(c) for c in caps]
        rt = rels[it["correct"]]
        ct, cf = caps[it["correct"]], caps[rels.index(WUB_OPP[rt])]
        st, et = parse(ct)
        sf, ef = parse(cf)
        if not et or not ef or (st[et[0][0]], st[et[0][1]]) != (sf[ef[0][0]], sf[ef[0][1]]):
            excluded += 1
            continue
        p = str(Path(data_root, it["image"]))
        items.append({"image": p, "captions": [ct, cf], "correct": 0, "rel": rt,
                      "task": "front_behind" if rt in ("in front of", "behind") else "left_right",
                      "load": (lambda p=p: Image.open(p).convert("RGB"))})
    return items, excluded


def load_vsr(vsr_root):
    items = []
    for split, fn in VSR_SPLITS:
        for line in Path(vsr_root, fn).read_text().splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            rel = it["relation"]
            if rel in VSR_HORIZ or rel in VSR_VERT:
                p = str(Path(vsr_root, "images", it["image"]))
                items.append({"image": p, "captions": [it["caption"]], "correct": int(it["label"]), "rel": rel,
                              "stratum": "horizontal" if rel in VSR_HORIZ else "vertical", "split": split,
                              "load": (lambda p=p: Image.open(p).convert("RGB"))})
    return items


def load_valse(valse_json, swig_root):
    d = json.loads(Path(valse_json).read_text())
    items = []
    for k in sorted(d.keys()):
        it = d[k]
        mt = it.get("mturk", {})
        if not (isinstance(mt, dict) and mt.get("caption", 0) >= 2):
            continue
        p = str(Path(swig_root, it["image_file"]))
        items.append({"image": p, "captions": [it["caption"], it["foil"]], "correct": 0, "key": k, "stratum": "actant_swap",
                      "load": (lambda p=p: Image.open(p).convert("RGB"))})
    return items
