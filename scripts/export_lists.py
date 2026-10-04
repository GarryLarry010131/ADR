"""Export every evaluation set as data/items/<set>.jsonl: position, source image, captions, correct index or label,
subset label, coverage flag and the derangement partner of each item.

    python scripts/export_lists.py --data-root DATA --vsr-root DATA/vsr --valse-json DATA/valse/actant-swap.json --swig-root DATA/swig/images_512
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adr import DEPTH, LEX, LEX_VSR, parse  # noqa: E402
from evalkit import K, load_kway, load_valse, load_vgrel, load_vgrel_verbs, load_vsr, load_whatsup_b  # noqa: E402
from evalkit.rules import null_assignment  # noqa: E402

COVER_LEX = {"vsr": LEX_VSR, "whatsup_b": dict(LEX, **DEPTH)}


def covers_main(cap):
    return bool(parse(cap, LEX)[1])


def write(name, items, out, extra=()):
    lex = COVER_LEX.get(name, LEX)
    sig, fix = null_assignment(name, items, K[name])
    with open(Path(out, name + ".jsonl"), "w") as f:
        for j, it in enumerate(items):
            rec = {"ix": j, "image": os.path.basename(it["image"]) if "/" in it["image"] or "\\" in it["image"] else it["image"],
                   "captions": it["captions"], "correct": it["correct"]}
            for k in ("stratum", "task", "rel", "split", "key") + tuple(extra):
                if k in it:
                    rec[k] = it[k]
            rec["covered"] = all(bool(parse(c, lex)[1]) for c in it["captions"])
            rec["partner"] = sig[j]
            f.write(json.dumps(rec) + "\n")
    print("%-13s %5d items, K=%d, fixups=%d" % (name, len(items), K[name], fix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--vsr-root", default="data/vsr")
    ap.add_argument("--valse-json", default=None)
    ap.add_argument("--swig-root", default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("ADR_CACHE_DIR"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "items"))
    ap.add_argument("--sets", nargs="*", default=["vgrel", "vgrel_verbs", "coco_spatial", "whatsup_a", "whatsup_b", "vsr", "valse"])
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    vg = None
    if "vgrel" in a.sets or "vgrel_verbs" in a.sets:
        vg = load_vgrel(a.cache_dir, covers=covers_main)
    if "vgrel" in a.sets:
        write("vgrel", vg, a.out, extra=("bbox",))
    if "vgrel_verbs" in a.sets:
        write("vgrel_verbs", load_vgrel_verbs(vg, covers_main), a.out)
    for name in ("coco_spatial", "whatsup_a"):
        if name in a.sets:
            write(name, load_kway(name, a.data_root), a.out)
    if "whatsup_b" in a.sets:
        lex = COVER_LEX["whatsup_b"]
        items, excluded = load_whatsup_b(a.data_root, lambda c: parse(c, lex))
        print("whatsup_b: %d pairs excluded by the parser" % excluded)
        write("whatsup_b", items, a.out)
    if "vsr" in a.sets:
        write("vsr", load_vsr(a.vsr_root), a.out)
    if "valse" in a.sets:
        assert a.valse_json and a.swig_root, "--valse-json and --swig-root are needed for VALSE"
        write("valse", load_valse(a.valse_json, a.swig_root), a.out)


if __name__ == "__main__":
    main()
