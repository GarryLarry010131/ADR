"""Run readouts on the evaluation sets from a YAML configuration.

    python scripts/run.py configs/adr_clip_b16.yaml [--sets ...] [--passes real null] [--limit N]

Each (set, readout, pass) writes <out>/<set>__<readout>[__null].json and an items.jsonl with every per-item score
or reply; when both passes ran, <out>/summary.json holds accuracy, null accuracy and grounded gain in percent.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adr import DEPTH, LEX, LEX_VSR, parse  # noqa: E402
from evalkit import grounded_gain, load_kway, load_valse, load_vgrel, load_vgrel_verbs, load_vsr, load_whatsup_b, run_set  # noqa: E402
from readouts import build  # noqa: E402

STRATA = {"vgrel": "stratum", "vgrel_verbs": "stratum", "whatsup_b": "task"}
LEXICON_OF = {"vgrel": "main", "vgrel_verbs": "main", "coco_spatial": "main", "whatsup_a": "main", "whatsup_b": "depth",
              "vsr": "vsr", "valse": "main"}
LEXICONS = {"main": LEX, "depth": dict(LEX, **DEPTH), "vsr": LEX_VSR}
_VG = {}


def covers_main(cap):
    return bool(parse(cap, LEX)[1])


def load_items(name, cfg):
    if name in ("vgrel", "vgrel_verbs"):
        if "items" not in _VG:
            _VG["items"] = load_vgrel(cfg.get("cache_dir") or os.environ.get("ADR_CACHE_DIR"), covers=covers_main)
        return _VG["items"] if name == "vgrel" else load_vgrel_verbs(_VG["items"], covers_main)
    if name == "valse":
        return load_valse(cfg["valse_json"], cfg["swig_root"])
    if name in ("coco_spatial", "whatsup_a"):
        return load_kway(name, cfg.get("data_root", "data"))
    if name == "whatsup_b":
        lex = LEXICONS["depth"]
        items, excluded = load_whatsup_b(cfg.get("data_root", "data"), lambda c: parse(c, lex))
        print("whatsup_b: %d pairs, %d items excluded by the parser" % (len(items), excluded), flush=True)
        return items
    if name == "vsr":
        return load_vsr(cfg.get("vsr_root", "data/vsr"))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--passes", nargs="*", choices=["real", "null"])
    ap.add_argument("--limit", type=int, default=0, help="first items of each set only")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sets = args.sets or cfg.get("sets", ["vgrel", "coco_spatial", "whatsup_a", "whatsup_b", "vsr"])
    passes = ["null" if p is None else str(p) for p in (args.passes or cfg.get("passes", ["real", "null"]))]  # YAML reads a bare null as None
    assert all(p in ("real", "null") for p in passes), "passes must be real and/or null, got %r" % passes
    out = cfg.get("out", "results")
    readouts = [build(r, args.device) for r in cfg["readouts"]]
    summary = {}
    for name in sets:
        items = load_items(name, cfg)
        lex = LEXICONS[LEXICON_OF[name]]
        covers = lambda c, lex=lex: bool(parse(c, lex)[1])
        for ro, tag in readouts:
            if hasattr(ro, "use_lexicon"):
                ro.use_lexicon(LEXICON_OF[name])
            res = {}
            for p in passes:
                res[p] = run_set(name, items, ro, p == "null", out, tag, covers, STRATA.get(name), args.limit)
                print(json.dumps(res[p]["strata"]), flush=True)
            if "real" in res and "null" in res:
                summary.setdefault(tag, {})[name] = grounded_gain(res["real"], res["null"])
    if summary:
        Path(out, "summary.json").write_text(json.dumps(summary, indent=2))
        for tag, per_set in summary.items():
            print("\n%s" % tag)
            for name, strata in per_set.items():
                for st, v in strata.items():
                    print("  %-14s %-12s n=%-5s acc %6.2f  null %6.2f  gain %+.1f" % (name, st, v["n"], v["accuracy"], v["null"], v["gain"]))


if __name__ == "__main__":
    main()
