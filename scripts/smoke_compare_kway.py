"""Compare per-caption scores (or chat credits) of a k-way run with a reference record.

    python scripts/smoke_compare_kway.py RUN.items.jsonl REFERENCE.jsonl <dotted field, e.g. scores.itm or gen>
"""
import json
import sys
from pathlib import Path


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def read(path):
    return {r["ix"]: r for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}


ours, ref = read(sys.argv[1]), read(sys.argv[2])
field = sys.argv[3]
common = sorted(set(ours) & set(ref))
if common and "scores" in ours[common[0]]:
    drift = max(abs(a - b) for i in common for a, b in zip(ours[i]["scores"], get(ref[i], field)))
    flips = sum((max(range(len(ours[i]["scores"])), key=ours[i]["scores"].__getitem__)
                 != max(range(len(get(ref[i], field))), key=get(ref[i], field).__getitem__)) for i in common)
    print("items compared: %d | max |drift| = %.2e | changed argmax = %d" % (len(common), drift, flips))
else:
    diff = sum(abs(ours[i]["credit"] - get(ref[i], field)) > 1e-9 for i in common)
    print("items compared: %d | credits that differ = %d" % (len(common), diff))
