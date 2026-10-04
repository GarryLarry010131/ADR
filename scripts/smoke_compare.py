"""Compare the per-item margins of a pair-set run with a reference margins file.

    python scripts/smoke_compare.py RUN.items.jsonl REFERENCE.jsonl <reference key>

The reference holds one json object per line with "ix" and "m": {<key>: margin}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adr import fair  # noqa: E402

def read(path):
    return {r["ix"]: r for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}


ours, ref = read(sys.argv[1]), read(sys.argv[2])
key = sys.argv[3]
common = sorted(set(ours) & set(ref))
drift = max(abs((ours[i]["scores"][0] - ours[i]["scores"][1]) - ref[i]["m"][key]) for i in common)
flips = sum(fair(ours[i]["scores"][0] - ours[i]["scores"][1]) != fair(ref[i]["m"][key]) for i in common)
print("items compared: %d | max |drift| = %.2e | changed decisions = %d" % (len(common), drift, flips))
