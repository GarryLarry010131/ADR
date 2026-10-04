"""One readout on one set, real pass or derangement pass, under the paper's rules.

Scoring readouts give one scalar per caption; their `kind` decides a true/false statement: "signed" by the sign,
"swap" by the sign of score minus the subject-object swapped score, "score" by an oracle threshold on the labels.
Chat readouts return a credit directly. `covers` is the set's coverage test and applies to every readout alike:
k-way and true/false sets count covered items only, pair sets count every item with uncovered captions at D = 0.
"""
import json
from pathlib import Path

from .rules import fair, kway_fair, null_assignment, oracle_accuracy, sign_fair
from .sets import K, KIND


def run_set(name, items, readout, derang, out_dir, tag, covers, strata_key=None, limit=0):
    kind = KIND[name]
    if limit:
        items = items[:limit]
    sig = list(range(len(items)))
    fix = 0
    if derang:
        sig, fix = null_assignment(name, items, K[name])
    is_chat = hasattr(readout, "choose")
    rows = []
    for j, it in enumerate(items):
        img = items[sig[j]]["load"]()
        caps = it["captions"]
        row = {"ix": j, "correct": it["correct"], "cov": all(covers(c) for c in caps)}
        for k in ("rel", "stratum", "task", "split"):
            if k in it:
                row[k] = it[k]
        if kind == "truefalse":
            if is_chat:
                row["credit"], row["log"] = readout.yesno(img, caps[0], it["correct"])
            else:
                row["score"] = readout.scores(img, caps)[0]
                if readout.kind == "swap":
                    row["score_swapped"] = readout.swap_score(img, caps[0])
        else:
            if is_chat:
                row["credit"], row["log"] = readout.choose(img, caps, it["correct"])
            else:
                row["scores"] = readout.scores(img, caps)
                row["credit"] = (fair(row["scores"][0] - row["scores"][1]) if kind == "pair"
                                 else kway_fair(row["scores"], it["correct"]))
        rows.append(row)
        if (j + 1) % 100 == 0:
            done = [r for r in rows if r["cov"] and "credit" in r]
            print("%s %d/%d%s | covered %d | accuracy so far %.4f" % (
                name, j + 1, len(items), " (null)" if derang else "", len(done),
                sum(r["credit"] for r in done) / max(len(done), 1)), flush=True)
    summary = {"set": name, "readout": tag, "derang": derang, "K": K[name] if derang else None, "fixups": fix, "n": len(items)}
    if kind == "truefalse":
        summary["strata"] = {}
        for st in ("horizontal", "vertical"):
            sel = [r for r in rows if r["stratum"] == st and r["cov"]]
            ent = {"n_covered": len(sel)}
            if sel:
                if is_chat:
                    ent["accuracy"] = round(sum(r["credit"] for r in sel) / len(sel), 4)
                elif readout.kind == "signed":
                    ent["accuracy"] = round(sum(sign_fair(r["score"], r["correct"]) for r in sel) / len(sel), 4)
                elif readout.kind == "swap":
                    ent["accuracy"] = round(sum(sign_fair(r["score"] - r["score_swapped"], r["correct"]) for r in sel) / len(sel), 4)
                else:
                    ent["accuracy"] = round(oracle_accuracy([r["score"] for r in sel], [r["correct"] for r in sel]), 4)
                    ent["rule"] = "oracle threshold on the evaluation labels"
            summary["strata"][st] = ent
    else:
        groups = {"all": rows}
        if strata_key:
            for r in rows:
                groups.setdefault(r.get(strata_key), []).append(r)
        summary["strata"] = {}
        for g, sel in groups.items():
            cov = [r for r in sel if r["cov"]]
            scored = sel if kind == "pair" else cov
            summary["strata"][str(g)] = {"n": len(sel), "n_covered": len(cov), "n_scored": len(scored),
                                         "accuracy": round(sum(r["credit"] for r in scored) / max(len(scored), 1), 4) if scored else None}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "%s__%s%s" % (name, tag, "__null" if derang else "")
    (out_dir / (stem + ".json")).write_text(json.dumps(summary, indent=2))
    with open(out_dir / (stem + ".items.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return summary


def grounded_gain(real, null):
    """Accuracy, null accuracy and gain in percent per stratum."""
    out = {}
    for st, ent in real["strata"].items():
        a, n = ent.get("accuracy"), null["strata"].get(st, {}).get("accuracy")
        if a is None or n is None:
            continue
        out[st] = {"n": ent.get("n_scored", ent.get("n_covered")), "accuracy": round(100 * a, 2), "null": round(100 * n, 2), "gain": round(100 * (a - n), 1)}
    return out
